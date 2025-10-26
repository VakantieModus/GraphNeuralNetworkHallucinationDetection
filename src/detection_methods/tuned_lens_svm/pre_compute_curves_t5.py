#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import logging
import re
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from tuned_lens import TunedLens

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

_PIDX_FILE_RE = re.compile(r"_p(\d{5})_l_out-")  # matches ..._p00001_l_out-123.npy


def choose_compute_dtype(device: str) -> torch.dtype:
    if device.startswith("cuda"):
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


@torch.no_grad()
def next_token_ids_from_final_layer_all_torch(
    hLT: np.ndarray,  # (L, T, H) numpy
    WU: torch.Tensor,  # (V, H) torch on device (compute dtype)
    device: str,
    amp: bool = True,
) -> np.ndarray:
    L, T, H = hLT.shape
    compute_dtype = WU.dtype
    h_final = torch.from_numpy(hLT[-1]).to(device=device, dtype=compute_dtype)  # (T,H)
    ctx = (
        torch.autocast(device_type="cuda", dtype=compute_dtype)
        if (amp and device == "cuda")
        else nullcontext()
    )
    with ctx:
        logits = h_final @ WU.T  # (T,V)
        tok_ids = torch.argmax(logits, dim=1).to(torch.int32).cpu().numpy()
    return tok_ids


@torch.no_grad()
def logit_lens_curves_all_torch(
    hLT: np.ndarray,  # (L, T, H) numpy
    WU: torch.Tensor,  # (V, H) on device
    tok_ids: np.ndarray,  # (T,)
    device: str,
    token_batch: int = 64,
    temperature: float | None = None,
    amp: bool = True,
) -> np.ndarray:
    L, T, H = hLT.shape
    compute_dtype = WU.dtype
    out = np.zeros((T, L), dtype=np.float32)
    tok_ids_t = torch.from_numpy(tok_ids).to(device=device)

    for l in range(L):  # noqa
        Hl_t = torch.from_numpy(hLT[l]).to(device=device, dtype=compute_dtype)  # (T,H)
        for start in range(0, T, token_batch):
            end = min(start + token_batch, T)
            H_chunk = Hl_t[start:end]  # (Tb,H)
            ids_chunk = tok_ids_t[start:end]  # (Tb,)
            ctx = (
                torch.autocast(device_type="cuda", dtype=compute_dtype)
                if (amp and device == "cuda")
                else nullcontext()
            )
            with ctx:
                logits = H_chunk @ WU.T  # (Tb,V)
                if temperature is not None:
                    logits = logits / float(temperature)
                probs = torch.softmax(logits.float(), dim=-1)  # fp32 softmax
                sel = probs.gather(1, ids_chunk.view(-1, 1)).squeeze(1)  # (Tb,)
            out[start:end, l] = sel.detach().cpu().numpy()
        del Hl_t
        if device == "cuda":
            torch.cuda.empty_cache()
    return out


def _iter_layer_files(root: Path, idx: int):
    # (same as your existing definition)
    p = f"{idx:05d}"
    patterns = [f"all_p{p}_l_out-*.npy", f"*p{p}_l_out-*.npy"]
    seen = set()
    for pat in patterns:
        for path in root.rglob(pat):
            try:
                l = int(path.name.split("l_out-")[-1].split(".")[0])  # noqa
                if l not in seen:
                    seen.add(l)
                    yield l, path
            except Exception:
                continue


def _load_all_tokens_all_layers_impute(
    tensors_dir: Path, idx: int, L: int, H_expected: int
) -> tuple[np.ndarray, list[int]]:
    """
    Returns:
      hLT: (L, T, H_expected)  per-layer, per-token hidden states
      imputed_layers: list[int] of missing layers that were imputed
    """
    found: dict[int, Path] = {}
    for l, path in _iter_layer_files(tensors_dir, idx):  # noqa
        found[l] = path

    # Determine T from the first found layer
    T = None
    for l in sorted(found):  # noqa
        arr = np.load(found[l])
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D array (T,H), got {arr.shape} for layer {l}")
        T = int(arr.shape[0])
        break
    if T is None:
        raise FileNotFoundError(f"No layers found at all for idx={idx}")

    rows, imputed_layers = [], []
    for l in range(L):  # noqa
        if l in found:
            arr = np.load(found[l]).astype(np.float32, copy=False)  # (T,H_cur)
            rows.append(arr)
        else:
            # deterministic fill for (T,H_expected)
            seed = (idx * 1_000_003 + l) % (2**32 - 1)
            rng = np.random.default_rng(seed)
            vec = rng.normal(0.0, 0.02, size=(T, H_expected)).astype(np.float32)
            rows.append(vec)
            imputed_layers.append(l)
            logging.warning(f"[impute] idx={idx}: layer {l} missing → imputed (T,H)")
    hLT = np.stack(rows, axis=0)  # (L,T,H_cur or H_expected)
    return hLT, imputed_layers


def discover_p_indices(tensors_dir: str) -> list[int]:
    root = Path(tensors_dir)
    seen = set()
    for f in root.rglob("*.npy"):
        m = _PIDX_FILE_RE.search(f.name)
        if m:
            seen.add(int(m.group(1)))
    for d in root.rglob("p[0-9][0-9][0-9][0-9][0-9]"):
        if d.is_dir():
            try:
                seen.add(int(d.name[1:]))  # strip 'p'
            except ValueError:
                pass
    return sorted(seen)


def _softmax_np(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z, dtype=np.float64)
    return (e / e.sum()).astype(np.float32)


def _load_last_token_all_layers(tensors_dir: Path, idx: int, L: int) -> np.ndarray:
    found: dict[int, Path] = {}
    for l, path in _iter_layer_files(tensors_dir, idx):  # noqa
        found[l] = path
    missing = [l for l in range(L) if l not in found]  # noqa
    if missing:
        raise FileNotFoundError(f"idx={idx} missing layers: {missing}")
    rows = []
    for l in range(L):  # noqa
        arr = np.load(found[l])
        rows.append(arr[-1, :].astype(np.float32, copy=False))  # last token
    return np.stack(rows, axis=0)  # (L,H)


def next_token_id_from_final_layer(
    h_last_token_layers: np.ndarray, unembed: torch.Tensor
) -> int:
    # unembed: [V, H], hL: [H]
    WU = unembed.cpu().numpy()
    hL = h_last_token_layers[-1]
    logits = WU @ hL
    return int(np.argmax(logits))


class TunedLensRunner:
    def __init__(
        self,
        tuned_lens: TunedLens,
        device: str = "cpu",
        temperature: float | None = None,
    ):
        self.tuned_lens = tuned_lens
        self.device = device
        self.temperature = temperature

    @torch.no_grad()
    def curve(self, h_last_token_layers: np.ndarray, tok_id: int) -> np.ndarray:
        L, H = h_last_token_layers.shape
        out = np.zeros((L,), dtype=np.float32)
        for idx in range(L):
            h = torch.tensor(
                h_last_token_layers[idx : idx + 1, :],
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(
                0
            )  # (1,1,H)
            logits = self.tuned_lens(h, idx=idx)  # (1,1,V)
            if self.temperature is not None:
                logits = logits / float(self.temperature)
            logits = logits.to(torch.float32)
            probs = torch.softmax(logits, dim=-1).squeeze()  # (V,)
            out[idx] = float(probs[tok_id].cpu())
        return out


def logit_lens_curve(
    h_last_token_layers: np.ndarray, unembed: torch.Tensor, tok_id: int
) -> np.ndarray:
    WU = unembed.cpu().numpy()
    L = h_last_token_layers.shape[0]
    out = np.zeros((L,), dtype=np.float32)
    for l in range(L):  # noqa
        probs = _softmax_np(WU @ h_last_token_layers[l])
        out[l] = probs[tok_id]
    return out


def _load_model_and_unembed(model_id: str):
    """
    Loads either a seq2seq (T5/T5-Gemma) or decoder-only model and returns:
      - model (on CPU map)
      - unembed weight (FloatTensor [V, H])
      - is_encoder_decoder flag
    """
    cfg = AutoConfig.from_pretrained(model_id)
    is_encdec = bool(getattr(cfg, "is_encoder_decoder", False))

    if is_encdec:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            device_map={"": "cpu"},
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map={"": "cpu"},
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )

    # Robustly get LM head
    unembed_module = model.get_output_embeddings()
    if unembed_module is None and hasattr(model, "lm_head"):
        unembed_weight = model.lm_head.weight.detach().to(torch.float32)
    else:
        unembed_weight = unembed_module.weight.detach().to(torch.float32)

    return model, unembed_weight, is_encdec


def _probe_shape(
    tensors_dir: Path, indices: list[int], L_expected: int, H_expected: int
):
    for probe_idx in indices[:10]:
        try:
            hL = _load_last_token_all_layers(tensors_dir, probe_idx, L_expected)
            H_cur = hL.shape[1]
            if H_cur != H_expected:
                logging.warning(
                    f"[probe] idx={probe_idx}: hidden size {H_cur} != LM head {H_expected}. "
                    f"These tensors likely come from a different base model."
                )
        except Exception as e:
            logging.warning(f"[probe] idx={probe_idx} failed: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensors-dir", required=True)
    ap.add_argument("--indexes", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--num-layers", type=int, required=True)
    ap.add_argument("--base-model-id", required=True)
    ap.add_argument("--tuned-lens-id", default=None)  # optional
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument(
        "--token-batch",
        type=int,
        default=64,
        help="Per-layer token batch for GPU matmuls.",
    )
    ap.add_argument("--no-amp", action="store_true", help="Disable CUDA autocast.")
    args = ap.parse_args()

    tensors_dir = Path(args.tensors_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_dtype = choose_compute_dtype(device)
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    logging.info(f"[init] using device: {device} (compute dtype: {compute_dtype})")

    # --- discover indices ---
    if args.indexes:
        with open(args.indexes) as f:
            indices = json.load(f)
    else:
        indices = discover_p_indices(args.tensors_dir)
        if not indices:
            raise RuntimeError(
                f"No p-indices found under {args.tensors_dir}. "
                "Expect files like *_p00001_l_out-*.npy or subdirs p00001/."
            )
    logging.info(
        f"[info] Will process {len(indices)} indices: first few = {indices[:10]}"
    )

    # --- load model (for LM head only; supports enc-dec or decoder-only) ---
    logging.info(f"[init] base model: {args.base_model_id}")
    model, unembed_cpu, is_encdec = _load_model_and_unembed(args.base_model_id)
    WU = unembed_cpu.to(
        device=device, dtype=compute_dtype, copy=True
    )  # (V,H) ON GPU if available
    H_expected = int(WU.shape[1])
    logging.info(
        f"[init] encoder_decoder={is_encdec} LM head: vocab={WU.shape[0]} hidden={H_expected}"
    )

    # --- Tuned Lens (optional) ---
    if args.tuned_lens_id and str(args.tuned_lens_id).lower() not in {
        "",
        "none",
        "null",
    }:
        try:
            logging.info(f"[init] tuned lens: {args.tuned_lens_id}")
            tuned_lens = TunedLens.from_model_and_pretrained(
                model, lens_resource_id=args.tuned_lens_id, map_location=device
            )
            try:
                tuned_lens.to(device)
            except Exception:
                pass
        except Exception as e:
            logging.warning(
                f"[tuned-lens] Could not load '{args.tuned_lens_id}': {e}. Proceeding with logit-lens only."
            )
    else:
        logging.info("[init] no tuned lens provided; using logit-lens only")

    # free model body
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    use_amp = (device == "cuda") and (not args.no_amp)

    # --- main loop ---
    for idx in tqdm(indices, desc="Writing curves"):
        out_path = out_dir / f"curves_idx{idx:05d}.npz"
        if args.skip_existing and out_path.exists():
            continue
        try:
            # Load (L,T,H) + imputed info
            hLT, imputed_layers = _load_all_tokens_all_layers_impute(
                tensors_dir, idx, L=args.num_layers, H_expected=H_expected
            )
            # width guard
            if hLT.shape[2] != H_expected:
                logging.warning(
                    f"[skip] idx={idx}: hidden size {hLT.shape[2]} != LM head {H_expected} "
                    f"(base_model_id={args.base_model_id}). Skipping."
                )
                continue

            # Decide next-token ids from FINAL layer (GPU)
            tok_ids = next_token_ids_from_final_layer_all_torch(
                hLT, WU, device=device, amp=use_amp
            )  # (T,)

            # Logit-lens curves for ALL tokens: (T, L) on GPU
            logit_curves = logit_lens_curves_all_torch(
                hLT,
                WU,
                tok_ids,
                device=device,
                token_batch=args.token_batch,
                temperature=args.temperature,
                amp=use_amp,
            )

            np.savez_compressed(
                out_path,
                tok_ids=tok_ids.astype(np.int32),  # (T,)
                logit_curves=logit_curves,  # (T, L)
                imputed_layers=np.array(imputed_layers, dtype=np.int32),
            )

            if device == "cuda":
                torch.cuda.empty_cache()

        except Exception as e:
            logging.warning(f"[warn] idx={idx} failed: {e}")

    logging.info(f"[done] curve files in: {out_dir}")


if __name__ == "__main__":
    main()
