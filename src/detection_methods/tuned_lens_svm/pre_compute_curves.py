#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import logging
import re
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from tuned_lens import TunedLens

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

_PIDX_FILE_RE = re.compile(r"_p(\d{5})_l_out-")  # matches ..._p00001_l_out-123.npy


def discover_p_indices(tensors_dir: str) -> list[int]:
    root = Path(tensors_dir)
    seen = set()

    # filenames like all_p00001_l_out-*.npy or ..._p00001_l_out-*.npy (recursive)
    for f in root.rglob("*.npy"):
        m = _PIDX_FILE_RE.search(f.name)
        if m:
            seen.add(int(m.group(1)))

    # directories named p00001
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


def _iter_layer_files(root: Path, idx: int):
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


def _load_last_token_all_layers_impute(
    tensors_dir: Path, idx: int, L: int, H_expected: int
) -> tuple[np.ndarray, list[int]]:
    """
    Load last-token vectors for layers [0..L-1]. If a layer file is missing,
    impute with a deterministic random vector of width H_expected.
    Returns:
      - h: np.ndarray, shape (L, H_expected)
      - imputed_layers: list[int] of layer indices that were imputed
    """
    found: dict[int, Path] = {}
    for l, path in _iter_layer_files(tensors_dir, idx):  # noqa
        found[l] = path

    rows = []
    imputed_layers: list[int] = []
    for l in range(L):  # noqa
        if l in found:  # noqa
            arr = np.load(found[l])
            vec = arr[-1, :].astype(np.float32, copy=False)
            # If stored width != expected, we still return it; width will be checked by caller.
            rows.append(vec)
        else:
            # deterministic RNG: seed depends on (idx, layer)
            seed = (idx * 1_000_003 + l) % (2**32 - 1)
            rng = np.random.default_rng(seed)
            vec = rng.normal(loc=0.0, scale=0.02, size=(H_expected,)).astype(np.float32)
            rows.append(vec)
            imputed_layers.append(l)
            logging.warning(
                f"[impute] idx={idx}: layer {l} missing → imputed random vector"
            )

    h = np.stack(rows, axis=0)
    return h, imputed_layers


def next_token_id_from_final_layer(
    h_last_token_layers: np.ndarray, unembed: torch.Tensor
) -> int:
    # unembed: [V, H], hL: [H]
    WU = unembed.cpu().numpy()
    hL = h_last_token_layers[-1]
    logits = WU @ hL
    return int(np.argmax(logits))


def logit_lens_curves_all(
    hLT: np.ndarray, unembed: torch.Tensor, tok_ids: np.ndarray, token_batch: int = 64
) -> np.ndarray:
    """
    hLT: (L, T, H), tok_ids: (T,)
    Returns curves: (T, L), where curves[t, l] is P_tok_id(t) read from layer l.
    """
    WU = unembed.cpu().numpy()  # (V, H)
    L, T, H = hLT.shape
    out = np.zeros((T, L), dtype=np.float32)

    for l in range(L):  # noqa
        H_l = hLT[l]  # (T, H)
        # Process tokens in mini-batches to control RAM
        for start in range(0, T, token_batch):
            end = min(start + token_batch, T)
            H_chunk = H_l[start:end]  # (Tb, H)
            logits = H_chunk @ WU.T  # (Tb, V)
            # softmax per row (token)
            logits -= logits.max(axis=1, keepdims=True)
            np.exp(logits, out=logits)
            denom = logits.sum(axis=1, keepdims=True)  # (Tb,1)
            probs_tok = logits[
                np.arange(end - start), tok_ids[start:end]
            ] / denom.squeeze(1)
            out[start:end, l] = probs_tok.astype(np.float32)
    return out


def choose_compute_dtype(device: str) -> torch.dtype:
    # bfloat16 is usually stable for logits/softmax on modern GPUs; fall back to float16 when needed.
    if device.startswith("cuda"):
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


@torch.no_grad()
def next_token_ids_from_final_layer_all_torch(
    hLT: np.ndarray,  # (L, T, H) in numpy
    WU: torch.Tensor,  # (V, H) torch on device (compute dtype)
    device: str,
    amp: bool = True,
) -> np.ndarray:
    """
    Returns tok_ids: (T,), using GPU matmul (final layer).
    """
    L, T, H = hLT.shape
    compute_dtype = WU.dtype
    h_final = torch.from_numpy(hLT[-1]).to(device=device, dtype=compute_dtype)  # (T,H)
    with torch.autocast(device_type="cuda", dtype=compute_dtype, enabled=amp):
        logits = h_final @ WU.T  # (T,V)
        tok_ids = torch.argmax(logits, dim=1).to(torch.int32).cpu().numpy()
    return tok_ids


@torch.no_grad()
def logit_lens_curves_all_torch(
    hLT: np.ndarray,  # (L, T, H) numpy
    WU: torch.Tensor,  # (V, H) on device (compute dtype)
    tok_ids: np.ndarray,  # (T,)
    device: str,
    token_batch: int = 64,
    temperature: float | None = None,
    amp: bool = True,
) -> np.ndarray:
    """
    GPU implementation. Returns (T, L) with probs for tok_ids[t] read from each layer.
    Processes tokens in mini-batches to control VRAM.
    """
    L, T, H = hLT.shape
    compute_dtype = WU.dtype
    out = np.zeros((T, L), dtype=np.float32)

    tok_ids_t = torch.from_numpy(tok_ids).to(device=device)

    for l in range(L):  # noqa
        # Send this layer's (T,H) to device (once)
        Hl_t = torch.from_numpy(hLT[l]).to(device=device, dtype=compute_dtype)  # (T,H)

        for start in range(0, T, token_batch):
            end = min(start + token_batch, T)
            H_chunk = Hl_t[start:end]  # (Tb,H)
            ids_chunk = tok_ids_t[start:end]  # (Tb,)

            with torch.autocast(device_type="cuda", dtype=compute_dtype, enabled=amp):
                logits = H_chunk @ WU.T  # (Tb, V)
                if temperature is not None:
                    logits = logits / float(temperature)
                # softmax and gather the chosen token prob
                probs = torch.softmax(
                    logits.float(), dim=-1
                )  # keep softmax in fp32 for stability
                p_sel = probs.gather(1, ids_chunk.view(-1, 1)).squeeze(1)  # (Tb,)
            out[start:end, l] = p_sel.detach().cpu().numpy()
        # free per-layer tensor ASAP
        del Hl_t
        torch.cuda.empty_cache()
    return out


def next_token_ids_from_final_layer_all(
    hLT: np.ndarray, unembed: torch.Tensor
) -> np.ndarray:
    """
    hLT: (L, T, H), unembed: [V, H]
    Returns tok_ids: (T,) where tok_ids[t] = argmax over vocab using final-layer state at position t.
    """
    WU = unembed.cpu().numpy()  # (V, H)
    h_final = hLT[-1]  # (T, H)
    logits = h_final @ WU.T  # (T, V)
    return np.asarray(np.argmax(logits, axis=1), dtype=np.int32)


def _load_all_tokens_all_layers_impute(
    tensors_dir: Path, idx: int, L: int, H_expected: int
) -> tuple[np.ndarray, list[int]]:
    """
    Return:
      hLT: (L, T, H_expected)  per-layer, per-token hidden states
      imputed_layers: list[int] of missing layers that were imputed
    Assumes all real layers for this (idx) share the same T. If every layer
    is missing, raises.
    """
    found: dict[int, Path] = {}
    for l, path in _iter_layer_files(tensors_dir, idx):  # noqa
        found[l] = path

    # Determine T from the first actually found layer
    T = None
    for l in sorted(found):  # noqa
        arr = np.load(found[l])
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D array (T,H), got {arr.shape} for layer {l}")
        T = int(arr.shape[0])
        break
    if T is None:
        raise FileNotFoundError(f"No layers found at all for idx={idx}")

    rows = []
    imputed_layers: list[int] = []
    for l in range(L):  # noqa
        if l in found:
            arr = np.load(found[l]).astype(np.float32, copy=False)  # (T, H_cur)
            if arr.shape[1] != H_expected:
                # width mismatch will be caught by caller; still return what we have
                pass
            rows.append(arr)
        else:
            # Deterministic imputation for the full (T, H_expected)
            seed = (idx * 1_000_003 + l) % (2**32 - 1)
            rng = np.random.default_rng(seed)
            vec = rng.normal(loc=0.0, scale=0.02, size=(T, H_expected)).astype(
                np.float32
            )
            rows.append(vec)
            imputed_layers.append(l)
            logging.warning(f"[impute] idx={idx}: layer {l} missing → imputed (T,H)")

    hLT = np.stack(rows, axis=0)  # (L, T, H_expected or H_cur)
    return hLT, imputed_layers


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
            if self.temperature:
                logits = logits / float(self.temperature)
            logits = logits.to(torch.float32)
            probs = torch.softmax(logits, dim=-1).squeeze()  # (V,)
            out[idx] = float(probs[tok_id].cpu())
        return out

    @torch.no_grad()
    def curves_all(
        self, hLT: np.ndarray, tok_ids: np.ndarray, token_batch: int = 0
    ) -> np.ndarray:
        """
        hLT: (L, T, H); returns (T, L) of probabilities at tok_ids[t].
        token_batch == 0 → process the full sequence at once per layer.
        """
        device = self.device
        L, T, H = hLT.shape
        out = np.zeros((T, L), dtype=np.float32)

        for l in range(L):  # noqa
            if token_batch and token_batch < T:
                # chunk over tokens
                acc = []
                for start in range(0, T, token_batch):
                    end = min(start + token_batch, T)
                    h = torch.from_numpy(hLT[l, start:end][None, :, :]).to(
                        device
                    )  # (1,Tb,H)
                    logits = self.tuned_lens(h, idx=l)  # (1,Tb,V)
                    if self.temperature:
                        logits = logits / float(self.temperature)
                    probs = torch.softmax(logits.to(torch.float32), dim=-1).squeeze(
                        0
                    )  # (Tb,V)
                    tok = torch.from_numpy(tok_ids[start:end]).to(probs.device)  # (Tb,)
                    acc.append(
                        probs[torch.arange(probs.size(0)), tok].float().cpu().numpy()
                    )
                out[:, l] = np.concatenate(acc, axis=0)
            else:
                h = torch.from_numpy(hLT[l][None, :, :]).to(device)  # (1,T,H)
                logits = self.tuned_lens(h, idx=l)  # (1,T,V)
                if self.temperature:
                    logits = logits / float(self.temperature)
                probs = torch.softmax(logits.to(torch.float32), dim=-1).squeeze(
                    0
                )  # (T,V)
                tok = torch.from_numpy(tok_ids).to(probs.device)  # (T,)
                out[:, l] = probs[torch.arange(T), tok].float().cpu().numpy()
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensors-dir", required=True)
    ap.add_argument(
        "--indexes",
        default=None,
        help="Optional JSON file with integer indices to process. If omitted, auto-discover from --tensors-dir.",
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--num-layers", type=int, required=True)
    ap.add_argument("--base-model-id", required=True)
    ap.add_argument("--tuned-lens-id", default=None)  # optional; if omitted we skip TL
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument(
        "--token-batch",
        type=int,
        default=64,
        help="Per-layer token batch size for GPU matmuls.",
    )
    ap.add_argument(
        "--no-amp", action="store_true", help="Disable autocast mixed precision on GPU."
    )
    args = ap.parse_args()

    tensors_dir = Path(args.tensors_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_dtype = choose_compute_dtype(device)
    logging.info(f"[init] using device: {device} (compute dtype: {compute_dtype})")

    # Prefer TF32 on Ampere+ for extra speed (harmless on non-CUDA)
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

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

    # --- load base model (for LM head only) ---
    logging.info(f"[init] base model: {args.base_model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id,
        device_map={"": "cpu"},
        dtype=torch.bfloat16,  # lives on CPU only; dtype here is not critical
        low_cpu_mem_usage=True,
    )
    lm_head = model.get_output_embeddings()
    WU = (
        (lm_head.weight if hasattr(lm_head, "weight") else lm_head)
        .detach()
        .to(device=device, dtype=compute_dtype)
    )  # (V, H) ON GPU
    H_expected = int(WU.shape[1])
    logging.info(f"[init] LM head: vocab={WU.shape[0]} hidden={H_expected}")

    # --- Tuned Lens (optional) ---
    tuned_lens = None
    tl_runner = None
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
                tuned_lens.to(device)  # if supported
            except Exception:
                pass
            tl_runner = TunedLensRunner(
                tuned_lens, device=device, temperature=args.temperature
            )
        except Exception as e:
            logging.warning(
                f"[tuned-lens] Could not load '{args.tuned_lens_id}': {e}. Proceeding with logit-lens only."
            )
    else:
        logging.info("[init] no tuned lens provided; using logit-lens only")

    # free base model (we only keep WU)
    del model, lm_head
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- main loop ---
    for idx in tqdm(indices, desc="Writing curves"):
        out_path = out_dir / f"curves_idx{idx:05d}.npz"
        if args.skip_existing and out_path.exists():
            continue

        try:
            # (L, T, H_expected) + imputation info
            hLT, imputed_layers = _load_all_tokens_all_layers_impute(
                tensors_dir, idx, L=args.num_layers, H_expected=H_expected
            )

            # width guard (in case dumps came from another base model)
            if hLT.shape[2] != H_expected:
                logging.warning(
                    f"[skip] idx={idx}: hidden size {hLT.shape[2]} != LM head {H_expected} "
                    f"(base_model_id={args.base_model_id}). Skipping."
                )
                continue

            # Next-token ids per position decided by FINAL layer (GPU)
            tok_ids = next_token_ids_from_final_layer_all_torch(
                hLT, WU, device=device, amp=not args.no_amp
            )  # (T,)

            # Logit-lens curves for ALL tokens: (T, L) on GPU
            logit_curves = logit_lens_curves_all_torch(
                hLT,
                WU,
                tok_ids,
                device=device,
                token_batch=args.token_batch,
                temperature=args.temperature,
                amp=not args.no_amp,
            )

            # Tuned-lens curves (if available): (T, L)
            tuned_curves = (
                tl_runner.curves_all(hLT, tok_ids, token_batch=args.token_batch)
                if tl_runner is not None
                else None
            )

            # Packaging: concat on the LAST axis → (T, 2L) if TL present
            curve_concat = (
                np.concatenate([logit_curves, tuned_curves], axis=1).astype(np.float32)
                if tuned_curves is not None
                else logit_curves.astype(np.float32)
            )

            np.savez_compressed(
                out_path,
                tok_ids=tok_ids.astype(np.int32),  # (T,)
                logit_curves=logit_curves,  # (T, L)
                tuned_curves=(
                    tuned_curves
                    if tuned_curves is not None
                    else np.array([], dtype=np.float32)
                ),  # (T, L) or empty
                curve_concat=curve_concat,  # (T, L) or (T, 2L)
                imputed_layers=np.array(imputed_layers, dtype=np.int32),
            )
        except Exception as e:
            logging.warning(f"[warn] idx={idx} failed: {e}")

    logging.info(f"[done] curve files in: {out_dir}")


if __name__ == "__main__":
    main()
