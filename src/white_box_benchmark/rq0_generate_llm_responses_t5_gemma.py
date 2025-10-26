#!/usr/bin/env python3
"""
Run T5-Gemma (google/t5gemma-xl-xl-prefixlm-it) with Transformers.

Features
- cpp-like structure with run_one_pass(...)
- dry-run: run only the FIRST prompt in the dataset then exit
- per-prompt subfolders p00000 / tensors
- tensor files named like llama.cpp output: l_out-{L}.txt (or .npy)
- logits.txt (final-step logits vector; optional per-step)
- explicit GPU pinning (--gpu-id or --cuda-visible-devices)

Install:
  pip install -U "transformers>=4.43" "accelerate>=0.33" torch tqdm
"""

import argparse
import json
import logging
import os
import pathlib
import random
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    GenerationConfig,
    set_seed,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("t5gemma_runner")


# -------------------------
# Utilities & dataclasses
# -------------------------


@dataclass
class RunConfig:
    model_id: str
    dataset: str
    output_dir: str
    max_new_tokens: int
    temperature: float | None
    top_p: float | None
    top_k: int | None
    repetition_penalty: float | None
    do_sample: bool
    repeats: int
    seed: int | None
    dtype: str
    device: str
    device_map: str
    batch_size: int
    dump_tensors: bool
    trust_remote_code: bool
    prefix_template: str
    gpu_id: int | None
    cuda_visible_devices: str | None
    dry_run: bool
    tensor_format: str
    dump_all_logit_steps: bool


def write_layer_token_blocks_txt(
    path: pathlib.Path, layer_matrix: np.ndarray, width: int = 16
) -> None:
    """
    Write per-token vectors with headers compatible with your parser:
      === TOKEN {n} [{H}]===
      DATA:
      <values...>

    layer_matrix: (num_steps, hidden_size)  rows=tokens, cols=hidden units
    """
    T, H = layer_matrix.shape
    with path.open("w", encoding="utf-8") as f:
        for n in range(T):
            f.write(f"=== TOKEN {n} [{H}]===" + "\n")
            f.write("DATA:" + "\n")
            vec = layer_matrix[n]
            # wrap to 'width' floats per line for readability
            for start in range(0, H, width):
                chunk = vec[start : start + width]
                f.write(" ".join(f"{x:.7g}" for x in chunk) + "\n")


def ensure_dir(p: str) -> None:
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)


def load_prompts(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def pick_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name in ("auto", "bf16", "bfloat16") and torch.cuda.is_available():
        return torch.bfloat16
    if name in ("fp16", "float16"):
        return torch.float16
    if name in ("fp32", "float32"):
        return torch.float32
    return torch.float32


def build_gen_config(cfg: RunConfig) -> GenerationConfig:
    params = dict(max_new_tokens=cfg.max_new_tokens, do_sample=cfg.do_sample)
    if cfg.temperature is not None:
        params["temperature"] = cfg.temperature
    if cfg.top_p is not None:
        params["top_p"] = cfg.top_p
    if cfg.top_k is not None:
        params["top_k"] = int(cfg.top_k)
    if cfg.repetition_penalty is not None:
        params["repetition_penalty"] = cfg.repetition_penalty
    return GenerationConfig(**params)


def format_prompt(raw: str, template: str) -> str:
    return template.format(question=raw)


def write_array_txt(path: pathlib.Path, arr: np.ndarray) -> None:
    # Save as space-delimited text (compatible with your cpp parsers)
    np.savetxt(
        path, arr.reshape(-1, arr.shape[-1]) if arr.ndim > 2 else arr, fmt="%.7g"
    )


def write_array(path: pathlib.Path, arr: np.ndarray, fmt: str) -> None:
    if fmt == "txt":
        write_array_txt(path, arr)
    else:
        np.save(path, arr)


# --------------------------------
# Core: single pass over prompts
# --------------------------------


def run_one_pass(
    *,
    cfg: RunConfig,
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    prompts: Sequence[str],
    out_dir: str,
    seed: int,
    last_only: bool = False,
) -> None:
    """Process prompts once (one 'repeat') into out_dir."""
    ensure_dir(out_dir)
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    gen_config = build_gen_config(cfg)

    for i, raw in enumerate(prompts):
        prompt_text = format_prompt(raw, cfg.prefix_template)
        prefix_dir = os.path.join(out_dir, f"p{i:05d}")
        ensure_dir(prefix_dir)

        # Tokenize
        inputs = tokenizer(
            prompt_text,
            return_tensors="pt",
            padding=False,
            truncation=True,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        need_scores = cfg.dump_tensors
        need_hidden = cfg.dump_tensors

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                generation_config=gen_config,
                return_dict_in_generate=True,
                output_scores=need_scores,
                output_hidden_states=need_hidden,  # decoder hidden states during generation
            )

        # Decode generated text
        text = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)[0]
        # Save exact generated token IDs and decoded subword tokens
        generated_ids = (
            outputs.sequences[0][1:].detach().cpu().tolist()
        )  # drop start token
        tokens_generated = tokenizer.convert_ids_to_tokens(
            generated_ids, skip_special_tokens=False
        )

        (pathlib.Path(prefix_dir) / "generated_ids.json").write_text(
            json.dumps(
                {
                    "generated_ids": generated_ids,
                    "tokens_generated": tokens_generated,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        (pathlib.Path(prefix_dir) / "stdout.txt").write_text(text, encoding="utf-8")

        # Minimal meta (like your stdout/stderr bookkeeping)
        meta = {
            "index": i,
            "seed": seed,
            "prompt": prompt_text,
            "model_id": cfg.model_id,
            "gen_config": gen_config.to_dict(),
            "device": str(model.device),
            "time": time.time(),
        }
        (pathlib.Path(prefix_dir) / "meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )

        # Tensors (cpp-like naming)
        if cfg.dump_tensors:
            tens_dir = pathlib.Path(prefix_dir) / "tensors"
            tens_dir.mkdir(exist_ok=True, parents=True)

            # -- Decoder hidden states --
            # outputs.decoder_hidden_states is a list over generation steps
            # each element is a tuple(layer0,...,layerN) with shape (1, seq_len, hidden_dim)
            if (
                hasattr(outputs, "decoder_hidden_states")
                and outputs.decoder_hidden_states
            ):
                steps = outputs.decoder_hidden_states
                num_steps = len(steps)
                num_layers = len(steps[0]) if num_steps > 0 else 0

                # which layers to dump?
                layer_indices = (
                    [num_layers - 1] if last_only else list(range(num_layers))
                )

                for L in layer_indices:
                    per_step_vecs = []
                    for s in range(num_steps):
                        hs = steps[s][L]  # (1, seq_len_t, H)
                        vec = hs[0, -1, :].detach().float().cpu().numpy()  # (H,)
                        per_step_vecs.append(vec)

                    layer_matrix = np.stack(per_step_vecs, axis=0).astype(
                        np.float32
                    )  # (T, H)

                    if cfg.tensor_format == "txt":
                        write_layer_token_blocks_txt(
                            tens_dir / f"l_out-{L}.txt", layer_matrix, width=16
                        )
                    else:
                        np.save(tens_dir / f"l_out-{L}.npy", layer_matrix)
            # -- Logits --
            # outputs.scores is a list[tensor] over generation steps with shape (batch, vocab)
            if hasattr(outputs, "scores") and outputs.scores:
                if cfg.dump_all_logit_steps:
                    # big: one row per step; saves as 2D [steps, vocab]
                    logits = torch.stack(
                        [s.detach().float().cpu() for s in outputs.scores], dim=0
                    ).numpy()
                    write_array(
                        tens_dir
                        / f"logits.{ 'txt' if cfg.tensor_format=='txt' else 'npy'}",  # noqa
                        logits,
                        cfg.tensor_format,
                    )
                else:
                    # smaller: only final step
                    final = (
                        outputs.scores[-1].detach().float().cpu().numpy()
                    )  # (1, vocab)
                    write_array(
                        tens_dir
                        / f"logits.{ 'txt' if cfg.tensor_format=='txt' else 'npy'}",  # noqa
                        final,
                        cfg.tensor_format,
                    )

        # DRY-RUN: stop after first prompt TOTAL
        if cfg.dry_run:
            log.info("Dry run: stopping after first prompt.")
            return

        if (i + 1) % 10 == 0:
            log.info("Processed %d/%d prompts", i + 1, len(prompts))


# -------------
# CLI / main
# -------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="google/t5gemma-xl-xl-prefixlm-it")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output-dir", default="generated_data/t5gemma_run")
    ap.add_argument("--max-new-tokens", type=int, default=512)

    # Sampler
    ap.add_argument("--temp", type=float, dest="temperature")
    ap.add_argument("--top-p", type=float)
    ap.add_argument("--top-k", type=int)
    ap.add_argument("--repeat-penalty", type=float, dest="repetition_penalty")
    ap.add_argument("--no-sample", action="store_true")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Run only the FIRST prompt in the dataset, then exit",
    )

    # Execution
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--seed", type=int)
    ap.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "bf16", "fp16", "fp32", "float16", "float32", "bfloat16"],
    )
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument(
        "--device-map",
        default="auto",
        choices=["auto", "sequential", "balanced", "cpu"],
    )

    # GPU targeting
    ap.add_argument(
        "--gpu-id", type=int, help="Pin to a single physical GPU id (e.g., 0 or 1)"
    )
    ap.add_argument(
        "--cuda-visible-devices",
        type=str,
        help="Set CUDA_VISIBLE_DEVICES (e.g., '1' or '1,0'). Overrides --gpu-id.",
    )

    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument(
        "--dump-tensors",
        action="store_true",
        help="Save decoder hidden states and logits per prompt",
    )
    ap.add_argument("--trust-remote-code", action="store_true")

    ap.add_argument(
        "--tensor-format",
        default="txt",
        choices=["txt", "npy"],
        help="How to write tensor dumps (default txt to match cpp parsers)",
    )
    ap.add_argument(
        "--dump-all-logit-steps",
        action="store_true",
        help="If set, save logits for every generation step; else only the final step",
    )
    ap.add_argument(
        "--last-repeats",
        type=int,
        default=5,
        help="Number of repeats for the last-layer-only pass (creates last_r0..last_r{n-1})",
    )

    ap.add_argument(
        "--prefix-template",
        default=(
            "Question: {question}\n"
            "Answer the question as precise as possible and add no irrelevant context.\n"
            "Explain your answer by reasoning step by step in a concise manner.\n"
            "Answer:"
        ),
        help="Python .format template with {question}",
    )

    args = ap.parse_args()

    # Handle GPU targeting EARLY
    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
        log.info("CUDA_VISIBLE_DEVICES set to '%s'", args.cuda_visible_devices)
    elif args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        log.info("Pinned to GPU id %d via CUDA_VISIBLE_DEVICES", args.gpu_id)

    cfg = RunConfig(
        model_id=args.model_id,
        dataset=args.dataset,
        output_dir=args.output_dir,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        do_sample=not args.no_sample,
        repeats=args.repeats,
        seed=args.seed,
        dtype=args.dtype,
        device=args.device,
        device_map=args.device_map,
        batch_size=args.batch_size,
        dump_tensors=bool(args.dump_tensors),
        trust_remote_code=bool(args.trust_remote_code),
        prefix_template=args.prefix_template,
        gpu_id=args.gpu_id,
        cuda_visible_devices=args.cuda_visible_devices,
        dry_run=bool(args.dry_run),
        tensor_format=args.tensor_format,
        dump_all_logit_steps=bool(args.dump_all_logit_steps),
    )

    ensure_dir(cfg.output_dir)
    log.info("Config: %s", json.dumps(asdict(cfg), indent=2))

    prompts = load_prompts(cfg.dataset)
    log.info("Loaded %d prompts", len(prompts))

    # Device & dtype
    torch_dtype = pick_dtype(cfg.dtype)
    if cfg.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = cfg.device

    # Tokenizer & model
    log.info("Loading tokenizer: %s", cfg.model_id)
    log.info("Loading tokenizer (slow SP): %s", cfg.model_id)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_id,
        use_fast=False,  # always slow
        trust_remote_code=cfg.trust_remote_code,
    )

    log.info("Loading model: %s", cfg.model_id)
    force_single_gpu = (
        bool(os.environ.get("CUDA_VISIBLE_DEVICES"))
        and "," not in os.environ["CUDA_VISIBLE_DEVICES"]
    )
    effective_device_map = (
        None
        if (device == "cuda" and force_single_gpu)
        else (None if device == "cpu" else cfg.device_map)
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        cfg.model_id,
        torch_dtype=torch_dtype,
        device_map=effective_device_map if effective_device_map != "cpu" else None,
        trust_remote_code=cfg.trust_remote_code,
    )
    # ---- Cache/config compatibility shim for T5-Gemma ----
    cfg_obj = model.config
    if not hasattr(cfg_obj, "num_hidden_layers"):
        if hasattr(cfg_obj, "num_decoder_layers"):
            cfg_obj.num_hidden_layers = cfg_obj.num_decoder_layers
        elif hasattr(cfg_obj, "num_layers"):
            cfg_obj.num_hidden_layers = cfg_obj.num_layers
        else:
            dec = getattr(getattr(model, "model", model), "decoder", None)
            if dec is not None and hasattr(dec, "block"):
                cfg_obj.num_hidden_layers = len(dec.block)
            else:
                cfg_obj.num_hidden_layers = 24

    if device == "cpu" or effective_device_map is None:
        target = "cpu" if device == "cpu" else "cuda:0"
        model.to(target)

    # Repeats: r0..r{repeats-1}
    total_start = time.time()
    # -------------------------
    # PASS A: all layers → ./all
    # -------------------------
    out_all = os.path.join(cfg.output_dir, "all")
    seed_base = cfg.seed if cfg.seed is not None else 1234
    log.info("🚀 PASS A (all layers) → %s", out_all)
    run_one_pass(
        cfg=cfg,
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        out_dir=out_all,
        seed=seed_base,
        last_only=False,
    )

    # -------------------------
    # PASS B: last layer only → ./last_r0..last_r{n-1}
    # -------------------------
    for r in range(getattr(args, "last_repeats", 5)):
        out_last = os.path.join(cfg.output_dir, f"last_r{r}")
        seed_r = (cfg.seed if cfg.seed is not None else 1234) + r * 9973
        log.info(
            "🚀 PASS B (last layer only) repeat %d/%d → %s (seed=%d)",
            r + 1,
            getattr(args, "last_repeats", 5),
            out_last,
            seed_r,
        )
        run_one_pass(
            cfg=cfg,
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            out_dir=out_last,
            seed=seed_r,
            last_only=True,
        )

    log.info("✅ Done in %.1fs", time.time() - total_start)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.exception("Fatal error: %s", e)
        sys.exit(1)
