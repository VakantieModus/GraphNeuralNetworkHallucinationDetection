import argparse
import logging
import os
import pathlib
import re
import subprocess
import tempfile

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

BIN_DEFAULT = "src/white_box_benchmark/llama.cpp/build/bin/llama-eval-callback"


def run_cmd(cmd: list[str], cwd: str | None = None) -> subprocess.CompletedProcess:
    # Fail loudly on nonzero exit
    return subprocess.run(cmd, text=True, capture_output=True, check=True, cwd=cwd)


def detect_last_l_out(binary: str, model: str) -> int:
    """Call the binary in --list-layers mode and return the max l_out-N found."""
    with tempfile.TemporaryDirectory() as td:
        prefix = os.path.join(td, "listing")
        cmd = [
            binary,
            "--model",
            model,
            "--list-layers",
            "--prompt",
            "x",
            "--n-predict",
            "1",
            "--output-prefix",
            prefix,
        ]
        res = run_cmd(cmd)
        listing_path = prefix + "_tensor_output.txt"
        text = (
            pathlib.Path(listing_path).read_text()
            if os.path.exists(listing_path)
            else res.stdout
        )
        matches = re.findall(r"\bl_out-(\d+)\b", text)
        if not matches:
            raise RuntimeError("No l_out-* layers found in listing output")
        last_idx = max(int(m) for m in matches)
        log.info("Auto-detected last l_out index: %d", last_idx)
        return last_idx


def run_one_pass(
    *,
    binary: str,
    model: str,
    prompts: list[str],
    output_dir: str,
    parse_layers: list[str],
    n_predict: int,
    n_gpu_layers: int,
    sampler_args: list[str] | None = None,
    dry_run: bool = False,
):
    os.makedirs(output_dir, exist_ok=True)

    base = [
        binary,
        "--model",
        model,
        "--n-predict",
        str(n_predict),
        "--n-gpu-layers",
        str(n_gpu_layers),
    ]
    for layer in parse_layers:
        base += ["--parse-layer", layer]
    if sampler_args:
        base += sampler_args

    for i, prompt in enumerate(prompts):
        prefix = os.path.join(output_dir, f"p{i:05d}")
        os.makedirs(prefix, exist_ok=True)
        prompt = (
            f"Question: {prompt}\n"
            f"Answer the question as precise as possible and add no irrelevant context.\n"
            f"Explain your answer by reasoning step by step in a concise manner.\n"
            f"Answer: "
        )
        cmd = base + ["--output-prefix", prefix, "--prompt", prompt]

        log.info("⏳ [%s] prompt %d", os.path.basename(output_dir), i)
        try:
            res = run_cmd(cmd)
        except subprocess.CalledProcessError as e:
            # Save stderr/stdout so we can inspect crashes
            pathlib.Path(os.path.join(output_dir, f"stderr_{i:05d}.txt")).write_text(
                e.stderr or ""
            )
            pathlib.Path(os.path.join(output_dir, f"stdout_{i:05d}.txt")).write_text(
                e.stdout or ""
            )
            raise

        # Save stdout/stderr per prompt for debugging anyway
        pathlib.Path(os.path.join(output_dir, f"stdout_{i:05d}.txt")).write_text(
            res.stdout or ""
        )
        pathlib.Path(os.path.join(output_dir, f"stderr_{i:05d}.txt")).write_text(
            res.stderr or ""
        )

        # Quick sanity check: tensors should exist & not be empty
        tens = os.path.join(prefix, "tensors")
        if not os.path.isdir(tens) or not any(
            name.endswith(".txt") for name in os.listdir(tens)
        ):
            log.warning("⚠️ No tensors found for prompt %d under %s", i, tens)

        if dry_run:
            log.info("Dry run: stopping after first prompt.")
            break


def load_prompts(path: str) -> list[str]:
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default=BIN_DEFAULT)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output-dir", default="generated_data/run")
    ap.add_argument("--n-predict", type=int, default=512)
    ap.add_argument("--n-gpu-layers", type=int, default=0)
    ap.add_argument("--temp", type=float)
    ap.add_argument("--top-p", type=float)
    ap.add_argument("--top-k", type=int)
    ap.add_argument("--repeat-penalty", type=float)
    ap.add_argument("--last-layer-idx", type=int, help="Override final l_out index")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    prompts = load_prompts(args.dataset)

    sampler = []
    if args.temp is not None:
        sampler += ["--temp", str(args.temp)]
    if args.top_p is not None:
        sampler += ["--top-p", str(args.top_p)]
    if args.top_k is not None:
        sampler += ["--top-k", str(args.top_k)]
    if args.repeat_penalty is not None:
        sampler += ["--repeat-penalty", str(args.repeat_penalty)]

    # Pass A: all residuals + final norm + logits
    out_all = os.path.join(args.output_dir, "all")
    run_one_pass(
        binary=args.binary,
        model=args.model,
        prompts=prompts,
        output_dir=out_all,
        parse_layers=["l_out-*"],
        n_predict=args.n_predict,
        n_gpu_layers=args.n_gpu_layers,
        sampler_args=sampler,
        dry_run=args.dry_run,
    )

    # Pass B: last residual only
    last_idx = (
        args.last_layer_idx
        if args.last_layer_idx is not None
        else detect_last_l_out(args.binary, args.model)
    )
    for r in range(5):
        out_last = os.path.join(args.output_dir, f"last_r{r}")
        sampler_r = list(sampler)
        # If llama-eval-callback supports --seed (llama.cpp does), vary it per repeat

        last_repeats = 5  # or args.last_repeats if you added the flag
        out_last = os.path.join(args.output_dir, f"last_r{r}")
        log.info(
            "🚀 Starting Pass B repeat %d/%d (output: %s)",
            r + 1,
            last_repeats,
            out_last,
        )

        run_one_pass(
            binary=args.binary,
            model=args.model,
            prompts=prompts,
            output_dir=out_last,
            parse_layers=[f"l_out-{last_idx}"],
            n_predict=args.n_predict,
            n_gpu_layers=args.n_gpu_layers,
            sampler_args=sampler_r,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
