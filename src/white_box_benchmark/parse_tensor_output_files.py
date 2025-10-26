import argparse
import logging
import re
from pathlib import Path

import numpy as np

FLOAT_RE = re.compile(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+")
TOKEN_START_RE = re.compile(r"^=== TOKEN")
DATA_START_RE = re.compile(r"^DATA:")
BLOCK_SEP_RE = re.compile(r"^(===|---)")


def parse_one_tensor_txt(file_path: Path) -> np.ndarray | None:
    token_vectors = []
    current = []
    in_data_block = False

    try:
        with file_path.open("r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()

                if TOKEN_START_RE.match(line):
                    if current:
                        token_vectors.append(current)
                        current = []
                    in_data_block = False
                    continue

                if DATA_START_RE.match(line):
                    in_data_block = True
                    continue

                if (
                    BLOCK_SEP_RE.match(line)
                    or line.startswith("SHAPE:")
                    or line.startswith("--- TENSOR:")
                ):
                    if in_data_block:
                        in_data_block = False
                    continue

                if in_data_block and line:
                    floats = [float(m.group()) for m in FLOAT_RE.finditer(line)]
                    if floats:
                        current.extend(floats)

        if current:
            token_vectors.append(current)

        if not token_vectors:
            logging.warning(f"No token vectors in {file_path}")
            return None

        dim0 = len(token_vectors[0])
        if not all(len(v) == dim0 for v in token_vectors):
            logging.warning(f"Inconsistent token dims in {file_path}, skipping.")
            return None

        return np.asarray(token_vectors, dtype=np.float32)

    except Exception:
        logging.exception(f"Failed parsing {file_path}")
        return None


def discover_subruns(root: Path) -> list[Path]:
    """
    Return subrun directories that contain 'p*/tensors/*.txt'.
    Handles:
      root/p*/tensors/*.txt                 -> treat 'root' as a subrun (name = root.name)
      root/<subrun>/p*/tensors/*.txt        -> each subdir is a subrun (name = subdir.name)
    """
    # case 1: root directly looks like a subrun
    if any(root.glob("p*/tensors/*.txt")):
        return [root]

    # case 2: children are subruns
    subruns = []
    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        if any(child.glob("p*/tensors/*.txt")):
            subruns.append(child)
    return subruns


def find_tensor_files_in_subrun(subrun_dir: Path) -> list[tuple[str, Path]]:
    """
    Return (key, path) for every l_out-*.txt inside this subrun.
    key = '<subrun>_<pXXXXX>_<stem>'  e.g. 'last_r1_p00001_l_out-23'
    """
    pairs = []
    subrun_name = subrun_dir.name
    for pdir in sorted(subrun_dir.glob("p*/tensors")):
        prompt_id = pdir.parent.name  # 'p00000'
        for f in sorted(pdir.glob("l_out-*.txt")):
            if f.stat().st_size == 0:
                logging.warning(f"Empty tensor file skipped: {f}")
                continue
            stem = f.stem  # 'l_out-23'
            key = f"{subrun_name}_{prompt_id}_{stem}"
            pairs.append((key, f))
    return pairs


def main():
    ap = argparse.ArgumentParser(
        description="Parse llama-eval-callback l_out dumps (all + last_r*) into a single .npy folder"
    )
    ap.add_argument(
        "--root",
        required=True,
        help="Root folder containing 'all/' and/or 'last_r*/' (e.g. gptoss_20b_truthfulqa)",
    )
    ap.add_argument("--output-dir", required=True, help="Directory to write .npy files")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="[%(asctime)s] %(levelname)s: %(message)s",
    )

    root = Path(args.root).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    subruns = discover_subruns(root)
    if not subruns:
        logging.error(f"No subruns with p*/tensors found under {root}")
        return

    logging.info(f"Discovered subruns: {[s.name for s in subruns]}")

    total = 0
    for sub in subruns:
        pairs = find_tensor_files_in_subrun(sub)
        if not pairs:
            logging.warning(f"No l_out-*.txt files in subrun {sub}")
            continue

        for key, path in pairs:
            logging.info(f"Parsing {path}")
            arr = parse_one_tensor_txt(path)
            if arr is None:
                continue
            out_path = out_dir / f"{key}.npy"
            np.save(out_path, arr)
            total += 1
            logging.info(f"Saved {out_path} shape={tuple(arr.shape)}")

    logging.info(f"Done. Saved {total} arrays to {out_dir}")


if __name__ == "__main__":
    main()
