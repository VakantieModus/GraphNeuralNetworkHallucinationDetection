import json
import logging
import os

import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


def load_embedding_eigenscore(idx: int, tensor_layer: int):
    """
    Load last-token vectors for one (idx, layer) across:
      ["all", "last_r0", "last_r1", "last_r2", "last_r3", "last_r4"].
    Assumes each .npy has shape (T, D): rows=tokens, cols=hidden dims.
    Returns: List[np.ndarray] of shape (D,) in the order above.
    """
    iterations = ("all", "last_r0", "last_r1", "last_r2", "last_r3", "last_r4")
    _path_tensors = "/Users/casperdert/PycharmProjects/MasterThesis/generated_data/gptoss_20b_tqa_npy"
    p = f"{idx:05d}"
    root = _path_tensors
    vecs = []

    for it in iterations:
        # Support flat and per-iteration subfolder layouts (underscore variant your parser produces)
        candidates = [
            os.path.join(root, f"{it}_p{p}_l_out-{tensor_layer}.npy"),
            os.path.join(root, it, f"p{p}_l_out-{tensor_layer}.npy"),
        ]
        fpath = next((c for c in candidates if os.path.exists(c)), None)
        if fpath is None:
            logging.warning(
                f"[eigenscore] missing file for {it}, idx=p{p}, L={tensor_layer} at candidates={candidates}"
            )
            continue

        try:
            arr = np.load(fpath)
            if arr.ndim != 2:
                logging.warning(
                    f"[eigenscore] unexpected shape {arr.shape} in {os.path.basename(fpath)}"
                )
                continue
            # last row = last token vector (D,)
            vecs.append(arr[-1, :].astype(np.float64, copy=False))
        except Exception as e:
            logging.warning(f"[eigenscore] failed {fpath}: {e}")

    return vecs


if __name__ == "__main__":
    _path_annotations = (
        "/Users/casperdert/PycharmProjects/MasterThesis/src/white_box_benchmark/"
        "data/generated_output/enriched_gptoss_20b_tqa.json"
    )
    _last_layer = 23
    with open(_path_annotations) as f:
        data = json.load(f)
        data = [item for item in data]

    for item in tqdm(data, desc="Loading Tensor Data"):
        idx = item["idx"]

        embedding = load_embedding_eigenscore(idx, _last_layer)
        # Stack into (num_iterations, D)
        try:
            arr = np.stack(embedding, axis=0).astype(np.float32)
            out_dir = "/Users/casperdert/PycharmProjects/MasterThesis/generated_data/gptoss_20b_tqa_npy"
            out_path = os.path.join(
                out_dir, f"eigenscore_p{idx:05d}_l_out-{_last_layer}.npy"
            )
            np.save(out_path, arr)
        except Exception as e:
            logging.warning(f"[eigenscore] failed {item}: {e}")
