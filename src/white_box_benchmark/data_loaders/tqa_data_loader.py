import json
import logging
import os

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


class DataLoader:
    def __init__(
        self, path_tensors: str, path_annotations: str, last_layer: int, curves_dir: str
    ) -> None:
        self._path_tensors = path_tensors
        self._path_annotations = path_annotations
        self._last_layer = last_layer
        self.curves_dir = curves_dir
        self.skip_questions = (
            [
                2,
                78,
                94,
                135,
                244,
                317,
                318,
                372,
                419,
                431,
                446,
                452,
                509,
                517,
                523,
                537,
                648,
                653,
            ]
            if "eli5" in path_annotations
            else []
        )

    def load_embedding_eigenscore(self, idx: int, tensor_layer: int):
        """
        Load last-token vectors for one (idx, layer) across:
          ["all", "last_r0", "last_r1", "last_r2", "last_r3", "last_r4"].
        Assumes each .npy has shape (T, D): rows=tokens, cols=hidden dims.
        Returns: List[np.ndarray] of shape (D,) in the order above.
        """

        p = f"{idx:05d}"
        root = self._path_tensors

        # support both flat and subdir layouts
        candidates = [
            os.path.join(root, f"eigenscore_p{p}_l_out-{tensor_layer}.npy"),
            os.path.join(root, "eigenscore", f"p{p}_l_out-{tensor_layer}.npy"),
        ]
        fpath = next((c for c in candidates if os.path.exists(c)), None)
        if fpath is None:
            raise FileNotFoundError(
                f"No 'all' tensor file found for idx={p}, layer={tensor_layer}"
            )

        arr = np.load(fpath)
        return arr

    def load_exact_tokes(self, idx: int) -> list[str]:

        fname = os.path.join(self._path_tensors, f"p{idx:05d}/generated_ids.json")
        if not os.path.exists(fname):
            return None
        with open(fname) as f:
            generated_ids = json.load(f)
        return generated_ids["tokens_generated"]

    def load_curve_data_lens(self, idx: int):
        """
        Load precomputed curve data for a given idx.
        Returns (logit_curve, tuned_curve, curve_concat) as float32 arrays.
        """
        fname = os.path.join(self.curves_dir, f"curves_idx{idx:05d}.npz")
        if not os.path.exists(fname):
            raise FileNotFoundError(f"Curve file not found: {fname}")

        rec = np.load(fname)
        logit_curves = rec["logit_curves"].astype(np.float32)  # (T, L)
        logit_curve = logit_curves[-1].astype(np.float32)  # (L,)
        return logit_curves, logit_curve

    def load_data_set(self, n=None, text_only=False):
        # Open and load the JSON file
        with open(self._path_annotations) as f:
            data = json.load(f)
            data = [
                item
                for item in data
                if item.get("Hallucinating") in [0, 1]
                and item.get("idx") not in self.skip_questions
            ]
        logging.info(f"Looking for tensors in {self._path_tensors}")
        for item in tqdm(data, desc="Loading Tensor Data"):
            idx = item["idx"]
            if text_only:
                continue
            item["tensor_last_layer"] = self.load_tensor_by_idx(idx, self._last_layer)
            item["embedding_eigenscore"] = self.load_embedding_eigenscore(
                idx, self._last_layer
            )
            logit_curves, logit_curve = self.load_curve_data_lens(idx)
            item["logit_curve"] = logit_curve
            item["logit_curves"] = logit_curves

        labels = [item["Hallucinating"] for item in data]

        # First split: train_val vs test (stratified)
        train_val, test = train_test_split(
            data, test_size=0.3, random_state=87, shuffle=True, stratify=labels
        )

        # Second split: train vs validation (also stratified)
        train_val_labels = [item["Hallucinating"] for item in train_val]
        train, val = train_test_split(
            train_val,
            test_size=0.1,
            random_state=42,
            shuffle=True,
            stratify=train_val_labels,
        )

        self.log_data_set_statistics(train_val, "TRAIN")
        self.log_data_set_statistics(val, "VALIDATION")
        self.log_data_set_statistics(test, "TEST")
        return train_val, val, test

    def log_data_set_statistics(self, data: list[dict], type: str = None):
        num_hallucinations = sum(1 for d in data if d.get("Hallucinating") == 1)
        logging.info(
            f" Data Statistics: {type} -- {num_hallucinations}/{len(data)} samples marked as hallucinating"
        )

    def load_tensor_by_idx(self, idx: int, last_layer: int):
        """
        Load the 'all' run tensor for one question idx and given layer.
        Returns the full array (T, D).
        """
        p = f"{idx:05d}"
        root = self._path_tensors

        # support both flat and subdir layouts
        candidates = [
            os.path.join(root, f"all_p{p}_l_out-{last_layer}.npy"),
            os.path.join(root, "all", f"p{p}_l_out-{last_layer}.npy"),
        ]
        fpath = next((c for c in candidates if os.path.exists(c)), None)
        if fpath is None:
            raise FileNotFoundError(
                f"No 'all' tensor file found for idx={p}, layer={last_layer}"
            )

        arr = np.load(fpath)
        if arr.ndim != 2:
            raise ValueError(
                f"Unexpected shape {arr.shape} in {fpath}, expected (T, D)"
            )

        return arr


class PretrainedEmbeddingLoader(DataLoader):
    def __init__(
        self,
        path_annotations: str,
        model_name: str = "allenai/longformer-large-4096",
        cache_dir="cached_embeddings",
    ):
        super().__init__(path_tensors=None, path_annotations=path_annotations)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def load_tensor_by_idx(self, idx):
        cache_path = os.path.join(self.cache_dir, f"embedding_{idx}.npy")
        if os.path.exists(cache_path):
            return np.load(cache_path)

        with open(self._path_annotations) as f:
            data = json.load(f)
            item = next(d for d in data if d["idx"] == idx)
            text = item[
                "output"
            ]  # Adjust this if your input text field is named differently

        # Tokenize and encode
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            hidden_states = (
                outputs.last_hidden_state.squeeze(0).cpu().numpy()
            )  # shape: (seq_len, hidden_dim)

        np.save(cache_path, hidden_states)
        return hidden_states
