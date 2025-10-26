#!/usr/bin/env python3
"""
Unified DPR/KILT pipeline:
  1) Build docstore.jsonl from shards (id, title, text)
  2) Build docstore.offsets.npy
  3) Build FAISS index:
     - --embeddings precomputed  -> stream 'embeddings' from HF parquet
     - --embeddings encode       -> encode texts with DPR ctx encoder (legacy)

Outputs in --out-dir:
  - <prefix>_docstore.jsonl
  - <prefix>_docstore.offsets.npy
  - <prefix>_<ivfpq|flat>.faiss
"""

import argparse
import json
import logging
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path

import faiss
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import DPRContextEncoder, DPRContextEncoderTokenizerFast


def setup_logging(verbosity: int = 1):
    level = (
        logging.WARNING
        if verbosity <= 0
        else (logging.INFO if verbosity == 1 else logging.DEBUG)
    )
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )


@contextmanager
def section(title: str):
    logging.info(f"▶ {title} ...")
    t0 = time.time()
    try:
        yield
    finally:
        dt = time.time() - t0
        logging.info(f"✔ {title} done in {dt:.1f}s")


def fsize(path: Path) -> str:
    if not path.exists():
        return "0 B"
    n = path.stat().st_size
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def parse_shards(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def shard_uris(pattern: str, shards: list[int]) -> list[str]:
    return [pattern.format(i=s) for s in shards]


# ------------------- Stage 1: Docstore -------------------
def build_docstore(
    shards: list[int], parquet_pattern: str, out_jsonl: Path, max_docs: int | None
) -> int:
    """
    Write one JSON line per doc: {"id","title","text"} (id = 0..N-1 across shards)
    """
    ensure_parent(out_jsonl)
    doc_id = 0
    written = 0
    with out_jsonl.open("w", encoding="utf-8") as f_out:
        for sid in shards:
            uri = parquet_pattern.format(i=sid)
            logging.info(f"[docstore] loading shard {sid:05d} from {uri}")
            ds = load_dataset(
                "parquet", data_files=uri, split="train"
            )  # not streaming; loads per-shard only
            t0 = time.time()
            for row in tqdm(ds, desc=f"[shard {sid:05d}] write docstore", unit="doc"):
                obj = {"id": doc_id, "title": row["title"], "text": row["text"]}
                f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                doc_id += 1
                written += 1
                if max_docs and written >= max_docs:
                    break
            dt = time.time() - t0
            logging.info(
                f"[docstore] shard {sid:05d}: wrote {written} total (shard time {dt:.1f}s)"
            )
            if max_docs and written >= max_docs:
                break
    return written


def build_offsets(docstore_jsonl: Path, offsets_npy: Path) -> int:
    ensure_parent(offsets_npy)
    offsets = []
    with docstore_jsonl.open("rb") as f:
        pos = f.tell()
        line = f.readline()
        while line:
            offsets.append(pos)
            pos = f.tell()
            line = f.readline()
    arr = np.array(offsets, dtype=np.int64)
    np.save(offsets_npy, arr)
    logging.info(f"[offsets] wrote {len(arr)} -> {offsets_npy} ({fsize(offsets_npy)})")
    return len(arr)


# ------------------- Stage 3A: Index from *encoded* texts (legacy) -------------------
def dpr_context_models(device: str):
    tok = DPRContextEncoderTokenizerFast.from_pretrained(
        "facebook/dpr-ctx_encoder-single-nq-base"
    )
    enc = (
        DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
        .to(device)
        .eval()
    )
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    return tok, enc


def encode_passages(
    texts: list[str], tok, enc, device: str, max_len: int = 256
) -> np.ndarray:
    with torch.no_grad():
        toks = tok(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_len,
        ).to(device)
        embs = enc(**toks).pooler_output.detach().cpu().numpy().astype("float32")
        faiss.normalize_L2(embs)
        return embs


def iter_texts(docstore_jsonl: Path) -> Iterable[str]:
    with docstore_jsonl.open("r", encoding="utf-8") as f:
        for ln in f:
            yield json.loads(ln)["text"]


def train_ivfpq_from_texts(
    tok,
    enc,
    device: str,
    dim: int,
    docstore_jsonl: Path,
    nlist: int,
    pq_m: int,
    nbits: int,
    train_size: int,
    batch: int,
) -> faiss.IndexIVFPQ:
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFPQ(
        quantizer, dim, nlist, pq_m, nbits, faiss.METRIC_INNER_PRODUCT
    )
    train_vecs, buf, total = [], [], 0
    for txt in iter_texts(docstore_jsonl):
        buf.append(txt)
        if len(buf) == batch:
            train_vecs.append(encode_passages(buf, tok, enc, device))
            total += len(buf)
            buf = []
            if total >= train_size:
                break
    if buf:
        train_vecs.append(encode_passages(buf, tok, enc, device))
    Xtr = np.vstack(train_vecs)
    logging.info(
        f"[train] IVFPQ train vectors: {Xtr.shape[0]}, dim={dim}, nlist={nlist}, m={pq_m}, nbits={nbits}"
    )
    index.train(Xtr)
    return index


def add_all_vectors_from_texts(
    index, tok, enc, device: str, docstore_jsonl: Path, batch: int
) -> int:
    added, buf = 0, []
    t0 = time.time()
    for txt in tqdm(iter_texts(docstore_jsonl), desc="[encode+add]"):
        buf.append(txt)
        if len(buf) == batch:
            E = encode_passages(buf, tok, enc, device)
            index.add(E)
            added += E.shape[0]
            buf = []
    if buf:
        E = encode_passages(buf, tok, enc, device)
        index.add(E)
        added += E.shape[0]
    dt = time.time() - t0
    logging.info(
        f"[index] added {added} vectors in {dt:.1f}s ({added/max(dt,1e-6):.1f} vec/s)"  # noqa
    )
    return added


# ------------------- Stage 3B: Index from *precomputed* embeddings -------------------
def iter_embeddings_uri(uri: str, batch: int = 2048) -> Iterator[np.ndarray]:
    ds = load_dataset("parquet", data_files=uri, split="train", streaming=True)
    buf = []
    for row in ds:
        buf.append(np.asarray(row["embeddings"], dtype="float32"))
        if len(buf) == batch:
            X = np.vstack(buf)
            faiss.normalize_L2(X)
            yield X
            buf = []
    if buf:
        X = np.vstack(buf)
        faiss.normalize_L2(X)
        yield X


def sample_train_from_uris(
    uri_list: list[str], need: int, per_chunk: int = 4096, max_shards: int = 64
) -> np.ndarray:
    Xs, total = [], 0
    for sidx, uri in enumerate(uri_list):
        for X in iter_embeddings_uri(uri, batch=per_chunk):
            take = min(X.shape[0], need - total)
            if take > 0:
                Xs.append(X[:take])
                total += take
            if total >= need:
                return np.vstack(Xs)
        if sidx + 1 >= max_shards:
            break
    if not Xs:
        raise RuntimeError("No training vectors sampled.")
    return np.vstack(Xs)


# ------------------- Main -------------------
def main():
    ap = argparse.ArgumentParser(
        description="Build DPR docstore + offsets + FAISS index"
    )

    ap.add_argument(
        "--shards",
        type=str,
        default="0",
        help="Comma-separated shard ids (e.g. '0,1,2') or ranges '0-9'.",
    )
    ap.add_argument(
        "--parquet-pattern",
        type=str,
        default="hf://datasets/facebook/wiki_dpr/data/psgs_w100/nq/train-{i:05d}-of-00157.parquet",
    )
    ap.add_argument("--max-docs", type=int, default=None)

    ap.add_argument(
        "--out-dir",
        type=str,
        default="src/detection_methods/kilt_retriever/knowledge_source",
    )
    ap.add_argument("--out-prefix", dest="out_prefix", type=str, default="wiki_small")
    ap.add_argument("--docstore", type=str, default=None)
    ap.add_argument("--offsets", type=str, default=None)
    ap.add_argument("--index", type=str, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verbose", "-v", action="count", default=1)

    ap.add_argument(
        "--embeddings",
        choices=["precomputed", "encode"],
        default="precomputed",
        help="Use 'precomputed' HF embeddings or 'encode' texts with DPR context model.",
    )
    ap.add_argument("--index-type", choices=["ivfpq", "flat"], default="ivfpq")
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--nlist", type=int, default=8192)
    ap.add_argument("--pq-m", type=int, default=32)
    ap.add_argument("--nbits", type=int, default=8)
    ap.add_argument("--train-size", type=int, default=500_000)
    ap.add_argument("--nprobe", type=int, default=16)
    ap.add_argument(
        "--checkpoint-every-shards",
        type=int,
        default=1,
        help="Write index after every N shards when using precomputed embeddings (0=disable)",
    )

    ap.add_argument(
        "--device", type=str, default=None, help="cuda|mps|cpu (auto if not set)"
    )
    args = ap.parse_args()

    setup_logging(args.verbose)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "ivfpq" if args.index_type == "ivfpq" else "flat"
    safe_prefix = (args.out_prefix or "wiki_small").replace("-", "_")

    docstore_path = (
        Path(args.docstore)
        if args.docstore
        else out_dir / f"{safe_prefix}_docstore.jsonl"
    )
    offsets_path = (
        Path(args.offsets)
        if args.offsets
        else out_dir / f"{safe_prefix}_docstore.offsets.npy"
    )
    index_path = (
        Path(args.index) if args.index else out_dir / f"{safe_prefix}_{suffix}.faiss"
    )

    logging.info(f"[paths] out_dir={out_dir.resolve()}")
    logging.info(
        f"[paths] docstore={docstore_path.name}  offsets={offsets_path.name}  index={index_path.name}"
    )

    def maybe_unlink(p: Path):
        if p.exists() and args.overwrite:
            logging.warning(f"[overwrite] removing existing {p}")
            p.unlink()

    shard_ids = parse_shards(args.shards)
    if not shard_ids:
        raise SystemExit("No shards specified.")
    logging.info(f"[shards] {shard_ids}")
    logging.info(f"[pattern] {args.parquet_pattern}")
    uris = shard_uris(args.parquet_pattern, shard_ids)

    device = args.device or pick_device()
    logging.info(f"[device] {device}")

    # Stage 1: Docstore
    with section("Build docstore"):
        if args.overwrite:
            maybe_unlink(docstore_path)
        if not docstore_path.exists():
            total = build_docstore(
                shard_ids, args.parquet_pattern, docstore_path, args.max_docs
            )
            logging.info(
                f"[docstore] wrote {total} -> {docstore_path} ({fsize(docstore_path)})"
            )
        else:
            logging.info(
                f"[docstore] exists -> {docstore_path} ({fsize(docstore_path)})"
            )

    # Stage 2: Offsets
    with section("Build offsets"):
        if args.overwrite:
            maybe_unlink(offsets_path)
        if not offsets_path.exists():
            n_off = build_offsets(docstore_path, offsets_path)
            logging.info(f"[offsets] count={n_off}")
        else:
            logging.info(f"[offsets] exists -> {offsets_path} ({fsize(offsets_path)})")

        # quick validation
        n_lines = sum(1 for _ in open(docstore_path, "rb"))
        n_offs = np.load(offsets_path).shape[0]
        if n_lines != n_offs:
            logging.warning(f"[validate] doc lines ({n_lines}) != offsets ({n_offs})")
        else:
            logging.info(f"[validate] doc lines == offsets == {n_offs}")

    # Stage 3: FAISS index
    with section("Build FAISS index"):
        if args.overwrite:
            maybe_unlink(index_path)

        if args.index_type == "ivfpq":
            quantizer = faiss.IndexFlatIP(args.dim)

            if args.embeddings == "precomputed":
                # Train from precomputed embeddings
                logging.info("[train] sampling precomputed embeddings ...")
                Xtr = sample_train_from_uris(
                    uris, need=args.train_size, per_chunk=4096, max_shards=64
                )
                logging.info(f"[train] sampled {Xtr.shape} vectors")
                index = faiss.IndexIVFPQ(
                    quantizer,
                    args.dim,
                    args.nlist,
                    args.pq_m,
                    args.nbits,
                    faiss.METRIC_INNER_PRODUCT,
                )
                index.train(Xtr)
                index.nprobe = int(args.nprobe)

                # Add all vectors, shard by shard
                added = 0
                for si, uri in enumerate(uris, start=1):
                    shard_added = 0
                    for X in iter_embeddings_uri(uri, batch=args.batch):
                        index.add(X)
                        shard_added += X.shape[0]
                        added += X.shape[0]
                    logging.info(
                        f"[add] shard {si:03d}/{len(uris)} {uri.split('/')[-1]}  +{shard_added}  ntotal={index.ntotal}"
                    )
                    if args.checkpoint_every_shards and (
                        si % args.checkpoint_every_shards == 0
                    ):
                        faiss.write_index(index, str(index_path))
                        logging.info(
                            f"[checkpoint] wrote {index_path} (ntotal={index.ntotal})"
                        )

            else:
                # Train+add by encoding texts (legacy path)
                tok, enc = dpr_context_models(device)
                index = faiss.IndexIVFPQ(
                    quantizer,
                    args.dim,
                    args.nlist,
                    args.pq_m,
                    args.nbits,
                    faiss.METRIC_INNER_PRODUCT,
                )
                index = train_ivfpq_from_texts(
                    tok,
                    enc,
                    device,
                    args.dim,
                    docstore_path,
                    nlist=args.nlist,
                    pq_m=args.pq_m,
                    nbits=args.nbits,
                    train_size=args.train_size,
                    batch=args.batch,
                )
                index.nprobe = int(args.nprobe)
                added = add_all_vectors_from_texts(
                    index, tok, enc, device, docstore_path, batch=args.batch
                )

        else:
            # Flat index (IP)
            if args.embeddings == "precomputed":
                index = faiss.IndexFlatIP(args.dim)
                added = 0
                for uri in uris:
                    for X in iter_embeddings_uri(uri, batch=args.batch):
                        index.add(X)
                        added += X.shape[0]
            else:
                tok, enc = dpr_context_models(device)
                index = faiss.IndexFlatIP(args.dim)
                added = add_all_vectors_from_texts(
                    index, tok, enc, device, docstore_path, batch=args.batch
                )

        # persist index
        faiss.write_index(index, str(index_path))
        logging.info(f"[faiss] wrote -> {index_path} ({fsize(index_path)})")

        # info & validation
        if hasattr(index, "nlist"):
            logging.info(
                f"[ivfpq] nlist={index.nlist}, nprobe={getattr(index,'nprobe',None)}, code_size={getattr(index,'code_size','?')}, ntotal={index.ntotal}"  # noqa
            )
        else:
            logging.info(f"[flat] ntotal={index.ntotal}")

        # final check: ntotal vs offsets
        n_offs = np.load(offsets_path).shape[0]
        if getattr(index, "ntotal", 0) != n_offs:
            logging.warning(
                f"[validate] index.ntotal ({getattr(index,'ntotal',0)}) != offsets ({n_offs})"  # noqa
            )
        else:
            logging.info(f"[validate] index.ntotal == offsets == {n_offs}")

    logging.info("🎉 All done.")


if __name__ == "__main__":
    main()
