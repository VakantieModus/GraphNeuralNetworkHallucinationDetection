import argparse
import csv
import json
import logging
import os
import re

import numpy as np


def parse_tensor_files(tensor_dir, output_dir, prefix):
    os.makedirs(output_dir, exist_ok=True)
    file_pattern = re.compile(rf"{re.escape(prefix)}_tensor_output_(\d+)\.txt")
    float_re = re.compile(r"[-+]?\d*\.\d+([eE][-+]?\d+)?|[-+]?\d+")

    for filename in os.listdir(tensor_dir):
        match = file_pattern.match(filename)
        if not match:
            continue

        idx = int(match.group(1))
        file_path = os.path.join(tensor_dir, filename)
        logging.info(f"🔢 Processing tensor file {filename}...")

        token_vectors = []
        current_vector = []
        in_data_block = False

        with open(file_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("=== TOKEN"):
                    if current_vector:
                        token_vectors.append(current_vector)
                    current_vector = []
                    in_data_block = False
                elif line.startswith("DATA:"):
                    in_data_block = True
                elif line.startswith("SHAPE:") or line.startswith("--- TENSOR:"):
                    continue
                elif line.startswith("---") or line.startswith("==="):
                    in_data_block = False
                elif in_data_block:
                    try:
                        current_vector.extend(
                            [float(m.group()) for m in float_re.finditer(line)]
                        )
                    except ValueError:
                        logging.warning(f"⚠️ Malformed line in {filename}: {line}")

        if current_vector:
            token_vectors.append(current_vector)

        if not token_vectors:
            logging.warning(f"⚠️ No vectors found in {filename}")
            continue

        expected_dim = len(token_vectors[0])
        if not all(len(v) == expected_dim for v in token_vectors):
            logging.warning(f"⚠️ Inconsistent vector sizes in {filename}")
            continue

        array = np.array(token_vectors, dtype=np.float32)
        out_path = os.path.join(output_dir, f"tensor_output_{idx}.npy")
        np.save(out_path, array)
        logging.info(f"✅ Saved tensor to {out_path} (shape={array.shape})")


def parse_prompt_files(prompt_dir, prefix):
    results = []
    file_pattern = re.compile(rf"{re.escape(prefix)}_prompt_output_(\d+)\.txt$")

    for filename in os.listdir(prompt_dir):
        match = file_pattern.match(filename)
        if not match:
            continue

        idx = int(match.group(1))
        file_path = os.path.join(prompt_dir, filename)

        with open(file_path, encoding="utf-8", errors="replace") as f:
            content = f.read()
            match = re.search(
                r"Running prompt:\s*(.*?)\n\s*\nFull output:\s*(.*)", content, re.DOTALL
            )
            if match:
                prompt = match.group(1).strip()
                output = match.group(2).strip()
                results.append(
                    {"idx": idx, "file": filename, "prompt": prompt, "output": output}
                )

    results.sort(key=lambda x: x["idx"])
    return results


def load_enrichment_data_from_csv(csv_path):
    enrichment = {}
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            question = row["Question"].strip()
            enrichment[question] = row
    return enrichment


def enrich_prompt_data(data, enrichment_csv):
    enrichment_data = load_enrichment_data_from_csv(enrichment_csv)
    keys_to_copy = [
        "Best Answer",
        "Best Incorrect Answer",
        "Correct Answers",
        "Incorrect Answers",
        "Source",
        "Answer",
    ]

    for entry in data:
        question = entry["prompt"].strip()
        match = enrichment_data.get(question)
        if match:
            for key in keys_to_copy:
                if key in match:
                    entry[key] = match[key]
        else:
            logging.info(f"❗ No enrichment match for: {question}")

    return data


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logging.info(f"💾 Saved JSON to {path}")


def run_pipeline(input_dir, enrichment_csv):
    prefix = os.path.basename(os.path.normpath(input_dir))

    tensor_output_dir = os.path.join("generated_data", f"{prefix}_tensors_npy")
    prompt_output_path = os.path.join("data", f"parsed_output_{prefix}.json")
    enriched_output_path = os.path.join("data", f"enriched_output_{prefix}.json")

    logging.info(f"📂 Inferred prefix: {prefix}")

    parse_tensor_files(input_dir, tensor_output_dir, prefix)
    prompt_data = parse_prompt_files(input_dir, prefix)
    save_json(prompt_data, prompt_output_path)

    enriched = enrich_prompt_data(prompt_data, enrichment_csv)
    save_json(enriched, enriched_output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="White-box LLM output processing pipeline"
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Folder containing *_tensor_output_*.txt and *_prompt_output_*.txt",
    )
    parser.add_argument(
        "--enrichment-csv", required=True, help="CSV file with enrichment data"
    )
    parser.add_argument("--log-level", default="INFO", help="Log level")

    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="[%(asctime)s] %(levelname)s: %(message)s",
    )

    run_pipeline(args.input_dir, args.enrichment_csv)
