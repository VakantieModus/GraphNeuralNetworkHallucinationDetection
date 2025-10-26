import argparse
import csv
import json
import logging
import os
import re


def parse_prompt_files(directory):
    parsed_data = []
    for filename in os.listdir(directory):
        match_filename = re.match(r"^p(\d+)_prompt_output\.txt$", filename)
        if not match_filename:
            continue

        idx = int(match_filename.group(1))
        file_path = os.path.join(directory, filename)

        with open(file_path, encoding="utf-8", errors="replace") as f:
            content = f.read()

            match = re.search(
                r"Running prompt:\s*(.*?)\n\s*\nFull output:\s*(.*)", content, re.DOTALL
            )
            if match:
                prompt = match.group(1).strip()
                output = match.group(2).strip()
                parsed_data.append(
                    {"idx": idx, "file": filename, "prompt": prompt, "output": output}
                )

    return parsed_data


def save_to_json(data, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logging.info(f"Saved to {output_file}")


def save_to_csv(data, output_file):
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["idx", "file", "prompt", "output"])
        writer.writeheader()
        writer.writerows(data)
    logging.info(f"Saved to {output_file}")


# Run it
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse prompt_output_*.txt files into .npy arrays"
    )

    parser.add_argument(
        "--log-level",
        "-l",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    #
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="[%(asctime)s] %(levelname)s: %(message)s",
    )

    folder = "/Users/casperdert/PycharmProjects/MasterThesis/generated_data_2/gptoss_20b_eli5/all"
    data = parse_prompt_files(folder)

    # Sort by idx for consistency
    data.sort(key=lambda x: x["idx"])

    # Save options
    save_to_json(data, "data/parsed_output_gptoss_20b_eli5.json")
