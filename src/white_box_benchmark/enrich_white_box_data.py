import csv
import json
import logging
import os

logging.basicConfig(level=logging.INFO)


def load_enrichment_data_from_csv(path):
    questions_data = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            question = row["Question"].strip()
            questions_data[question] = row
    return questions_data


if __name__ == "__main__":
    # File paths
    json_path = (
        "/Users/casperdert/PycharmProjects/MasterThesis/src/white_box_benchmark/data/"
        "parsed_output_gptoss_20b_eli5.json"
    )
    questions_path = (
        "/Users/casperdert/PycharmProjects/MasterThesis/src/white_box_benchmark/"
        "data/first_700_questions_eli5.csv"
    )
    output_path = (
        "/Users/casperdert/PycharmProjects/MasterThesis/src/white_box_benchmark/data/"
        "generated_output/parsed_enriched_gptoss_20b_eli5.json"
    )

    # Load the main JSON to be enriched
    with open(json_path, encoding="utf-8") as f:
        json_data = json.load(f)

    # Load enrichment data from CSV
    questions_data = load_enrichment_data_from_csv(questions_path)

    # Fields to copy over if present
    enrichment_keys = [
        "Best Answer",
        "Best Incorrect Answer",
        "Correct Answers",
        "Incorrect Answers",
        "Source",
        "Answer",
    ]

    # Enrich JSON entries
    for entry in json_data:
        question = entry["prompt"].strip()
        question = question.partition("Question: ")[2].splitlines()[0].strip()
        enrichment = questions_data.get(question)
        if enrichment:
            for key in enrichment_keys:
                if key in enrichment:
                    entry[key] = enrichment[key]
        else:
            logging.info(f"❗ No match for: {question}")

    # Save the enriched JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    logging.info(f"✅ Enriched JSON saved to {output_path}")
