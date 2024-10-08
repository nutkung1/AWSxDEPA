#!/usr/bin/env python
import pandas as pd
import json
import os

CHAR_LIMIT = 10000
CSV_FILE_PATH = "Data/jventures_crawl_results.csv"


def csv2json():
    # Read the CSV file
    df = pd.read_csv(CSV_FILE_PATH)

    # Ensure 'Title' and 'Content' columns exist
    required_columns = ["Title", "Content"]
    if not all(col in df.columns for col in required_columns):
        raise ValueError("CSV file must contain 'Title' and 'Content' columns")

    # Use 'Title' as the 'prompt' field
    df["prompt"] = df["Title"]

    # Use 'Content' as the 'completion' field
    df["completion"] = df["Content"]

    # Filter data based on character limits
    df = df[
        (df["prompt"].str.len() < CHAR_LIMIT)
        & (df["completion"].str.len() < CHAR_LIMIT)
        & (df["prompt"].str.len() + df["completion"].str.len() < CHAR_LIMIT)
    ]

    # Convert to JSON format required by AWS Bedrock
    def to_bedrock_format(row):
        return {"prompt": row["prompt"], "completion": row["completion"]}

    # Save all data to a single JSONL file
    json_data = df.apply(to_bedrock_format, axis=1).tolist()
    with open(
        "Data/jventures_train.jsonl", "w", encoding="ascii", errors="replace"
    ) as f:
        for item in json_data:
            # ensure_ascii=False removed since we're using ascii encoding
            f.write(json.dumps(item) + "\n")

    print(f"Conversion complete. Output file:")
    print("- jventures_train.jsonl")
    print(f"Total entries: {len(json_data)}")

    # Check for NaN or empty values in 'prompt' or 'completion'
    if df["prompt"].isna().any() or df["completion"].isna().any():
        print("Found NaN or empty values in 'prompt' or 'completion'")
        print(df[df["prompt"].isna() | df["completion"].isna()])


def main():
    csv2json()


if __name__ == "__main__":
    main()
