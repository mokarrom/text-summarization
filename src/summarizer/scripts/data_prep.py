"""Data preparation script."""
import json
import logging
import jsonlines
import pandas as pd
from typing import Dict


def process_row(row: pd.Series) -> Dict:
    return {
        "text": row["chapter"],
        "summary": row["summary_text"],
        "metadata": {
            "bid": row["bid"],
            "is_aggregate": row["is_aggregate"],
            "source": row["source"],
            "chapter_path": row["chapter_path"],
            "summary_path": row["summary_path"],
            "book_id": row["book_id"],
            "summary_id": row["summary_id"],
            "chapter_length": int(row["chapter_length"]),
            "summary_name": row["summary_name"],
            "summary_url": row["summary_url"],
            "summary_analysis": row["summary_analysis"],
            "summary_length": int(row["summary_length"]),
            "analysis_length": int(row["analysis_length"])
        }
    }


def prepare_train_val_data(channel_name: str):
    file_path = f"D:/UpWork/my-data/{channel_name}.csv"
    df = pd.read_csv(file_path, encoding="utf-8")

    row_count = 0
    with jsonlines.open(f"booksum-{channel_name}.jsonl", "w") as file_obj:
        for row in df.iterrows():
            file_obj.write(process_row(row[1]))
            row_count += 1

    print(f"Rows = {row_count}; DF shape={df.shape}")


def prepare_test_data():
    file_path = "D:/UpWork/my-data/test.csv"
    df = pd.read_csv(file_path, encoding="utf-8")

    unique_chapters = dict()
    remaining_chapters = list()
    test_file = jsonlines.open("booksum-test.jsonl", "w")
    for row in df.iterrows():
        row = row[1]
        chapter_path = row['chapter_path']
        file_name = chapter_path.replace(".", "/").split("/")[-2]
        if file_name.isdigit() and chapter_path not in unique_chapters:
            unique_chapters[chapter_path] = row
        else:
            remaining_chapters.append(row)
        test_file.write(process_row(row))
    test_file.close()

    count = 0
    with jsonlines.open("booksum-test_unique-chapters.jsonl", "w") as file_obj:
        for row in unique_chapters.values():
            file_obj.write(process_row(row))
            count += 1
            if count == 10:
                break

    with jsonlines.open("booksum-test_remaining-chapters.jsonl", "w") as file_obj:
        for row in remaining_chapters:
            file_obj.write(process_row(row))

    print(f"unique = {len(unique_chapters)}, remaining = {len(remaining_chapters)}; DF shape={df.shape}")


def split_remaining_chapters():
    file_path = "D:/UpWork/my-data/booksum-test_remaining-chapters.jsonl"

    chpt_file = jsonlines.open("booksum-test_remaining-chapters_chapts.jsonl", "w")
    section_file = jsonlines.open("booksum-test_remaining-chapters_sections.jsonl", "w")
    for line in jsonlines.open(file_path):
        chapter_path = line["metadata"]["chapter_path"]
        file_name = chapter_path.replace(".", "/").split("/")[-2]
        if file_name.isdigit():
            chpt_file.write(line)
        else:
            section_file.write(line)
    chpt_file.close()
    section_file.close()


def extract_multiple_sums():
    file_path = "D:/UpWork/my-data/test.csv"
    df = pd.read_csv(file_path, encoding="utf-8")

    chapt_sum = dict()
    for row in df.iterrows():
        row = row[1]
        chapter_path = row['chapter_path']
        if chapter_path in chapt_sum:
            chapt_sum[chapter_path].append(row["summary_text"])
        else:
            chapt_sum[chapter_path] = [row["summary_text"]]

    multi_sum = dict()
    for key, val in chapt_sum.items():
        if len(val) == 2:
            multi_sum[key] = val

    with open("sum-dict.json", "w") as file_obj:
        json.dump(multi_sum, file_obj)

    print(f"#Chapters = {len(multi_sum)}")


def load_and_test_data(channel_name: str):
    file_path = f"D:/UpWork/chapter_summarization_api/src/summarizer/scripts/booksum-{channel_name}.jsonl"

    line_count = 0
    with jsonlines.open(file_path) as file_obj:
        for line in file_obj:
            line_count += 1

    with open(file_path, encoding="utf-8") as fp:
        data = list(fp)

    assert len(data) == line_count

    df = pd.read_json(path_or_buf=file_path, lines=True)
    assert df.shape == (line_count, 3)


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    # prepare_test_data()
    # channel_name = "test"
    # # prepare_train_val_data(channel_name)
    # load_and_test_data(channel_name)

    # extract_multiple_sums()

    # split_remaining_chapters()

    count = 0
    for line in jsonlines.open("D:/UpWork/my-data/booksum-test_remaining-chapters.jsonl"):
        count += 1
    print(f"\n rows: {count}")
