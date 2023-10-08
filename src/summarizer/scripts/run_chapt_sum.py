"""Summarise chapter text."""
import os
import json
import time
import logging
import argparse
import jsonlines
import os.path
from pathlib import Path
from summarizer.util import SRC_RESOURCES_DIR
from summarizer.model.summarizer import TextSummarizer, SummarizerFactory

CHAPTER_DIR = os.path.join(SRC_RESOURCES_DIR, "chapter")
SUMMARY_DIR = os.path.join(SRC_RESOURCES_DIR, "summary")
BOOKSUM_DIR = os.path.join(SRC_RESOURCES_DIR, "booksum")


def summarize_chapter(chapter_dir: str, summary_dir: str):
    """Summarize chapter text and save the summary."""
    chapter_path = Path(chapter_dir).resolve()
    summary_path = Path(summary_dir).resolve()
    Path(summary_path).mkdir(parents=True, exist_ok=True)

    gpt3_summarizer = SummarizerFactory.create_summarizer("gpt3")
    text_summarizer = TextSummarizer(gpt3_summarizer)

    for chapter in os.listdir(chapter_path):
        if not chapter.lower().endswith(".txt"):
            continue

        chap_file_path = os.path.join(chapter_path, chapter)
        sum_text = text_summarizer.summarize_file(chap_file_path)

        sum_file_path = os.path.join(summary_path, chapter)
        with open(sum_file_path, "w") as file_obj:
            file_obj.write(sum_text)
            logger.info(f"Summary save into: {sum_file_path}")


def summarize_chapter_batch(input_file: str, output_file: str):
    gpt3_summarizer = SummarizerFactory.create_summarizer("gpt3")
    text_summarizer = TextSummarizer(gpt3_summarizer)

    with jsonlines.open(output_file, mode="w") as out_file:
        line_count = 0
        for line in jsonlines.open(input_file, mode="r"):
            line_count += 1
            doc_id = line["metadata"]["chapter_path"] + "-" + line["metadata"]["summary_path"]
            logger.info(f"Summarizing line# {line_count} document: {doc_id}")
            line["system_sum"] = text_summarizer.summarize_text(line["text"], doc_id)
            out_file.write(line)

            if line_count % 100 == 0:
                logger.info(f"Sleeping for 30s at ... {line_count}")
                time.sleep(30)


def summarize_chapter_text(chapter_text: str):
    gpt3_summarizer = SummarizerFactory.create_summarizer("gpt3")
    text_summarizer = TextSummarizer(gpt3_summarizer)
    return text_summarizer.summarize_text(chapter_text)


def summarize_book(book_path: str):
    with open(book_path, encoding="utf8") as fp:
        chap_data = json.load(fp)
        chapters = [chapter["text"] for chapter in chap_data["chapters"]]
        book_id = chap_data["book_id"]

    primary_summarizer = SummarizerFactory.create_summarizer("gpt4")
    secondary_summarizer = SummarizerFactory.create_summarizer("gpt3.5")
    text_summarizer = TextSummarizer(primary_summarizer, secondary_summarizer)

    summary = text_summarizer.summarize_chapters(chapters, book_id)
    print(json.dumps(summary, indent=4, sort_keys=False))


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(
        # filename="gpt-log.txt",
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger()
    parser = argparse.ArgumentParser(description="Command line utility for running chapter summarizer.")
    parser.add_argument("--chapter_dir", type=str, default=CHAPTER_DIR)
    parser.add_argument("--summary_dir", type=str, default=SUMMARY_DIR)

    args = parser.parse_args()
    logger.info(f"Input args: {vars(args)}")

    # summarize_chapter(args.chapter_dir, args.summary_dir)
    # summarize_chapter_batch(
    #     os.path.join(BOOKSUM_DIR, "booksum-10chapt.jsonl"),
    #     os.path.join(BOOKSUM_DIR, "booksum-10chapt-sum.jsonl")
    # )
    start_time = time.perf_counter()
    summarize_book(os.path.join(CHAPTER_DIR, "1232-chapters_19.txt.json"))
    logger.info(f"Execution time: {(time.perf_counter() - start_time) / 60:.2f} mins")
