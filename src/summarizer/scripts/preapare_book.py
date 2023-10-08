import json
import os
import logging
import argparse

ENCODING = "UTF8"


def prepare_book(book_path: str):
    book_name = book_path.split("\\")[-1]

    chapters = []
    for chapter in os.listdir(book_path):
        chapter_name = chapter.split(".")[0]
        if chapter_name.isnumeric():
            chapt_file = os.path.join(book_path, chapter)
            with open(chapt_file, "r", encoding=ENCODING) as file_obj:
                text = file_obj.read()
                chapters.append({
                    "id": int(chapter_name),
                    "text": text
                })

    chapters.sort(key=lambda x: x["id"])
    with open(os.path.join(book_path, book_name + ".json"), "w", encoding="utf-8") as fp:
        json.dump({"chapters": chapters, "book_id": book_name}, fp)


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(
        # filename="gpt-log.txt",
        level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger()

    parser = argparse.ArgumentParser(description="Command line utility for running chapter summarizer.")
    parser.add_argument("--book_path", type=str, default="D:\\UpWork\\booksum\\all_chapterized_books\\61-chapters")

    args = parser.parse_args()
    logger.info(f"Input args: {vars(args)}")

    prepare_book(args.book_path)
