"""A python script to hit a SageMaker endpoint."""
import os
import sys
import time
import json
import jsonlines
import boto3
import logging
import argparse as ap
from typing import Dict
from summarizer.util import SRC_RESOURCES_DIR
from botocore.config import Config

BOOKSUM_DIR = os.path.join(SRC_RESOURCES_DIR, "booksum")
CHAPTER_DIR = os.path.join(SRC_RESOURCES_DIR, "chapter")
ENDPOINT_NAME = "chpater-sum-gpt3-apr23-endpoint"
config = Config(
    read_timeout=300,
    retries={"max_attempts": 0},  # This value can be adjusted to 5 to go up to the 360s max timeout
)
SM_RUNTIME = boto3.client("sagemaker-runtime")


def invoke_sm_endpoint_v2(endpoint_name: str, data: Dict) -> str:
    """Invoke the endpoint for the given data and return the response."""
    payload_json = json.dumps(data)

    response = SM_RUNTIME.invoke_endpoint(
        EndpointName=endpoint_name,
        Body=payload_json.encode('utf-8'),
        ContentType="application/json",
        Accept="application/json"
    )
    response_body = response["Body"].read().decode("utf-8")
    summary = json.loads(response_body)
    return summary


def summarize_book_endpoint(endpoint_name: str, input_file: str, output_file: str):
    with open(input_file, encoding="utf8") as fp:
        chap_data = json.load(fp)

    summary = invoke_sm_endpoint_v2(endpoint_name, chap_data)
    logger.info(json.dumps(summary, indent=4, sort_keys=False))

    with open(output_file, encoding="utf8", mode="w") as fp:
        json.dump(summary, fp)


def invoke_sm_endpoint(endpoint_name: str, long_text: str, doc_id: str) -> str:
    """Invoke the endpoint for the given data and return the response."""
    payload_json = json.dumps({"text": long_text, "doc_id": doc_id})

    response = SM_RUNTIME.invoke_endpoint(
        EndpointName=endpoint_name,
        Body=payload_json.encode('utf-8'),
        ContentType="application/json",
        Accept="application/json"
    )
    response_body = response["Body"].read().decode("utf-8")
    summary_text = json.loads(response_body)["summary"]
    return summary_text


def summarize_chapters_endpoint(endpoint_name: str, input_file: str, output_file: str):
    with jsonlines.open(output_file, mode="w") as out_file:
        line_count = 0
        for line in jsonlines.open(input_file, mode="r"):
            line_count += 1
            doc_id = line["metadata"]["chapter_path"] + "-" + line["metadata"]["summary_path"]
            logger.info(f"Summarizing line# {line_count} document: {doc_id}")
            line["system_sum"] = invoke_sm_endpoint(endpoint_name, line["text"], doc_id)
            out_file.write(line)

            if line_count % 100 == 0:
                logger.info(f"Sleeping for 30s at ... {line_count}")


def _get_args(args):
    """Handles CLI help and arguments."""
    parser = ap.ArgumentParser(
        prog=os.path.basename(__file__),
        description="CLI script to invoke the SageMaker endpoint"
    )

    parser.add_argument(
        "--endpoint_name", type=str, metavar="endpoint-name", default=ENDPOINT_NAME,
        help="The endpoint name to invoke."
    )
    parser.add_argument(
        "--input_file", type=str, metavar="input-file",
        default=os.path.join(CHAPTER_DIR, "1232-chapters_19.txt.json"),
        help="A JSON Lines input file where each line is a JSON object."
    )
    parser.add_argument(
        "--output_file", type=str, metavar="input-file",
        default=os.path.join(CHAPTER_DIR, "61-chapters-sum.json"),
        help="A JSON Lines output file where each line is a JSON object."
    )

    return parser.parse_args(args[1:])


def main(args):
    """The main function."""
    args = _get_args(args)
    logging.info(f"Provided args: {args}")

    try:
        start_time = time.perf_counter()
        summarize_book_endpoint(args.endpoint_name, args.input_file, args.output_file)
    except Exception as error:
        logging.error(f"Error while invoking the endpoint: {str(error)}")
    finally:
        logger.info(f"Execution time: {(time.perf_counter() - start_time) / 60:.2f} mins")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger()

    main(sys.argv)
