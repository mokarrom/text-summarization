"""A python script to hit a SageMaker endpoint."""
import os
import sys
import json
import jsonlines
import boto3
import logging
import argparse as ap
from summarizer.util import SRC_RESOURCES_DIR

BOOKSUM_DIR = os.path.join(SRC_RESOURCES_DIR, "booksum")
ENDPOINT_NAME = "chpater-sum-gpt3-endpoint"
SM_RUNTIME = boto3.client("sagemaker-runtime")


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
        default=os.path.join(BOOKSUM_DIR, "booksum-10chapt.jsonl"),
        help="A JSON Lines input file where each line is a JSON object."
    )
    parser.add_argument(
        "--output_file", type=str, metavar="input-file",
        default=os.path.join(BOOKSUM_DIR, "booksum-10chapt-sum.jsonl"),
        help="A JSON Lines output file where each line is a JSON object."
    )

    return parser.parse_args(args[1:])


def main(args):
    """The main function."""
    args = _get_args(args)
    logging.info(f"Provided args: {args}")

    try:
        summarize_chapters_endpoint(args.endpoint_name, args.input_file, args.output_file)
    except Exception as error:
        logging.error(f"Error while invoking the endpoint: {str(error)}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger()

    main(sys.argv)
