"""A python script to hit a SageMaker endpoint."""
import os
import sys
import uuid
import time
import json
import boto3
import urllib
import logging
import threading
import sagemaker
import argparse as ap
from typing import Tuple
from botocore.exceptions import ClientError
from summarizer.util import SRC_RESOURCES_DIR

AWS_REGION = "eu-central-1"
ENDPOINT_NAME = "gpt-book-sum-endpoint"
CONTENT_TYPE = "application/json"
INVOCATION_TIMEOUT = 3600
BOOKSUM_DIR = os.path.join(SRC_RESOURCES_DIR, "booksum")
CHAPTER_DIR = os.path.join(SRC_RESOURCES_DIR, "chapter")
SM_RUNTIME = boto3.client("sagemaker-runtime", region_name=AWS_REGION)
SM_SESSION = sagemaker.session.Session()
S3_CLIENT = boto3.client("s3")
BUCKET_NAME = "chapter-summarization"
BUCKET_PREFIX = "book-sum-async-inference/input/"
SLEEP_TIME = 5


def get_output_from_s3(output_s3uri: str, failure_s3uri: str) -> str:
    """Retrieve output from S3 by retrieving the file."""
    def _get_output_worker(s3uri: str, ch: str):
        s3url = urllib.parse.urlparse(s3uri)
        bucket = s3url.netloc
        key = s3url.path[1:]
        nonlocal output

        while output is None:
            try:
                with lock:
                    output = SM_SESSION.read_s3_file(bucket=bucket, key_prefix=key)
            except ClientError as e:
                if e.response["Error"]["Code"] == "NoSuchKey":
                    print(ch, end="")
                    time.sleep(SLEEP_TIME)
                    continue
                raise

    output = None
    lock = threading.Lock()
    t1 = threading.Thread(target=_get_output_worker, args=(output_s3uri, "*"))
    t2 = threading.Thread(target=_get_output_worker, args=(failure_s3uri, "#"))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    print()
    return output


def invoke_sm_endpoint_async(endpoint_name: str, input_location: str, inference_id: str) -> Tuple[str, str]:
    """Invoke the endpoint for the given data and return the response."""
    response = SM_RUNTIME.invoke_endpoint_async(
        EndpointName=endpoint_name,
        InputLocation=input_location,
        InferenceId=inference_id,
        ContentType=CONTENT_TYPE,
        Accept=CONTENT_TYPE,
        InvocationTimeoutSeconds=INVOCATION_TIMEOUT
    )
    return response["OutputLocation"], response["FailureLocation"]


def summarize_book(endpoint_name: str, input_file: str, output_file: str):
    """Summarize the given file and save summary into provided file."""
    inference_id = str(uuid.uuid4())
    object_name = BUCKET_PREFIX + inference_id + ".json"
    try:
        S3_CLIENT.upload_file(input_file, BUCKET_NAME, object_name)
    except ClientError as e:
        logger.error(e)
    input_location = f"s3://{BUCKET_NAME}/{object_name}"
    logger.info(f"Input location: {input_location}")

    output_location, failure_location = invoke_sm_endpoint_async(endpoint_name, input_location, inference_id)
    logger.info(f"Output location: {output_location}")
    logger.info(f"Failure location: {failure_location}")
    output = get_output_from_s3(output_location, failure_location)
    summary = json.loads(output)
    logger.info(f"Received summary: {json.dumps(summary, indent=4, sort_keys=False)}")

    with open(output_file, encoding="utf8", mode="w") as fp:
        json.dump(summary, fp)


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
        default=os.path.join(CHAPTER_DIR, "2252-chapters.json"),
        help="A JSON Lines input file where each line is a JSON object."
    )
    parser.add_argument(
        "--output_file", type=str, metavar="input-file",
        default=os.path.join(CHAPTER_DIR, "2252-chapters-sum.json"),
        help="A JSON Lines output file where each line is a JSON object."
    )

    return parser.parse_args(args[1:])


def main(args):
    """The main function."""
    args = _get_args(args)
    logger.info(f"Provided args: {args}")

    start_time = time.perf_counter()
    try:
        summarize_book(args.endpoint_name, args.input_file, args.output_file)
    except Exception as error:
        logger.error(f"Error while invoking the endpoint: {str(error)}")
    finally:
        logger.info(f"Execution time: {(time.perf_counter() - start_time) / 60:.2f} mins")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger()

    main(sys.argv)
