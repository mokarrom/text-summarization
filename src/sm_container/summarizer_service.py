"""Chapter Summarizer."""
import os
import sys
import json
import threading
import traceback
import flask
import logging
from typing import Dict
from summarizer.model.summarizer import SummarizerFactory, TextSummarizer

prefix = "/opt/ml/"
model_path = os.path.join(prefix, "model")

TEXT_FIELD = "text"
SUMMARY_FIELD = "summary"

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
pid = os.getpid()


class SummarizerService(object):
    """A singleton class for initializing and holding summarizer."""

    _summarizer: TextSummarizer = None  # Where we keep the summarizer
    _lock = threading.Lock()

    @classmethod
    def get_summarizer(cls):
        """Initialized the summarizer object if it is not already initialized."""
        if cls._summarizer is None:
            with cls._lock:
                if not cls._summarizer:
                    cls._summarizer = TextSummarizer(SummarizerFactory.create_summarizer("gpt3"))
                    logger.info(f"Process id: {pid} - Initialized Summarizer!")
        return cls._summarizer

    @classmethod
    def summarize(cls, long_text: str, doc_id: str) -> str:
        """For the given long text, summarize it.

        Args:
            long_text (str): The long text that needs to be summarized.
            doc_id (str): Document id or text id or chapter id

        Returns:
            summary_text (str): summary of the long text.
            """
        summarizer = cls.get_summarizer()
        return summarizer.summarize_text(long_text, doc_id)


# The flask app for serving predictions
app = flask.Flask(__name__)
app.logger.info(f"Process id: {pid} - Importing Summarizer Service.")


@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy.

    In this sample container, we declare it healthy if we can initialize the model successfully.
    """
    health = SummarizerService.get_summarizer() is not None
    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')


@app.route('/invocations', methods=['POST'])
def summarization():
    """Run SummarizationService on the given text.

    Request type: POST
    Request body: A JSON object that contains 'text' field.
    Response:
        200 OK - Success.
        A JSON object that contains 'summary' filed.

        415 Unsupported Media Type - Fail.
        This endpoint only supports application/json data

        400 Bad Request - Fail.
        Failed to decode given CasePair JSON. Provide a valid JSON encoded CasePair.

        422 Unprocessable Entity - Fail.
        The given input is in the expected format but lacks enough attributes.

        500 Internal Server Error - Fail.
        Algorithm error - The algorithm fails to process the request, please check the error message/stack trace.
        Please note that the load balancer/web server also sends 500's errors. Search for "Algorithm error:" in the
        error message to get machine-learning algorithm error.
    """
    if flask.request.content_type == 'application/json':
        payload: str = flask.request.data.decode('utf-8')
    else:
        return flask.Response(
            response='Invalid request data type, only json is supported.',
            status=415,
            mimetype='text/plain'
        )

    try:
        data_dict: Dict = json.loads(payload)
        if TEXT_FIELD not in data_dict:
            app.logger.error(f"The request must contain '{TEXT_FIELD}' field!")
            return flask.Response(
                response=f"The request must contain '{TEXT_FIELD}' field", status=400, mimetype="text/plain"
            )
        else:
            text = data_dict[TEXT_FIELD]
            doc_id = data_dict.get("doc_id", "n/a")
            app.logger.info(f"Summarizing text length: {len(text)} of document: {doc_id}")
            summary = SummarizerService.summarize(text, doc_id)
            resp_json = json.dumps({SUMMARY_FIELD: summary})
            return flask.Response(response=resp_json, status=200, mimetype="application/json")
    except Exception as ex:
        err_msg = f"Algorithm error: {type(ex)}; message: {ex.args}; error: {traceback.format_exc()}"
        app.logger.error(err_msg)
        return flask.Response(response=err_msg, status=500, mimetype="text/plain")


def main():  # pragma: no cover
    """Start/bind the service."""
    app.run(threaded=True, host="0.0.0.0", port=8080)


if __name__ == "__main__":
    main()
