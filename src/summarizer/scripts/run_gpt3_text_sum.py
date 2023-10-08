import os
import logging
from summarizer.model.gpt3_summarizer import Gpt3Summarizer
from summarizer.util import SRC_RESOURCES_DIR

MODE_NAME = "text-davinci-003"
TEMPERATURE = 0.7
MAX_TOKENS = 600
TOKEN_RATIO = 0.9
TOP_P = 1.0
FREQUENCY_PENALTY = 0.0
PRESENCE_PENALTY = 1


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    text_tokens = int(TOKEN_RATIO * MAX_TOKENS)
    summary_tokens = MAX_TOKENS - text_tokens
    gpt3_text_sum = Gpt3Summarizer(
        model_name=MODE_NAME,
        temperature=TEMPERATURE,
        text_tokens=text_tokens,
        summary_tokens=summary_tokens,
        top_p=TOP_P,
        frequency_penalty=FREQUENCY_PENALTY,
        presence_penalty=PRESENCE_PENALTY
    )

    file_path = os.path.join(SRC_RESOURCES_DIR, "sample-text.txt")
    with open(file_path) as fp:
        text = fp.read()
        logging.info(gpt3_text_sum.summarize(text))

    print("hello")
