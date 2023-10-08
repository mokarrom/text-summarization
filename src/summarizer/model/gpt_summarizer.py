"""Interface with GPT-3 API."""
import os
import openai
import tiktoken
import logging
import dotenv
from abc import ABC, abstractmethod
from openai import OpenAIError
from transformers import GPT2TokenizerFast
from summarizer.util import SRC_RESOURCES_DIR
from summarizer.aws_secrert_manager import get_openai_api_key_from_sm

TLDR_TAG = "\n\nTl;dr"
GPT2_TOKENIZER = GPT2TokenizerFast.from_pretrained("gpt2")
ENCODING = tiktoken.get_encoding("cl100k_base")  # cl100k_base = gpt-4, gpt-3.5-turbo, text-embedding-ada-002

logger = logging.getLogger(__name__)


def _get_openai_api_key() -> str:
    api_key = dotenv.dotenv_values(os.path.join(SRC_RESOURCES_DIR, ".env"))["OPENAI_API_KEY"]
    if api_key == "None":
        api_key = get_openai_api_key_from_sm()
    else:
        logger.info("OpenAI API key retrieved from dotenv.")
    return api_key


class Summarizer(ABC):
    @abstractmethod
    def __init__(self, model_name: str, max_tokens: int, sum_ratio: float):
        self.model_name = model_name
        self.max_tokens: int = max_tokens
        self.sum_ratio: float = sum_ratio
        if sum_ratio:
            self.update_sum_ratio(sum_ratio)

    def summarize(self, input_text, **kwargs) -> str:
        """Summarize the given text.
        Args:
            input_text: A large text.
        Returns:
            summary_text: The summary of the large text.
        """
        pass

    def update_sum_ratio(self, sum_ratio: float):
        """Update token ratio and the corresponding maximum tokens."""
        self.sum_ratio: float = sum_ratio
        self.max_prompt_tokens = int(self.max_tokens / (1 + sum_ratio))
        """The maximum number of tokens used for text (i.e., prompt)"""
        self.max_completion_tokens = self.max_tokens - self.max_prompt_tokens
        """The maximum number of tokens used for summary (i.e., completion)"""
        logger.info(f"{self.model_name}: set token ratio: {sum_ratio:.2f}, max_tokens: {self.max_tokens}, "
                    f"max_prompt_token: {self.max_prompt_tokens}, max_completion_token: {self.max_completion_tokens}")


class Gpt3Summarizer(Summarizer):
    def __init__(
            self,
            model_name: str,
            temperature: float,
            max_tokens: int,
            sum_ratio: float,
            top_p: float,
            frequency_penalty: float,
            presence_penalty: float
    ):
        super().__init__(model_name, max_tokens, sum_ratio)
        openai.api_key = _get_openai_api_key()
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        logger.info(f"OpenAI initialized with: model_name={model_name}, temperature={temperature}, "
                    f"max_tokens={max_tokens}, max_text_tokens={self.max_prompt_tokens}, "
                    f"summary_tokens={self.max_completion_tokens}, "
                    f"top_p={top_p} frequency_penalty={frequency_penalty}, presence_penalty={presence_penalty}")

    def summarize(self, input_text, **kwargs) -> str:
        """Return summary of the given input_text."""
        logger.debug(f"############### Input Text ###############\n{input_text}\n.................................\n")
        try:
            response = openai.Completion.create(
                model=self.model_name,
                prompt=input_text + TLDR_TAG,
                temperature=self.temperature,
                max_tokens=self.max_completion_tokens,
                top_p=self.top_p,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty
            )
            summary_text: str = response["choices"][0]["text"]
        except OpenAIError as ex:
            logger.error(f"OpenAI Error: {ex.user_message}")
            raise RuntimeError("OpenAI Error", ex.user_message)
        logger.debug(f"############### Summary Text ###############\n{summary_text}\n..............................\n")
        return summary_text


class Gpt4Summarizer(Summarizer):
    """GPT-3.5 and 4 API"""

    def __init__(self, model_name: str, temperature: float, max_tokens: int, sum_ratio: float):
        super().__init__(model_name, max_tokens, sum_ratio)
        openai.api_key = _get_openai_api_key()
        self.temperature = temperature
        logger.info(f"OpenAI initialized with: model_name={model_name}, temperature={temperature}, "
                    f"max_tokens={max_tokens}, token_ratio={sum_ratio}.")

    def summarize(self, input_text, **kwargs) -> str:
        """Return summary of the given input_text."""
        logger.debug(f"############### Input Text ###############\n{input_text}\n.................................\n")
        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": input_text}
                ],
                # max_tokens=self.max_summary_tokens,
                stop=None,
                temperature=self.temperature,
            )
            summary_text: str = response["choices"][0]["message"]["content"]
        except OpenAIError as ex:
            logger.error(f"OpenAI Error: {ex.user_message}")
            raise RuntimeError("OpenAI Error", ex.user_message)
        logger.debug(f"############### Summary Text ###############\n{summary_text}\n..............................\n")
        return summary_text


def count_tokens(text: str):
    """Return the number of tokens in the given text."""
    return len(GPT2_TOKENIZER(text)["input_ids"])


def count_tokens_v2(text: str):
    """Return the number of tokens in the given text."""
    return len(ENCODING.encode(text))
