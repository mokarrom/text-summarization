import logging
from typing import List
from summarizer.model.gpt3_summarizer import Summarizer, Gpt3Summarizer
from summarizer.dao.chunker import TextChunker
from summarizer.model import gpt3_config

logger = logging.getLogger(__name__)


class TextSummarizer:
    """Summarize text class."""

    def __init__(self, summarizer: Summarizer):
        self.summarizer = summarizer
        self.chunker = TextChunker(num_of_tokens=summarizer.text_tokens)

    def summarize_text(self, text: str, doc_id="n/a") -> str:
        chunk_sum: List[str] = []
        for chunk in self.chunker.chunk_generator_from_text(text, doc_id):
            logger.info(f"Summarizing chunk: {chunk.id}, of {chunk.document_id}, tokens: {chunk.num_of_tokens}")
            chunk_sum.append(self.summarizer.summarize(chunk.text()))

        return " ".join(chunk_sum)

    def summarize_file(self, txt_file: str) -> str:
        chunk_sum: List[str] = []
        for chunk in self.chunker.chunk_generator_from_file(txt_file):
            logger.info(f"Summarizing chunk: {chunk.id}, of {chunk.document_id}, tokens: {chunk.num_of_tokens}")
            chunk_sum.append(self.summarizer.summarize(chunk.text()))

        return " ".join(chunk_sum)


class SummarizerFactory:
    """"Summarizer Factory."""

    @staticmethod
    def create_summarizer(sum_type: str) -> Summarizer:
        if sum_type.lower() == "gpt3":
            summarizer = Gpt3Summarizer(
                model_name=gpt3_config.model_name,
                temperature=gpt3_config.temperature,
                text_tokens=gpt3_config.text_tokens,
                summary_tokens=gpt3_config.summary_token,
                top_p=gpt3_config.top_p,
                frequency_penalty=gpt3_config.frequency_penalty,
                presence_penalty=gpt3_config.presence_penalty
            )
        else:
            raise NotImplementedError(f"Summarizer type: {sum_type} is not implemented yet.")

        return summarizer
