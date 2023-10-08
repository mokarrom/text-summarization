"""Summarization module."""
import os
import re
import time
import json
import logging
import threading
import multiprocessing
from typing import List, Dict
from multiprocessing.dummy import Pool as ThreadPool
from summarizer.model.gpt_summarizer import Summarizer, Gpt3Summarizer, Gpt4Summarizer, count_tokens_v2
from summarizer.dao.chunker import TextChunker, Chunk
from summarizer.util import SRC_RESOURCES_DIR
from summarizer.model import gpt3_config, gpt35_config, gpt4_config

logger = logging.getLogger(__name__)

SUMMARY_WORD_COUNT = 1800
"""Number of words expected in the final summary."""
BATCH_SIZE = 22
"""gpt-3.5-turbo has limit of 90_000 tokens per minute. Therefore, we're using this batch size to ensure 
that we're not sending more than 90_000 tokens within a minute."""
MIN_SUM_WORD = 15
"""Minimum number of words requested for a summary."""


class TextSummarizer:
    """Summarize text class."""
    summary_prompt, intro_prompt = None, None

    def __init__(self, primary_summarizer: Summarizer, secondary_summarizer: Summarizer = None):
        if TextSummarizer.summary_prompt is None or TextSummarizer.intro_prompt is None:
            TextSummarizer._load_summary_prompt()
        self.primary_summarizer = primary_summarizer
        self.secondary_summarizer = secondary_summarizer
        self.chunker = TextChunker(primary_summarizer.max_prompt_tokens - count_tokens_v2(self.summary_prompt))


    @classmethod
    def _load_summary_prompt(cls):
        with open(os.path.join(SRC_RESOURCES_DIR, "gpt-summarization-prompt.json")) as fp:
            prompt = json.load(fp)
            cls.summary_prompt = prompt["summary-prompt"]
            cls.intro_prompt = prompt["into-conclusion-prompt"]
            logger.info("GPT prompt has been initialized!")

    def summarize_text(self, text: str, doc_id="n/a") -> str:
        chunk_sum: List[str] = []
        for chunk in self.chunker.chunk_generator_from_text(text, doc_id):
            logger.info(f"Summarizing chunk: {chunk.id}, of {chunk.document_id}, tokens: {chunk.num_of_tokens}")
            chunk_sum.append(self.primary_summarizer.summarize(chunk.text()))

        return " ".join(chunk_sum)

    def summarize_file(self, txt_file: str) -> str:
        chunk_sum: List[str] = []
        for chunk in self.chunker.chunk_generator_from_file(txt_file):
            logger.info(f"Summarizing chunk: {chunk.id}, of {chunk.document_id}, tokens: {chunk.num_of_tokens}")
            chunk_sum.append(self.primary_summarizer.summarize(chunk.text()))

        return " ".join(chunk_sum)

    def _summarize_by_primary(self, chunk: Chunk) -> str:
        chunk_tok_count = count_tokens_v2(chunk.text())
        sum_tok_count = int(self.primary_summarizer.sum_ratio * chunk_tok_count)
        sum_word_count = int(0.68 * sum_tok_count)
        sum_prompt = self.summary_prompt.replace("XXX", str(sum_word_count)).replace("CHUNK-TEXT", chunk.text())
        logger.info(f"PS ({self.primary_summarizer.model_name}): Thread# {threading.get_ident()}: "
                    f"Summarizing chunk#{chunk.id} of {chunk_tok_count} tokens with expected summary of "
                    f"{sum_tok_count} tokens (~{sum_word_count} words)...")
        summary_text = self.primary_summarizer.summarize(sum_prompt)
        logger.info(f"PS ({self.primary_summarizer.model_name}): Thread# {threading.get_ident()}: "
                    f"Received summary for chunk#{chunk.id} of ~{len(summary_text.split())} words.")
        return summary_text

    def _summarize_by_secondary(self, chunk: Chunk) -> str:
        chunk_tok_count = count_tokens_v2(chunk.text())
        sum_tok_count = int(self.secondary_summarizer.sum_ratio * chunk_tok_count)
        sum_word_count = max(int(0.68 * sum_tok_count), MIN_SUM_WORD)
        sum_prompt = self.summary_prompt.replace("XXX", str(sum_word_count)).replace("CHUNK-TEXT", chunk.text())
        logger.info(f"SS ({self.secondary_summarizer.model_name}): Thread# {threading.get_ident()}: "
                    f"Summarizing chunk#{chunk.id} of {chunk_tok_count} tokens with expected summary of "
                    f"{sum_tok_count} tokens (~{sum_word_count} words)...")
        summary_text = self.secondary_summarizer.summarize(sum_prompt)
        logger.info(f"SS ({self.secondary_summarizer.model_name}): Thread# {threading.get_ident()}: "
                    f"Received summary for chunk#{chunk.id} of ~{len(summary_text.split())} words.")
        return summary_text

    def concurrent_summarize(self, chapters: List[str], book_id="n/a") -> str:
        """Summarize chapters concurrently using secondary summarizer."""
        full_book = "\n\n".join(chapters)
        total_tokens = count_tokens_v2(full_book + self.summary_prompt)
        if total_tokens > self.primary_summarizer.max_prompt_tokens:
            sum_ratio = max((self.primary_summarizer.max_prompt_tokens / total_tokens) - .05, 0.01)
            self.secondary_summarizer.update_sum_ratio(sum_ratio)
            logger.info(f"SS ({self.secondary_summarizer.model_name}): Full book tokens: {total_tokens}, "
                        f"max_prompt_tokens={self.primary_summarizer.max_prompt_tokens},"
                        f" sum ratio: {sum_ratio:.2f}")
            chunker_max_tokens = self.secondary_summarizer.max_prompt_tokens - count_tokens_v2(self.summary_prompt)
            chunker = TextChunker(num_of_tokens=chunker_max_tokens)
            logger.info(f"SS ({self.secondary_summarizer.model_name}): Initialize a chunker with {chunker_max_tokens} "
                        f"max tokens")
            chunks: List[Chunk] = [chunk for chunk in chunker.chunk_generator_from_chapters(chapters, book_id)]
            num_of_workers = min(4, len(chunks))
            logger.info(f"SS ({self.secondary_summarizer.model_name}): Number of workers : {num_of_workers}, "
                        f"total chunks: {len(chunks)}")
            summaries = []
            for i in range(0, len(chunks), BATCH_SIZE):
                start_time = time.perf_counter()
                with ThreadPool(processes=num_of_workers) as pool:
                    summaries.extend(pool.map(self._summarize_by_secondary, chunks[i:i+BATCH_SIZE]))
                sleep_time = max(int(60 - (time.perf_counter() - start_time)), 0)
                logging.warning(f"Sleep for {sleep_time} seconds to avoid {self.secondary_summarizer.model_name}"
                                f" model's rate limit")
                time.sleep(sleep_time)
            full_book = "\n\n".join(summaries)

        return full_book

    def summarize_chapters(self, chapters: List[str], book_id="n/a") -> Dict[str, str]:
        full_book: str = self.concurrent_summarize(chapters, book_id)
        prompt_tok_count = count_tokens_v2(full_book + self.summary_prompt)
        if prompt_tok_count > self.primary_summarizer.max_prompt_tokens:
            logger.warning(f"PS ({self.primary_summarizer.model_name}): Full book cannot fit! total tokens: "
                           f"{prompt_tok_count}, max_prompt_tokens={self.primary_summarizer.max_prompt_tokens},"
                           f" sum ratio: {self.primary_summarizer.sum_ratio:.2f}")
            chunks: List[Chunk] = [chunk for chunk in self.chunker.chunk_generator_from_text(full_book, book_id)]
            num_of_workers = min(multiprocessing.cpu_count() if multiprocessing.cpu_count() > 1 else 4, len(chunks))
            logger.info(f"PS ({self.primary_summarizer.model_name}): Number of workers : {num_of_workers}, "
                        f"total chunks: {len(chunks)}")
            with ThreadPool(processes=num_of_workers) as pool:
                summaries = pool.map(self._summarize_by_primary, chunks)
            summary = "\n\n".join(summaries)
        elif int(0.75 * prompt_tok_count) > SUMMARY_WORD_COUNT:
            sum_prompt = self.summary_prompt.replace("XXX", str(SUMMARY_WORD_COUNT)).replace("CHUNK-TEXT", full_book)
            logger.info(f"PS ({self.primary_summarizer.model_name}): Summarizing a text of "
                        f"{prompt_tok_count} tokens with expected summary of "
                        f"{self.primary_summarizer.max_completion_tokens} tokens (~{SUMMARY_WORD_COUNT} words)...")
            summary = self.primary_summarizer.summarize(sum_prompt)
            logger.info(f"PS ({self.primary_summarizer.model_name}): Received summary of ~{len(summary.split())} words.")
        else:
            summary = full_book
            logger.warning(f"Skipping primary summarizer for a text of {prompt_tok_count} tokens.")

        # Generate intro and conclusion using primary summarizer.
        intro_prompt = self.intro_prompt.replace("SUMMARY-TEXT", summary)
        logger.info(f"PS ({self.primary_summarizer.model_name}): Generating intro and conclusion from "
                    f"{len(summary.split())} words summary...")
        intro_conclusion = self.primary_summarizer.summarize(intro_prompt)
        # intro_conclusion = "Intro: This is intro.\n\nConclusion: This is conclusion."
        logger.info(f"PS ({self.primary_summarizer.model_name}): Received intro and conclusion of "
                    f"~{len(intro_conclusion.split())} words.")

        book_sum = dict()
        book_sum["intro"] = re.search("Intro: (.+?)\n\nConclusion:", intro_conclusion).group(1)
        book_sum["summary"] = summary
        book_sum["conclusion"] = intro_conclusion.split("\n\nConclusion: ")[-1]
        return book_sum


class SummarizerFactory:
    """"Summarizer Factory."""

    @staticmethod
    def create_summarizer(sum_type: str) -> Summarizer:
        if sum_type.lower() == "gpt3":
            summarizer = Gpt3Summarizer(
                model_name=gpt3_config.model_name,
                temperature=gpt3_config.temperature,
                max_tokens=gpt3_config.max_tokens,
                sum_ratio=gpt3_config.sum_ratio,
                top_p=gpt3_config.top_p,
                frequency_penalty=gpt3_config.frequency_penalty,
                presence_penalty=gpt3_config.presence_penalty
            )
        elif sum_type.lower() == "gpt3.5":
            summarizer = Gpt4Summarizer(
                model_name=gpt35_config.model_name,
                max_tokens=gpt35_config.max_tokens,
                sum_ratio=0.5,
                temperature=gpt35_config.temperature
            )
        elif sum_type.lower() == "gpt4":
            summarizer = Gpt4Summarizer(
                model_name=gpt4_config.model_name,
                max_tokens=gpt4_config.max_tokens,
                sum_ratio=gpt4_config.sum_ratio,
                temperature=gpt4_config.temperature
            )
        else:
            raise NotImplementedError(f"Summarizer type: {sum_type} is not implemented yet.")

        return summarizer
