import re
import logging
from typing import Iterator, List, Tuple
from dataclasses import dataclass
from nltk.tokenize import sent_tokenize
from summarizer.model.gpt3_summarizer import count_tokens

PARA_REGEX = r"(?:\r?\n){2,}"
"""Greedily match two or more new-lines in Windows/Linux/Mac."""

logger = logging.getLogger(__name__)


@dataclass()
class Chunk:
    """A sequence of non-empty paragraphs separated by a newline."""

    id: int
    """Auto-increment number in a given text document."""
    num_of_tokens: int
    """The number of tokens in the current chunk."""
    document_id: str
    """Identifier of the document where this chunk belongs to. It could be file name or chapter name or document id."""
    paragraphs: Tuple[str]
    """A set of paragraphs of this chunk. We use tuple here to make paragraphs immutable, not for heterogeneous."""

    def text(self) -> str:
        """Return all the paragraph's text of this chunk."""
        return "\n".join(self.paragraphs)


def paragraphs(file_obj, separator="\n", joiner="".join) -> Iterator[str]:
    """Iterate a file object paragraph by paragraph. A paragraph is identified by an empty new line."""
    paragraph = []
    for line in file_obj:
        if line == separator:
            if paragraph:
                yield joiner(paragraph)
                paragraph = []
        else:
            paragraph.append(line)
    if paragraph:
        yield joiner(paragraph)


class TextChunker:
    """Process an input text file as a set of chunks."""

    def __init__(self, num_of_tokens: int):
        self.max_tokens = num_of_tokens

    def chunk_generator_from_file(self, file_name: str) -> Iterator[Chunk]:
        """Iterate a file object by chunk. A chunk consist of a set of paragraphs."""
        counter = 0
        chunk_tokens = 0
        chunk_paragraphs = []
        with open(file_name, "r", encoding="utf8") as file_obj:
            for paragraph in paragraphs(file_obj):
                para_tokens = count_tokens(paragraph)
                if para_tokens > self.max_tokens:
                    logger.error(f"Paragraph tokens: {para_tokens} must be less than the max_token: {self.max_tokens}")
                    raise RuntimeError(f"Too long paragraph, tokens: {para_tokens}, max_tokens: {self.max_tokens}")
                if chunk_tokens + para_tokens > self.max_tokens:
                    counter += 1
                    yield Chunk(counter, chunk_tokens, file_name, tuple(chunk_paragraphs))
                    chunk_paragraphs = [paragraph]
                    chunk_tokens = para_tokens
                else:
                    chunk_tokens += (para_tokens + 1 if chunk_paragraphs else para_tokens)
                    chunk_paragraphs.append(paragraph)
            if chunk_paragraphs:
                counter += 1
                yield Chunk(counter, chunk_tokens, file_name, tuple(chunk_paragraphs))

    def chunk_generator_from_text(self, text: str, doc_id: str) -> Iterator[Chunk]:
        """Iterate a text(str) by chunk. A chunk consist of a set of paragraphs."""
        def chunk_generator_from_paragraph(paragraph_text: str) -> Iterator[Chunk]:
            """Iterate a paragraph(str) by chunk. A chunk consist of a set of sentences/paragraphs."""
            nonlocal counter, chunk_tokens
            chunk_sentences = []
            for sentence in sent_tokenize(paragraph_text):
                sent_tokens = count_tokens(sentence)
                if sent_tokens > self.max_tokens:
                    logger.error(f"Skipping very long sentence, tokens: {sent_tokens}, max_tokens: {self.max_tokens}")
                    continue
                if chunk_tokens + sent_tokens > self.max_tokens:
                    counter += 1
                    chunk_paragraphs.append(" ".join(chunk_sentences))
                    yield Chunk(counter, chunk_tokens, doc_id, tuple(chunk_paragraphs))
                    chunk_paragraphs.clear()
                    chunk_sentences = [sentence]
                    chunk_tokens = sent_tokens
                else:
                    chunk_tokens += (sent_tokens + 1 if chunk_paragraphs and not chunk_sentences else sent_tokens)
                    chunk_sentences.append(sentence)
            if chunk_sentences:
                chunk_paragraphs.append(" ".join(chunk_sentences))

        counter = 0
        chunk_tokens = 0
        chunk_paragraphs: List[str] = []
        for paragraph in re.split(PARA_REGEX, text.strip()):
            para_tokens = count_tokens(paragraph)
            if para_tokens > self.max_tokens:
                logger.warning(f"Too long paragraph, tokens: {para_tokens}, max_tokens: {self.max_tokens}")
                logger.warning(f"Document-{doc_id}: Chunking long paragraph.")
                yield from chunk_generator_from_paragraph(paragraph)
            elif chunk_tokens + para_tokens > self.max_tokens:
                counter += 1
                yield Chunk(counter, chunk_tokens, doc_id, tuple(chunk_paragraphs))
                chunk_paragraphs = [paragraph]
                chunk_tokens = para_tokens
            else:
                chunk_tokens += (para_tokens + 1 if chunk_paragraphs else para_tokens)
                chunk_paragraphs.append(paragraph)
        if chunk_paragraphs:
            counter += 1
            yield Chunk(counter, chunk_tokens, doc_id, tuple(chunk_paragraphs))




