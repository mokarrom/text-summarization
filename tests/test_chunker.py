import os
from nltk.tokenize import sent_tokenize

from summarizer.dao.chunker import paragraphs, TextChunker
from summarizer.model.gpt3_summarizer import count_tokens


def test_paragraphs(sample_file):
    para_words = []
    paras = []
    with open(sample_file, "r") as file_obj:
        for para in paragraphs(file_obj):
            paras.append(para)
            para_words.extend(para.split())
    para_text = " ".join(para_words)

    with open(sample_file) as file_obj:
        words = file_obj.read().split()
        text = " ".join(words)

    assert len(paras) == 11
    assert para_text == text


def test_chunk_generator_from_file(sample_file):
    chunker = TextChunker(num_of_tokens=100)
    chunks = [chunk for chunk in chunker.chunk_generator_from_file(sample_file)]
    assert len(chunks) == 5

    chunk_words = []
    for chunk in chunks:
        chunk_words.extend(chunk.text().split())
        assert chunk.num_of_tokens <= chunker.max_tokens
        assert chunk.document_id == sample_file
        assert chunk.num_of_tokens == count_tokens(chunk.text())
    chunk_text = " ".join(chunk_words)

    with open(sample_file, "r", encoding="utf8") as file_obj:
        words = file_obj.read().split()
        text = " ".join(words)
    print(f"Number of words in full text: {len(words)}")
    print(f"Number of tokens in full text: {count_tokens(text)}")

    assert chunk_text == text
    assert len(chunk_words) == len(words)


def test_chunk_generator_from_text(resource_path):
    chunker_test_file = os.path.join(resource_path, "chunker-test-2.txt")
    chunker = TextChunker(num_of_tokens=100)
    with open(chunker_test_file, "r") as file_obj:
        raw_text = file_obj.read()
    chunks = [chunk for chunk in chunker.chunk_generator_from_text(raw_text, "n/a")]
    assert len(chunks) == 8

    chunk_words = []
    last_word_pos = []
    for chunk in chunks:
        chunk_words.extend(chunk.text().split())
        last_word_pos.append(raw_text.find(chunk_words[-1]))
        assert chunk.num_of_tokens <= chunker.max_tokens
        assert chunk.document_id == "n/a"
        assert chunk.num_of_tokens == count_tokens(chunk.text())
    chunk_text = " ".join(chunk_words)
    assert last_word_pos == sorted(last_word_pos)

    words = raw_text.split()
    clean_text = " ".join(words)
    print(f"\nNumber of words in full text: {len(words)}")
    print(f"Number of tokens in raw text: {count_tokens(raw_text)}")
    print(f"Number of tokens in clean text: {count_tokens(clean_text)}")

    assert chunk_text == clean_text
    assert len(chunk_words) == len(words)


def test_large_paragraph_chunking():
    para_text = "While running on the entire test set, I notice, rarely, there is a paragraph which is greater than " \
                "our max_tokens. Currently, for those excessively long paragraphs, we through an exception " \
                "(RuntimeError: Too long paragraph, tokens: 2378, max_tokens: 1836). Please let me know if that's " \
                "okay or do you wanna process them by splitting into multiple chunks? I'd rather have a solution for" \
                " this in place than throw an error. When a paragraph is longer than max tokens (which indeed seems " \
                "very unlikely to happen often) let's just split such paragraph into separate chunks where we use a " \
                "sentence to break up the chunks (dots as opposed to line-breaks). In the ideal scenario we " \
                "shouldn't have errors so let's try to prevent them. So we have decided to split long paragraph into" \
                " multiple chunks. In this special case a chunk will consist of a set of sentences where a sentence " \
                "is define by period. So a sentence can't be part of two chunks, either we will include the sentence" \
                " or save it for the next chunk. We will not mixed up long paragraph with small paragraph."

    chunker = TextChunker(num_of_tokens=100)
    chunks = [chunk for chunk in chunker.chunk_generator_from_text(para_text, "n/a")]

    chunk_words = []
    for chunk in chunks:
        chunk_words.extend(chunk.text().split())
        assert chunk.num_of_tokens <= chunker.max_tokens
        assert chunk.document_id == "n/a"
        assert chunk.num_of_tokens == count_tokens(chunk.text())
    chunk_text = " ".join(chunk_words)
    assert chunk_text == para_text


def test_sentence_splitter():
    sample_1 = "Mr. John Johnson Jr. was born in the U.S.A but earned his Ph.D. in Israel before joining Nike Inc. as" \
               " an engineer. He also worked at craigslist.org as a business analyst."
    sample_2 = "Good morning Dr. Adams. The patient is waiting for you in room number 3."
    sample_3 = "The first time you see The Second Renaissance it may look boring. Look at it at least twice and " \
               "definitely watch part 2. It will change your view of the matrix. Are the human people the ones " \
               "who started the war? Is AI a bad thing?"
    sample_4 = "Recognizing the rising opportunity Jerusalem Venture Partners opened up their Cyber Labs incubator, " \
               "giving a home to many of the city’s promising young companies. International corporates like EMC " \
               "have also established major centers in the park, leading the way for others to follow! On a visit " \
               "last June, the park had already grown to two buildings with the ground being broken for the " \
               "construction of more in the near future. This is really interesting! What do you think?"
    sample_5 = "Faded C. Louting C. Appellant C." \
               "Mouldy C. Discouraged C. Swagging C." \
               "Musty C. Surfeited C. Withered C." \
               "Paltry C. Peevish C. Broken-reined C." \
               "Senseless C. Translated C. Defective C."

    assert len(sent_tokenize(sample_1)) == 2
    assert len(sent_tokenize(sample_2)) == 2
    assert len(sent_tokenize(sample_3)) == 5
    assert len(sent_tokenize(sample_4)) == 5
    assert len(sent_tokenize(sample_5)) == 1
