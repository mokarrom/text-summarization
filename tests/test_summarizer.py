import os.path
from pytest import raises
from summarizer.model.summarizer import SummarizerFactory, TextSummarizer
from summarizer.model.gpt3_summarizer import count_tokens

MOCKED_SUMMARY = ["mocked summary " + str(i + 1) for i in range(4)]


def test_text_summarizer(sample_file, mocker):
    gpt3_summarizer = SummarizerFactory.create_summarizer("gpt3")
    gpt3_summarizer.text_tokens = 100

    mock_summarizer = mocker.patch("summarizer.model.gpt3_summarizer.Gpt3Summarizer.summarize")
    mock_summarizer.side_effect = MOCKED_SUMMARY
    text_summarizer = TextSummarizer(gpt3_summarizer)

    with open(sample_file, "r") as file_obj:
        text = file_obj.read()
        summary = text_summarizer.summarize_text(text)
        assert summary == " ".join(MOCKED_SUMMARY)


def test_max_tokens(resource_path):
    file_path = os.path.join(resource_path, "chapter", "01.txt")
    gpt3_summarizer = SummarizerFactory.create_summarizer("gpt3")
    with open(file_path, "r") as file_obj:
        words = file_obj.read().split()

    index = 0
    largest_chunk = ""
    while len(largest_chunk) < 4 * gpt3_summarizer.text_tokens:
        largest_chunk = largest_chunk + " " + words[index]
        index += 1

    print(f"\nSummarizing the largest chunk of {len(largest_chunk)} length and {len(largest_chunk) // 4} tokens.")
    summary = gpt3_summarizer.summarize(largest_chunk)
    print(f"Summary received of {len(summary)} length and {len(summary) // 4} tokens")

    largest_chunk += "Additional tokens to check api threshold. Still it is fine, so add more tokens. We will be " \
                     "sending each chunk to the system and recording the summary of the chunk. Finally concatenating " \
                     "all chunks summary to get the chapter summary, return the chapter summary as system output. " \
                     "Since you can’t summarize an entire book all at once, you’ll have to divide the book into " \
                     "chunks. To make sure these chunks are as much as possible about one topic."

    print(f"\nSummarizing out of limit chunk of {len(largest_chunk)} length and {len(largest_chunk) // 4} tokens.")
    with raises(RuntimeError) as ex_info:
        gpt3_summarizer.summarize(largest_chunk)
    exception_raised = ex_info.value.args
    assert "OpenAI Error" == exception_raised[0]
    assert "This model's maximum context length is" in exception_raised[1]
    print(f"Error message: {exception_raised}")


def test_count_tokens():
    text1 = "The number of tokens processed in a given API request depends on the length of both your inputs and " \
           "outputs. As a rough rule of thumb, 1 token is approximately 4 characters or 0.75 words for English text."
    text2 = "One limitation to keep in mind is that your text prompt and generated completion combined must be no " \
            "more than the model's maximum context length"
    text3 = "Tokens can be words or just chunks of characters. For example, the word “hamburger” gets broken up into" \
            " the tokens “ham”, “bur” and “ger”, while a short and common word like “pear” is a single token. Many " \
            "tokens start with a whitespace, for example “ hello” and “ bye”."

    assert count_tokens(text1) == 43
    assert count_tokens(text2) == 26
    assert count_tokens(text3) == 86
