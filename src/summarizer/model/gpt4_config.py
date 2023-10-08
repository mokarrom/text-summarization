"""GPT3 model config."""
max_tokens = 8000  # actual tokens = 8192
model_name = "gpt-4"
temperature = 0.7
sum_ratio = 0.45  # reserved enough tokens for 2000 words summary
"""Indicate x% summarization of the original text, i.e., 100 tokens text produce x tokens summary. Range [0-1]."""

