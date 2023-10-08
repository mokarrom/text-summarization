"""GPT3 model config."""
max_tokens = 2040
model_name = "text-curie-001"
temperature = 0.7
top_p = 1.0
frequency_penalty = 0.0
presence_penalty = 1

token_ratio = 0.90
"""The ratio of number of tokens used in text and summary."""
text_tokens = int(token_ratio * max_tokens)
"""The maximum number of tokens used for text (i.e., prompt)"""
summary_token = max_tokens - text_tokens
"""The maximum number of tokens used for summary (i.e., completion)"""
