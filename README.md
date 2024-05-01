# GPTtokenizer
A reconstruction of  the GPT-4 tokenizer (cl100k_base in OpenAI's tiktoken).
The substitution of this tokenizer from the "gpt2" one in the GPT-4 model improved its capabilities and preformance (along with new architecture, training data, etc).

This tokenizer breaks down strings and converts them into integers. These integers are looked-up in a table that gives the vector versions. The vectors are then fed into the transformer as an input.

This tackles the issue that occurs when using UTF-8 encoding which converts each character of each word into a byte stream that is between 1 and 4 bytes. This inflates our encodings and can be difficult to create/train a model on because we can only have a finite length of context that we can support in the transformer (increasing context length can be incredibly inefficient and computationally expensive). Instead, we use a byte pair encoding algorithm which utilizes the utf-8 encodings but condenses our context lengths to be better supported by the model.

# Byte Pair Encoding
An algorithm that iteratively replaces the most common sequences of consecutive characters with an unused character. This condenses the original string and sometimes the the vocab size.

Example: 

\left\{Step 1: 'zeggghggghc', length = 11, vocab size = 5

Step 2: Replacing gg with A; 'zeAghAghc', length = 9, vocab size = 6

Step 3: Repeat the process by replacing gh with B; 'zeABABc', length = 7, vocab size = 5

Step 4: Repeat by replacing AB with X; 'zeXXc', length = 5, vocab size = 4 \right\}

# References
MEGABYTE: Predicting Million-byte Sequences with Multiscale Transformers
