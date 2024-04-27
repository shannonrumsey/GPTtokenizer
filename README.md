# GPTtokenizer
A reconstruction of  the GPT-4 tokenizer (cl100k_base in OpenAI's tiktoken).
The substitution of this tokenizer from the "gpt2" one in the GPT-4 model improved its capabilities and preformance (along with new architecture, training data, etc).

This tokenizer breaks down strings and converts them into integers. These integers are looked-up in a table that gives the vector versions. The vectors are then fed into the transformer as an input.
