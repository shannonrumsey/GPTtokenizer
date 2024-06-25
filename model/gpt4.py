# this is a wrapper for the RegexTokenizer so that we can easily implement it into gpt.py as a tokenizer

import tiktoken
from .regex import RegexTokenizer

# merges byte sequences based on rank
def bpe(mergeable_ranks, token, max_rank):
    # converts token into list of individual byte sequences
    parts = [bytes([b]) for b in token]
    while True:
        min_idx = None
        min_rank = None
        for i, pair in enumerate(zip(parts[:-1], parts[1:])):
            # retrives the rank of the current pair
            rank = mergeable_ranks.get(pair[0] + pair[1])
            # if the current pair's rank is lower than the previous, update variables
            if rank is not None and (min_rank is None or rank < min_rank):
                min_idx = i
                min_rank = rank
        # terminates loop if no more merges
        if min_rank is None or (max_rank is not None and min_rank >= max_rank):
            break
            # ensures that a valid  pair was found for merging
        assert min_idx is not None
        parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2:]
    return parts

# reconstructs the original byte pair merge sequences from a given set of mergable ranks
def recover_subs(mergeable_ranks):
    subs = {}
    for token, rank in mergeable_ranks.items():
        # skips tokens that are raw bytes (single bytes) because they do not result from a merge
        if len(token) == 1:
            continue 
        # reconstructs original pair of bytes that were merged
        pair = tuple(bpe(mergeable_ranks, token, max_rank = rank))
        # ensures that the pair has exactly two elements
        assert len(pair) == 2
        ix0 = mergeable_ranks[pair[0]]
        ix1 = mergeable_ranks[pair[1]]
        subs[(ix0, ix1)] = rank
    return subs


split_pat = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
special_tokens = {
    '<|endoftext|>': 100257,
    '<|fim_prefix|>': 100258,
    '<|fim_middle|>': 100259,
    '<|fim_suffix|>': 100260,
    '<|endofprompt|>': 100276
}

# A wrapper on RegexTokenizer that matches GPT-4's tokenizer
class GPT4Tokenizer(RegexTokenizer):
    def __init__(self):
        # loads pretrained tokenizer for 'cl100k_base'
        super().__init__(pattern = split_pat)
        # get tiktoken tokenizer and its merges
        enc = tiktoken.get_encoding('cl100k_base')
        mergeable_ranks = enc._mergeable_ranks
        self.subs = recover_subs(mergeable_ranks)
        # reconstruct vocab from merges
        vocab = {replacement: bytes([replacement]) for replacement in range(256)}
        for (p0, p1), replacement in self.subs.items():
            vocab[replacement] = vocab[p0] + vocab[p1]
        self.vocab = vocab
        self.byte_shuffle = {i: mergeable_ranks[bytes([i])] for i in range(256)}
        self.inverse_byte_shuffle = {v: k for k, v in self.byte_shuffle.items()}
        self.register_special_tokens(special_tokens)

    def decode(self, tokens):
        # reconstructs byte sequence from vocab
        text_bytes = b''.join(self.vocab[replacement] for replacement in tokens)
        # applies the inverse byte shuffle to get the original bytes
        text_bytes = bytes(self.inverse_byte_shuffle[b] for b in text_bytes)
        text = text_bytes.decode('utf-8', errors = 'replace')
        return text

    def _encode_chunk(self, text_bytes):
        # apply the byte shuffle to input bytes
        text_bytes = bytes(self.byte_shuffle[b] for b in text_bytes)
        # use the _encode_chunk function from RegexTokenizer
        tokens = super()._encode_chunk(text_bytes)
        return tokens
    
    # model is pretrained and not intended to be trained
    def train(self, text, vocab_size, verbose = False):
        raise NotImplementedError('GPT4Tokenizer is pretrained and cannot be re-trained')

    def save(self, file_prefix):
        raise NotImplementedError('GPT4Tokenizer cannot be saved')

    def load(self, model_file):
        raise NotImplementedError('GPT4Tokenizer cannot be loaded')
