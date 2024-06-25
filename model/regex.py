
import regex as re
from .base import Tokenizer, get_stats, sub

# GPT-4 text split pattern
split_pat = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
class RegexTokenizer(Tokenizer):

    def __init__(self, pattern = None):
        super().__init__()
        self.pattern = split_pat if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = {}
        self.inverse_special_tokens = {}

    def train(self, text, vocab_size, verbose = False):
        assert vocab_size >= 256
        num_subs = vocab_size - 256
        # split text into chunks
        chunks = re.findall(self.compiled_pattern, text)
        # input text preprocessing
        tokens = [list(chunk.encode('utf-8')) for chunk in chunks]
        # iteratively replace most common pairs
        subs = {}
        vocab = {replacement: bytes([replacement]) for replacement in range(256)}

        for i in range(num_subs):
            # count number of times consecutive pair occurs
            stats = {}
            for chunk_tokens in tokens:
                # this updates counts
                get_stats(chunk_tokens, stats)
            # find pair with highest count
            pair = max(stats, key = stats.get)
            replacement = 256 + i
            tokens = [sub(chunk_tokens, pair, replacement) for chunk_tokens in tokens]

            # save the merge
            subs[pair] = replacement
            # we add two bytes objects together
            vocab[replacement] = vocab[pair[0]] + vocab[pair[1]]

            if verbose:
                print(f'merge {i + 1} / {num_subs}: {pair} -> {replacement} ({vocab[replacement]}) had {stats[pair]} occurrences')

        # save class variables
        self.subs = subs # used in encode
        self.vocab = vocab # used in decode

    def register_special_tokens(self, special_tokens):
        # A dictionary that maps the special token to its associated token/bytes
        self.special_tokens = special_tokens
        # we switch the order of values and keys so that we can get the character association of the bytes
        # will look like {'<START>': 139964}
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}


    def decode(self, tokens):
        # given the ids, return text
        part_bytes = []
        for replacement in tokens:
            if replacement in self.vocab:
                part_bytes.append(self.vocab[replacement])
            elif replacement in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[replacement].encode('utf-8'))
            else:
                raise ValueError(f'invalid token id: {replacement}')

        text_bytes = b''.join(part_bytes)
        text = text_bytes.decode('utf-8', errors = 'replace')
        return text


# Encode Functions ------------------------------------------
    def encode_chunk(self, chunk_bytes):
        """
        Encodes each chunk
        - stats: counts of how often pairs occur in our sequence
        - pair: first detected repitition pattern that we would like to replace with 'replacement'
        """
        # gives us the raw bytes
        tokens = list(chunk_bytes)
        while len(tokens) >= 2:
            # find pair that was replaced first
            stats = get_stats(tokens)
            pair = min(stats, key = lambda p: self.subs.get(p, float('inf')))
            # if nothing else to substitute
            if pair not in self.subs:
                break
            replacement = self.subs[pair]
            tokens = sub(tokens, pair, replacement)
        return tokens

    def encode_ordinary(self, text):
        """
        Encode process if there are no special tokens
        """
        # splits text into chunks based on regex pattern
        text_chunks = re.findall(self.compiled_pattern, text)
        # chunks are encoded spearately, then joined together
        tokens = []
        for chunk in text_chunks:
            # converts to raw bytes
            chunk_bytes = chunk.encode('utf-8')
            chunk_ids = self.encode_chunk(chunk_bytes)
            tokens.extend(chunk_ids)
        return tokens
        
    
    def encode(self, text, allowed_special = 'none_raise'):
        """
        Evaluates whether there are special tokens and encodes appropriately
        """
        # allowed_special can take all, none, none_raise, or a specific set of allowed specials
        # none_raise will raise an error if any special token is encountered in text
        special = None
        if allowed_special == 'all':
            special = self.special_tokens
        elif allowed_special == 'none':
            special = {}
        elif allowed_special == 'none_raise':
            special = {}
            # checks to see if any special tokens are found in text, if so, raises AssertionError
            assert all(token not in text for token in self.special_tokens)
        # if allowed_special is a set of custom special tokens, then it will change special to only include those
        elif isinstance(allowed_special, set):
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        # input to allowed_special (misspelling, etc.) raises ValueError
        else:
            raise ValueError(f'allowed_special = {allowed_special} not understood')
        # no special tokens, use regular encoding
        if not special:
            return self.encode_ordinary(text)
        # takes all special tokens and creates a regex pattern
        special_pattern = '(' + '|'.join(re.escape(k) for k in special) + ')'
        # splits text based on occurrence of special tokens
        special_chunks = re.split(special_pattern, text)
        # encodes each chunk separately, then joins the results
        tokens = []
        for part in special_chunks:
            if part in special:
                tokens.append(special[part])
            else:
                tokens.extend(self.encode_ordinary(part))
        return tokens