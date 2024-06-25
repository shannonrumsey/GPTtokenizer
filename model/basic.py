# Takes functions and Tokenizer class from .base to encode, train, and decode the model using BPE

from .base import Tokenizer, get_stats, sub

class BasicTokenizer(Tokenizer):

    def __init__(self):
        # Inherits behavior of parent class, in this case, "Tokenizer"
        super().__init__()

    def train(self, text, vocab_size, verbose = False):
        """
        - vocab size: the final length of the vocabulary size we want are tokenizer to have
        """
        assert vocab_size >= 256
        num_subs = vocab_size - 256
        # creates a copy of the list
        tokens = list(text.encode('utf-8'))
        subs = {}
        vocab = {replacement: bytes([replacement]) for replacement in range(256)}

        for i in range(num_subs):
            stats = get_stats(tokens)
            pair = max(stats, key = stats.get)
            replacement = 256 + i
            tokens = sub(tokens, pair, replacement)
            subs[pair] = replacement
            # we add two bytes objects together
            vocab[replacement] = vocab[pair[0]] + vocab[pair[1]]

            if verbose:
                print(f'merge {i + 1} / {num_subs}: {pair} -> {replacement} ({vocab[replacement]}) had {stats[pair]} occurrences')

        self.subs = subs
        self.vocab = vocab
    

    def decode(self, tokens):
        # iterate over ids to get byte objects
        text_bytes = b''.join(self.vocab[replacement] for replacement in tokens)
        # not all byte sequences are valid utf-8 (i.e. 128), so we need to include errors = 'replace'
        text = text_bytes.decode('utf-8', errors = 'replace')
        return text

    def encode(self, text):
        """
        - stats: counts of how often pairs occur in our sequence
        - pair: first detected repitition pattern that we would like to replace with 'replacement'
        """
        # gives us the raw bytes
        tokens = list(text.encode('utf-8'))
        while len(tokens) >= 2:
            stats = get_stats(tokens)
            pair = min(stats, key = lambda p: self.subs.get(p, float('inf')))
            # if nothing else to substitute
            if pair not in self.subs:
                break
            replacement = self.subs[pair]
            tokens = sub(tokens, pair, replacement)
        return tokens
