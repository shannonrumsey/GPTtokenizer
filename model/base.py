#Defines helper functions and a (Parent) base class for our child class in basic.py
import unicodedata
"""
get_stats: grabs all consecutive pairs and counts the occurrences
sub: iteratively replace highest ranking pairs
- tokens: individual byte encoding for each character
- pair: pair we want to replace
- replacement: the token id of what we replace the pair with
"""

def get_stats(tokens, counts = None):
    counts = {} if counts is None else counts
    for pair in zip(tokens, tokens[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def sub(tokens, pair, replacement):
    newids = []
    i = 0
    while i < len(tokens):
        # checks if the encodings at the current position matches with the pair
        if tokens[i] == pair[0] and i < len(tokens) - 1 and tokens[i + 1] == pair[1]:
            newids.append(replacement)
            # skips over the pair
            i += 2
        else:
            newids.append(tokens[i])
            i += 1
    return newids

def replace_control_characters(s: str) -> str:
    # removes control characters
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != 'C':
            chars.append(ch)
        else:
            chars.append(f'\\u{ord(ch):04x}') # escape
    return ''.join(chars)


def render_token(t: bytes) -> str:
    # pretty print tokens
    s = t.decode('utf-8', errors = 'replace') # 'replace' to replace invalid strings with replacement char
    s = replace_control_characters(s)
    return s

# Tokenizer model -------------------------------------
# This will be our base class
class Tokenizer:

    def __init__(self):
        """
        - set defaults to vocab size of 256, no substitutions, patterns, or special tokens
        - subs: keeps track of all pairs and replacements resulting from BPE
        - pattern: regex pattern to split string into tokens by (see regex.py)
        - vocab: all possible tokens; derived from subs
        """
        self.subs = {}
        self.pattern = ""
        self.special_tokens = {} # these are typically start and stop symbols
        self.vocab = self._build_vocab()


    def train(self, text, vocab_size, verbose = False):
        # BasicTokenizer class will not run if train is not utilized
        raise NotImplementedError

    def decode(self, tokens):
        # BasicTokenizer class will not run if decode is not utilized
        raise NotImplementedError

    def encode(self, text):
        # BasicTokenizer class will not run if encode is not utilized
        raise NotImplementedError

    def _build_vocab(self):
        """
        - vocab: a mapping/dictionary between the token id and the bytes object of the token
        - token1 and token2 are the bytes being replaced by 'replacement' (i.e. (131, 92) = 257)
        """
        vocab = {replacement: bytes([replacement]) for replacement in range(256)}
        # we iterate in order of the substitutions
        for (p0, p1), replacement in self.subs.items():
            # we add two bytes objects together
            vocab[replacement] = vocab[p0] + vocab[p1]
        # if we have special tokens
        for special, replacement in self.special_tokens.items():
            vocab[replacement] = special.encode('utf-8')
        return vocab

    def save(self, file_prefix):
        # Saves .model to be used with load()
        model_file = file_prefix + '.model'
        with open(model_file, 'w') as f:
            f.write("minbpe v1\n")
            f.write(f'{self.pattern}\n')
            f.write(f"{len(self.special_tokens)}\n")
            for special, replacement in self.special_tokens.items():
                f.write(f"{special} {replacement}\n")
            # writes the pairs that get replaced
            for p0, p1 in self.subs:
                f.write(f'{p0} {p1}\n')
        
        # saves the vocab file (for human use)
        vocab_file = file_prefix + '.vocab'
        inverted_subs = {replacement: pair for pair, replacement in self.subs.items()}
        with open(vocab_file, 'w', encoding = 'utf-8') as f:
            for replacement, token in self.vocab.items():
                s = render_token(token)
                # find children of token if any
                if replacement in inverted_subs:
                    # it token has children, render nicely
                    replace0, replace1 = inverted_subs[replacement]
                    s0 = render_token(self.vocab[replace0])
                    s1 = render_token(self.vocab[replace1])
                    f.write(f'[{s0}][{s1}] -> [{s}] {replacement}\n')
                else:
                    f.write(f'[{s}] {replacement}\n')


    def load(self, model_file):
        assert model_file.endswith('.model')
        # read model file
        subs = {}
        special_tokens = {}
        replacement = 256
        with open(model_file, 'r', encoding = 'utf-8') as f:
            # read version
            version = f.readline().strip()
            # read regex pattern
            self.pattern = f.readline().strip()
            # read special tokens
            num_special = int(f.readline().strip())

            # read the special tokens and their replacements
            for _ in range(num_special):
                special, special_replace = f.readline().strip()
                special_tokens[special] = int(special_replace)

            # read the substituitions/replacements
            for line in f:
                token1, token2 = map(int, line.split())
                subs[(token1, token2)] = replacement
                replacement += 1
                
        self.subs = subs
        self.special_tokens = special_tokens
        self.vocab = self._build_vocab()
