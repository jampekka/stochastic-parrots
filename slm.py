import sys
from collections import defaultdict, Counter
import random
from itertools import islice
from slm_torch import *
import numpy as np

class LanguageModel:
    """A base class for language models"""
    def __init__(self, tokenizer, embedder, predictor):
        self.tokenizer = tokenizer
        self.embedder = embedder
        self.predictor = predictor
    
    def train(self, tokens):
        embeddings = self.embedder(tokens)
        self.predictor.train(embeddings)
    
    def generate(self, context_tokens, max_tokens=None):
        if max_tokens is not None:
            return islice(self.generate(context_tokens), max_tokens)

        embeddings = self.embedder(context_tokens)
        return self.predictor.generate(embeddings)

class WhitespaceTokenizer:
    def __call__(self, text):
        return text.split()
    
    def decode(self, tokens):
        return ' '.join(tokens)

class CharacterTokenizer:
    def __call__(self, text):
        return list(text)
    
    def decode(self, tokens):
        return ''.join(tokens)

class SpaceTokenizer:
    def __call__(self, text):
        return text.split(' ')
    
    def decode(self, tokens):
        return ' '.join(tokens)

class NullEmbedder:
    def __call__(self, tokens): return tokens
    def decode(self, embeddings): return embeddings

def get_ngrams(tokens, n):
    # TODO: Empty tokens?
    for i in range(len(tokens) - n + 1):
        yield tokens[i:i+n]


class FrequencyTablePredictor(LanguageModel):
    def __init__(self, context_length):
        self.context_length = context_length
        self.follower_table = defaultdict(Counter)
    
    def __call__(self, context):
        context = tuple(context)
        candidates = list(self.follower_table[context].items())
        if not candidates:
            return None
        
        tokens, counts = zip(*candidates)
        weights = np.array(counts, dtype=float)
        weights /= np.sum(weights)
        return np.random.choice(tokens, p=weights)

    def train(self, tokens):
        for *context, next_word in get_ngrams(tokens, self.context_length + 1):
            context = tuple(context)
            self.follower_table[context][next_word] += 1

    def generate(self, context):
        # No inf range in python :(
        context = tuple(context)
        yield from context
        while True:
            next_token = self(context)
            if next_token is None:
                return
            yield next_token
            context = (*context[1:], next_token)

