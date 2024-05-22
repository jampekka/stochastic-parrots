import sys
from collections import defaultdict, Counter
import random
from itertools import islice
from slm_torch import *
import numpy as np

# TODO: Put the embedder into the predictor
class LanguageModel:
    def __init__(self, tokenizer, embedder, predictor):
        self.tokenizer = tokenizer
        self.embedder = embedder
        self.predictor = predictor

    @property
    def context_length(self):
        return self.predictor.context_length

    def get_training_data(self, tokens):
        # TODO: Handle missing ends
        for *context, target in get_ngrams(tokens, self.predictor.context_length+1):
            yield self.embedder(context), target

    def train(self, tokens):
        data = self.get_training_data(tokens)
        self.predictor.train(data)
    
    def generate(self, context_tokens, max_tokens=9999999, include_initial=True, end_token=None):
        # Max tokens is a hack. No infinite range in Python stdlib I think :(
        if include_initial:
            yield from context_tokens

        for i in range(max_tokens):
            if i >= max_tokens:
                return
            context_embedding = self.embedder(context_tokens)
            next_token = self.predictor(context_embedding)
            if next_token is end_token:
                return
            yield next_token
            context_tokens = (*context_tokens[1:], next_token)
            context_embedding = self.embedder(context_tokens)
            


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

class FrequencyTablePredictor:
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

    def train(self, xys):
        for context, target in xys:
            context = tuple(context)
            self.follower_table[context][target] += 1
    
    def generate(self, context):
        context = tuple(context)
        while True:
            next_token = self(context)
            if next_token is None:
                return
            yield next_token
            context = (*context[1:], next_token)


