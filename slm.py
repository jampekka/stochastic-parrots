import sys
from collections import defaultdict, Counter
import random
from itertools import islice

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
        self.follower_table = defaultdict(list)
    
    def __call__(self, context):
        context = tuple(context)
        candidates = self.follower_table[context]
        if not candidates:
            return None
        return random.choice(candidates)

    def train(self, tokens):
        for *context, next_word in get_ngrams(tokens, self.context_length + 1):
            context = tuple(context)
            self.follower_table[context].append(next_word)

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

