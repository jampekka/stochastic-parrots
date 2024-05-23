import sys
from collections import defaultdict, Counter
import random
from itertools import islice
import numpy as np



class LanguageModel:
    def __init__(self, tokenizer, predictor):
        self.tokenizer = tokenizer
        self.predictor = predictor

    @property
    def context_length(self):
        return self.predictor.context_length

    def get_training_data(self, tokens):
        # TODO: Handle missing ends
        for *context, target in get_ngrams(tokens, self.predictor.context_length+1):
            yield context, target

    def train(self, tokens):
        data = self.get_training_data(tokens)
        self.predictor.train(data)

    def pad_context(self, context):
        l = len(context)
        if l > self.context_length:
            return context[-self.context_length:]
        if l < self.context_length:
            n_pad = self.context_length - l
            return (*[0]*n_pad, *context)
        return context
    
    def generate(self, context, max_tokens=9999999, include_initial=True, end_token=None, pad_initial=True):
        # Max tokens is a hack. No infinite range in Python stdlib I think :(
        if include_initial:
            yield from context
        
        # Very ugly!
        if pad_initial:
            context = self.pad_context(context)

        for i in range(max_tokens):
            if i >= max_tokens:
                return
            next_token = self.predictor(context)
            
            if next_token is None: return
            yield next_token
            if next_token == end_token: return

            context = (*context[1:], next_token)
            

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
    def __init__(self, context_length, *, bail_to_random=False):
        self.context_length = context_length
        self.follower_table = defaultdict(Counter)
        self.bail_to_random = bail_to_random
        self.sample_most_likely = False
    
    def __call__(self, context):
        context = tuple(context)

        candidates = self.follower_table.get(context, None)
        if not candidates:
            if self.bail_to_random:
                return random.choice(list(self.follower_table.keys()))[-1]
            else:
                return None

        tokens, counts = zip(*candidates.items())
        weights = np.array(counts, dtype=float)
        weights /= np.sum(weights)
        if not self.sample_most_likely:
            return np.random.choice(tokens, p=weights)
        else:
            return tokens[np.argmax(weights)]

    def train(self, xys):
        for context, target in xys:
            self.train_one(context, target)

    def train_one(self, context, target):
        context = tuple(context)
        self.follower_table[context][target] += 1


