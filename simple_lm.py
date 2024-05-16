import random
from collections import defaultdict
import itertools

class WhitespaceTokenizer:
    def tokenize(self, text):
        return text.split()
    
    def untokenize(self, tokens):
        return ' '.join(tokens)

class SpaceTokenizer:
    def tokenize(self, text):
        return text.split(' ')
    
    def untokenize(self, tokens):
        return ' '.join(tokens)

class CharacterTokenizer:
    def tokenize(self, text):
        return list(text)
    
    def untokenize(self, tokens):
        return ''.join(tokens)

def get_ngrams(tokens, n):
    for i in range(len(tokens) - n + 1):
        context = tokens[i:i+n]
        yield context

def get_next_token_table(tokens, context_length, freq_table=None):
    if freq_table is None:
        freq_table = defaultdict(list)
    for *context, next_word in get_ngrams(tokens, context_length+1):
        freq_table[tuple(context)].append(next_word)
    return freq_table


def generate_tokens(freq_table, initial_context):
    context = tuple(initial_context)
    yield from context

    while True:
        candidates = freq_table[context]
        if not candidates:
            return
        next_word = random.choice(candidates)
        context = (*context[1:], next_word)     

        yield next_word

