import random
from collections import defaultdict

text = open("sample_data/blowin_in_the_wind_verses.txt")
text = text.read()

def tokenize(text):
    return text.split()

def untokenize(tokens):
    return ' '.join(tokens)

tokens = tokenize(text)

def get_ngrams(tokens, n):
    for i in range(len(tokens) - n + 1):
        context = tokens[i:i+n]
        yield context

def get_next_token_table(tokens, context_length):
    freq_table = defaultdict(list)
    for *context, next_word in get_ngrams(tokens, context_length+1):
        freq_table[tuple(context)].append(next_word)
    return freq_table

freq_table = get_next_token_table(tokens, 2)

def generate_tokens(freq_table, context_length, initial_context):
    output = list(initial_context)
    yield from output
    while True:
        context = output[-context_length:]
        candidates = freq_table[tuple(context)]
        if not candidates:
            return
        next_word = random.choice(candidates)
        output.append(next_word)

        yield next_word


output = []
for i, word in enumerate(generate_tokens(freq_table, 2, ["Yes,", "and"])):
    if i >= 100: break
    output.append(word)

print(untokenize(output))