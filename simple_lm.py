import random
from collections import defaultdict

def tokenize(text):
    return text.split()

def untokenize(tokens):
    return ' '.join(tokens)


def get_ngrams(tokens, n):
    for i in range(len(tokens) - n + 1):
        context = tokens[i:i+n]
        yield context

def get_next_token_table(tokens, context_length):
    freq_table = defaultdict(list)
    for *context, next_word in get_ngrams(tokens, context_length+1):
        freq_table[tuple(context)].append(next_word)
    return freq_table


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



def main():
    input_file = "sample_data/blowin_in_the_wind_verses.txt"
    context_length = 2
    max_tokens = 100

    text = open(input_file).read()
    tokens = tokenize(text)
    freq_table = get_next_token_table(tokens, context_length)
    
    output = []
    for i, word in enumerate(generate_tokens(freq_table, context_length, ["Yes,", "and"])):
        if i >= 100: break
        output.append(word)

    print(untokenize(output))

if __name__ == "__main__":
    main()