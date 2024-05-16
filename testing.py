from simple_lm import *

input_file = "sample_data/blowin_in_the_wind_verses.txt"
context_length = 2
max_tokens = 100
tokenizer = WhitespaceTokenizer()
#tokenizer = SpaceTokenizer()

text = open(input_file).read()
tokens = tokenizer.tokenize(text)

initial_context = tokens[:context_length]

freq_table = get_next_token_table(tokens, context_length)

generator = generate_tokens(freq_table, initial_context)
output_tokens = itertools.islice(generator, max_tokens)
output_text = tokenizer.untokenize(output_tokens)

print(output_text)