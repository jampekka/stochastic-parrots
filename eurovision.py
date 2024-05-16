import json
from pprint import pprint
from collections import defaultdict
from simple_lm import *
import re
import random
import gzip

context_length = 3
tokenizer = SpaceTokenizer()

data = json.load(gzip.GzipFile("data/eurovision-lyrics-2023.json.gz"))

freq_table = defaultdict(list)

songs = []
for _, d in data.items():
    if d['Language'] == "English":
        lyrics = d['Lyrics']
    else:
        lyrics = d['Lyrics translation']
    
    # Remove verse/chorus annotations
    lyrics = re.sub(r"\[.*\]", "", lyrics)
    lyrics = re.sub(r"\(.*\)", "", lyrics)
    # Trim the lines
    lyrics = "\n".join(s.strip() for s in lyrics.split('\n'))
    d['tokens'] = tokens = tokenizer.tokenize(lyrics)
    freq_table = get_next_token_table(tokens, context_length, freq_table)
    songs.append(d)

finnish_songs = [s for s in songs if s['Country'] == 'Finland']
start_song = random.choice(finnish_songs)


initial_context = start_song['tokens'][:context_length]
generator = generate_tokens(freq_table, initial_context)

for k in "Country Year Artist Song".split():
    print(f"{k}: {start_song[k]}")
print()

print(tokenizer.untokenize(generator))
