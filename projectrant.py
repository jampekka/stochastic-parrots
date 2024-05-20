import json
from pprint import pprint
from collections import defaultdict
from simple_lm import *
import re
import random
import gzip
import pandas as pd

context_length = 2
tokenizer = SpaceTokenizer()

table = pd.read_csv("outputgreaterthan.csv")
table.to_json("output.json", orient="index")
file = open("output.json", "r")
data = json.load(file)


# ------- Preprocessing here ----------

# -------------------------------------

freq_table = defaultdict(list)

rants = []
for _, d in data.items():
    # Trim the lines
    body = d['body']

    body = "\n".join(s.strip() for s in body.split('\n'))
    d['tokens'] = tokens = tokenizer.tokenize(body)
    freq_table = get_next_token_table(tokens, context_length, freq_table)
    rants.append(d)

bodies = [b for b in rants]
initial_context = ["Am","I", "stupid?"]
generator = generate_tokens(freq_table, initial_context)
text = tokenizer.untokenize(generator)
print(text)
"""
for i in range(20):
    print("--------- START RANT ---------\n")

    #start = random.choice(bodies)
    initial_context = start['tokens'][:context_length]
    print(initial_context)
    
    #generator = generate_tokens(freq_table, initial_context)
    #text = tokenizer.untokenize(generator)
    #print(text)

    print("\n--------- END ---------\n")
    """
"""

    for t in bodies:
        if text == t["body"]:
            print("Duplicate!")
    """

"""
for k in "Country Year Artist Song".split():
    print(f"{k}: {start[k]}")
print()
"""