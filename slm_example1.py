from slm import *

text = open("sample_data/blowin_in_the_wind_verses.txt").read()

context_length = 2
model = LanguageModel(
        #tokenizer=CharacterTokenizer(),
        tokenizer=WhitespaceTokenizer(),


        embedder=NullEmbedder(),
        predictor=FrequencyTablePredictor(context_length)
        )

tokens = model.tokenizer(text)
model.train(tokens)

initial_context = tokens[:context_length]
generated_tokens = model.generate(initial_context, 100)
generated_text = model.tokenizer.decode(generated_tokens)
print(generated_text)
