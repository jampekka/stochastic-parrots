from slm_nn import *

model = NnLanguageModel("jampekka/english_subs")

context = model.tokenize("Hello")
gen = list(model.generate(context, max_tokens=100))

print(model.detokenize(gen))
