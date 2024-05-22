from transformers import AutoTokenizer

class Gpt2Tokenizer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
    
    def __call__(self, text):
        tokens = self.tokenizer(text)
        return tokens['input_ids']
    
    def decode(self, tokens):
        return self.tokenizer.decode(list(tokens))

