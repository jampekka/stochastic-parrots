import torch
from transformers import AutoTokenizer, AutoModel
# TODO: Try to use torch-only
import numpy as np
from collections import Counter

class Gpt2Tokenizer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        self.tokenizer.model_max_length = 99999999
    
    def __call__(self, text):
        tokens = self.tokenizer(text)
        return tokens['input_ids']
    
    def decode(self, tokens):
        return self.tokenizer.decode(list(tokens))

class Gpt2Embedder:
    def __init__(self):
        self.model = AutoModel.from_pretrained('gpt2')
        self.embedder = self.model.get_input_embeddings()

    def __call__(self, tokens):
        embeddings = self.embedder(torch.tensor(tokens))
        return embeddings

# TODO: Very slow and stupid
class EmbeddingTablePredictor:
    def __init__(self, context_length):
        self.context_length = context_length
        
        self.contexts = []
        self.followers = []
    
    def train(self, xys):
        for context, target in xys:
            closest_i, diff = self.get_closest_context(context)
            if closest_i is None or not torch.isclose(diff, torch.tensor(0.0)):
                self.contexts.append(context)
                self.followers.append(Counter({target: 1}))
                continue
            
            self.followers[closest_i][target] += 1
    
    def __call__(self, context):
        closest_i, _ = self.get_closest_context(context)
        candidates = list(self.followers[closest_i].items())
        
        tokens, counts = zip(*candidates)
        weights = np.array(counts, dtype=float)
        weights /= np.sum(weights)
        return np.random.choice(tokens, p=weights)



    def get_closest_context(self, embedding):
        if len(self.contexts) == 0:
            return None, np.nan

        contexts = torch.stack(self.contexts)
        contexts = torch.flatten(contexts, 1)
        errors = contexts - torch.flatten(embedding)
        errors = torch.mean(torch.abs(errors), dim=1)
        
        diff, i = torch.min(errors, 0)
        return i, diff


