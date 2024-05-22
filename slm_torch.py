import torch
from transformers import AutoTokenizer, AutoModel
# TODO: Try to use torch-only
import numpy as np
from collections import Counter
from slm import *


class Gpt2Tokenizer:
    def __init__(self):
        # TODO: Does this check for updates?
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

class EmbeddingTablePredictor(FrequencyTablePredictor):
    def __init__(self, embedder, context_length):
        super().__init__(context_length=context_length)
        self.embedder = embedder
        
        # TODO: using dict here is extremely slow.
        # A sparse tensor could work? Or a proper k-NN if needed
        self.context_embeddings = {}

    def train_one(self, context, target):
        context = tuple(context)
        if context not in self.context_embeddings:
            self.context_embeddings[context] = self.embedder(context)
        super().train_one(context, target)
    
    def __call__(self, context):
        output = super().__call__(context)
        if output is not None: return output
        
        context, dist = self._get_closest_context(context)
        return super().__call__(context)

    def _get_closest_context(self, context):
        # Oh lord, this is really slow!
        assert len(context) == self.context_length

        contexts, embeddings = zip(*self.context_embeddings.items())
        
        embedding = self.embedder(context)
        embeddings = torch.stack(embeddings)
        
        embeddings = torch.flatten(embeddings, 1)
        embedding = torch.flatten(self.embedder(context))
        errors = embeddings - embedding
        errors = torch.mean(torch.abs(errors), dim=1)
        
        diff, i = torch.min(errors, 0)
        return contexts[i], diff


