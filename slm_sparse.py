import sparse
import numpy as np

class FrequencyTablePredictor:
    def __init__(self, context_length, vocabulary_size):
        self.context_length = context_length
        shape = [vocabulary_size]*(context_length + 1)
        self.follower_table = sparse.DOK(shape=tuple(shape), dtype=float) 

    def __call__(self, context):
        context = tuple(context)
        candidates = self.follower_table[context]
        
        # TODO: Could be faster
        if not candidates.data: return None
        tokens, counts = zip(*((k[0], v) for k, v in candidates.data.items()))
        weights = np.array(counts, dtype=float)
        weights /= np.sum(weights)
        return np.random.choice(tokens, p=weights)

    def train(self, xys):
        for context, target in xys:
            self.train_one(context, target)

    def train_one(self, context, target):
        idx = (*context, target)
        self.follower_table[idx] += 1
