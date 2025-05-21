import numpy as np


def recall_at_k(targets, predictions, k=10):
    targets = np.array(targets)
    hits = np.array([np.isin(target, pred[:k]) for target, pred in zip(targets, predictions)])
    return hits.mean()


def mean_reciprocal_rank(targets, predictions):
    targets = np.array(targets)
    ranks = []
    for target, pred in zip(targets, predictions):
        if target in pred:
            ranks.append(1 / (pred.index(target) + 1))
        else:
            ranks.append(0)
    return np.mean(ranks)
