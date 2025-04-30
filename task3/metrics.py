import numpy as np


def recall_at_k(target, predict, k=10):
    correct = 0
    for i in range(len(target)):
        if target[i] in predict[i][:k]:
            correct += 1
    return correct / len(target)


def mean_reciprocal_rank(target, predict):
    mrr = 0
    for i in range(len(target)):
        rank = -1
        for j, doc in enumerate(predict[i]):
            if doc == target[i]:
                rank = j + 1
                break
        if rank != -1:
            mrr += 1 / rank
    return mrr / len(target)
