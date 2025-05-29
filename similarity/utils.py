from sklearn.metrics import jaccard_score
import numpy as np

def jaccard_similarity(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    intersection = set1 & set2
    union = set1 | set2
    return len(intersection) / len(union) if union else 0
