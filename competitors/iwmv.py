import random
from collections import defaultdict
import math
import numpy as np


def list_mean(list_):
    sum_ = 0
    for ele in list_:
        sum_ += ele
    return sum_ * 1.0 / len(list_)

def dict_values_allclose(dict1, dict2, rtol = 1e-3, atol=1e-03):
    for item in dict1.keys():
        for class_ in dict1[item].keys():
            if math.fabs(dict1[item][class_] - dict2[item][class_]) > atol + rtol * math.fabs(dict2[item][class_]):
                return False
    return True

def iwmv(e2wl, w2el, label_set, its_ = 3,
         T_required=False):
    reliabilities = []
    truths = {}
    votes = {}
    v = dict()
    v = defaultdict(lambda: 1, v)
    worker_num_correct = {}
    worker_num_correct = defaultdict(lambda: 0, worker_num_correct)
    for it in range(its_):
        worker_num_correct.clear()
        # update truth
        for item in e2wl.keys():
            item_votes = {}
            for class_ in label_set:
                item_votes[class_] = 0
            for worker, label in e2wl[item]:
                for class_ in label_set:
                    if label == class_:
                        item_votes[class_] += v[worker]
            truths[item] = extract_truth_from_dict(item_votes)
            for worker, label in e2wl[item]:
                if label == truths[item]:
                    worker_num_correct[worker] += 1
            votes[item] = item_votes

        # if len(prev_votes) != 0 and dict_values_allclose(prev_votes, votes):
        #     break

        # update worker ability
        for worker, worker_labels in w2el.items():
            if it == its_ -1 :
                reliabilities.append((worker_num_correct[worker] / len(worker_labels)))
            v[worker] = len(label_set) * (worker_num_correct[worker] / len(worker_labels)) - 1
    if T_required:
        labels_distribution = estimate_labels_distribution(truths, label_set)
        return truths, it+1, labels_distribution, generate_T(reliabilities)
    else:
        return truths, it+1

def generate_T(samples):
    T_matrices = []
    for single_sample in samples:
        T_matrix = np.array([[single_sample, 1-single_sample], [1-single_sample, single_sample]])
        T_matrices.append(T_matrix)
    weighted_sum = np.mean(np.array(T_matrices), axis=0)
    return weighted_sum


def extract_truth_from_dict(dict_):
    max_ = -9999
    candidate = []
    for class_ in dict_.keys():
        val = dict_[class_]
        if val > max_:
            max_ = val
            candidate.clear()
            candidate.append(class_)
        elif val == max_:
            candidate.append(class_)
        else:
            continue
    return random.choice(candidate)

def estimate_labels_distribution(truths, label_set):
    truths = np.array(list(truths.values()))
    _, counts = np.unique(truths, return_counts=True)
    counts = counts / truths.shape[0]
    while counts.shape[0] < len(label_set):
        counts = np.append(counts, 0.0)
    return counts 