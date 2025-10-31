#!/usr/bin/env python3

import argparse

import numpy as np
from sklearn import metrics

from kiwano.utils import read_keys, read_scores


def compute_fpr_fnr_threshold(keys, scores):
    labels = []
    distances = []

    for x in keys:
        labels.append(int(keys[x]))
        distances.append(float(scores[x]))

    """
    for label in keys.values():
        labels.append(int(label))

    for distance in scores.values():
        distances.append(float(distance))
    """

    positive_label = 1
    fpr, tpr, thresholds = metrics.roc_curve(
        labels, distances, pos_label=positive_label
    )
    return metrics.auc(fpr, tpr)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "keys",
        metavar="keys",
        type=str,
        help="the path to the the file where the keys are stocked",
    )
    parser.add_argument(
        "scores",
        metavar="scores",
        type=str,
        help="the path to the file where the scores are stocked",
    )
    parser.add_argument(
        "--p-target",
        type=float,
        dest="p_target",
        default=0.01,
        help="The prior probability of the target speaker in a trial.",
    )
    parser.add_argument(
        "--c-miss",
        type=float,
        dest="c_miss",
        default=1,
        help="Cost of a missed detection.  This is usually not changed.",
    )
    parser.add_argument(
        "--c-fa",
        type=float,
        dest="c_fa",
        default=1,
        help="Cost of a spurious detection.  This is usually not changed.",
    )

    args = parser.parse_args()

    trials = read_keys(args.keys)

    scores = read_scores(args.scores)

    auc = compute_fpr_fnr_threshold(trials, scores)

    print(auc)
