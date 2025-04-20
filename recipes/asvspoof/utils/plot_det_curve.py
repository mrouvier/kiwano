#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import sklearn.metrics
from scipy.stats import norm

from recipes.resnet.utils.compute_eer import compute_eer
from recipes.resnet.utils.scoring import (
    compute_fpr_fnr_threshold,
    read_keys,
    read_scores,
)


def plot_det_curve(keys, scores, systems, output_dir, nameOutputFile):
    """
    arg1 keys: list of dictionaries with key : tuple with the names of the pairs of audio files, value : labels (0 : not the same speaker, 1 : same speaker)
    arg2 scores: list of dictionaries with key : tuple with the names of the pairs of audio files, value : the similarity between their two xvectors
    arg3 systems: list of the names of the systems
    arg4 output_dir: the path to the directory where the plots will be stored
    arg5 nameOutputFile: the name of the future plot file
    This function creates a plot based on the different systems in input
    A part of this code is taken from https://github.com/wenet-e2e/wespeaker/blob/master/wespeaker/utils/score_metrics.py#L119"""

    output_file = Path(output_dir + "/" + nameOutputFile)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    xytick = [
        0.0001,
        0.0002,
        0.0005,
        0.001,
        0.002,
        0.005,
        0.01,
        0.02,
        0.05,
        0.1,
        0.2,
        0.4,
    ]
    xytick_labels = map(str, [x * 100 for x in xytick])

    plt.xticks(norm.ppf(xytick), xytick_labels)
    plt.yticks(norm.ppf(xytick), xytick_labels)
    plt.xlim(norm.ppf([0.00051, 0.5]))
    plt.ylim(norm.ppf([0.00051, 0.5]))
    plt.xlabel("false-alarm rate [%]", fontsize=12)
    plt.ylabel("false-reject rate [%]", fontsize=12)

    for i in range(len(keys)):

        fpr, fnr, _ = compute_fpr_fnr_threshold(keys[i], scores[i])

        p_miss = norm.ppf(fnr)
        p_fa = norm.ppf(fpr)
        eer = compute_eer(fpr, fnr)

        plt.plot(
            p_fa,
            p_miss,
            label=systems[i] + ", eer = " + str(round((eer * 100), 2)) + " %",
        )
        plt.plot(norm.ppf(eer), norm.ppf(eer), "o")

    plt.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True
    )
    plt.tight_layout()
    plt.grid()

    if output_dir is not None:
        plt.savefig(output_file)
        plt.clf()
    else:
        plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "output_dir",
        metavar="output_dir",
        type=str,
        help="the path to the directory where the plots will be stored",
    )
    parser.add_argument(
        "name_output_file",
        metavar="name_output_file",
        type=str,
        help="the name of the future plot file",
    )
    parser.add_argument(
        "keys_scores_systems",
        metavar="keys_scores_systems",
        nargs="+",
        type=str,
        help="sequence of : a path to a file where keys are stocked, path to the associated file where the scores are stocked, and then the name of the associated system",
    )

    args = parser.parse_args()

    elements = args.keys_scores_systems

    path_keys = []
    path_scores = []
    systems = []

    if len(elements) % 3 != 0:
        parser.error("keys must be associated with a score and a system name")

    for i in range(0, len(elements), 3):
        path_keys.append(elements[i])
        path_scores.append(elements[i + 1])
        systems.append(elements[i + 2])

    dic_keys = []
    for key in path_keys:
        dic_keys.append(read_keys(key))
    dic_scores = []
    for score in path_scores:
        dic_scores.append(read_scores(score))

    plot_det_curve(
        dic_keys, dic_scores, systems, args.output_dir, args.name_output_file
    )
