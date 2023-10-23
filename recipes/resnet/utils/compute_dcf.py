#!/usr/bin/env python3

import numpy as np


from recipes.resnet.utils.scoring import read_scores, read_xvector, read_keys, compute_fpr_fnr_threshold
from recipes.resnet.utils.compute_cosine import scoring_xvector
import argparse


def compute_dcf(fpr, fnr, thresholds, p_target, c_miss, c_fa):
    """
     This function calculates the value of the dcf based on the values fpr and fnr
     This code is taken from https://github.com/kaldi-asr/kaldi/blob/71f38e62cad01c3078555bfe78d0f3a527422d75/egs/sre08/v1/sid/compute_min_dcf.py
     """
    min_c_det = float("inf")
    min_c_det_threshold = thresholds[0]
    for i in range(0, len(fnr)):
        # See Equation (2).  it is a weighted sum of false negative
        # and false positive errors.
        c_det = c_miss * fnr[i] * p_target + c_fa * fpr[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = thresholds[i]
    # See Equations (3) and (4).  Now we normalize the cost.
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def
    return min_dcf, min_c_det_threshold


def compute_score(keys, scores, p_target, c_miss, c_fa):
    """
     arg1 keys: dictionary with key : tuple with the names of the pairs of audio files, value : labels (0 : not the same speaker, 1 : same speaker)
     arg2 scores: dictionary with key : tuple with the names of the pairs of audio files, value : the similarity between their two xvectors
     arg3 p_target: The prior probability of the target speaker in a trial.
     arg4 c_miss: Cost of a missed detection.  This is usually not changed.
     arg5 c_fa: Cost of a spurious detection.  This is usually not changed.
     This function calculates the dcf from all the scores
     """

    fpr, fnr, thresholds = compute_fpr_fnr_threshold(keys, scores)

    return compute_dcf(fpr, fnr, thresholds, p_target, c_miss, c_fa)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('keys', metavar='keys', type=str,
                        help='the path to the the file where the keys are stocked')
    parser.add_argument('scores', metavar='scores', type=str,
                        help='the path to the file where the scores are stocked')
    parser.add_argument('--p-target', type=float, dest="p_target",
                        default=0.01,
                        help='The prior probability of the target speaker in a trial.')
    parser.add_argument('--c-miss', type=float, dest="c_miss", default=1,
                        help='Cost of a missed detection.  This is usually not changed.')
    parser.add_argument('--c-fa', type=float, dest="c_fa", default=1,
                        help='Cost of a spurious detection.  This is usually not changed.')

    args = parser.parse_args()

    trials = read_keys(args.keys)

    scores = read_scores(args.scores)

    dcf, _ = compute_score(trials, scores, args.p_target, args.c_miss, args.c_fa)
    print(dcf)




