#!/usr/bin/env python3

import sys
from pathlib import Path
from typing import Optional, Union, List

import numpy as np


from kiwano.utils import read_scores, read_keys, compute_fpr_fnr_threshold

import argparse



def compute_eer(fpr, fnr):
    """
     This function calculates the value of the EER on the values fpr and fnr
     This code is taken from https://github.com/YuanGongND/python-compute-eer
     """
    # the threshold of fnr == fpr
    #eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]

    # theoretically eer from fpr and eer from fnr should be identical but they can be slightly differ in reality
    eer_1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer_2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]

    # return the mean of eer from fpr and from fnr
    eer = (eer_1 + eer_2) / 2
    return eer


def compute_score(keys, scores):
    """
     arg1 keys: dictionary with key : tuple with the names of the pairs of audio files, value : labels (0 : not the same speaker, 1 : same speaker)
     arg2 scores: dictionary with key : tuple with the names of the pairs of audio files, value : the similarity between their two xvectors
     This function calculates the value of the EER of all the scores
     """

    fpr, fnr, _ = compute_fpr_fnr_threshold(keys, scores)

    return compute_eer(fpr, fnr)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('keys', metavar='keys', type=str,
                        help='the path to the the file where the keys are stocked')
    parser.add_argument('scores', metavar='scores', type=str,
                        help='the path to the file where the scores are stocked')

    args = parser.parse_args()

    trials = read_keys(args.keys)

    scores = read_scores(args.scores)

    err = compute_score(trials, scores)
    print(err*100)




