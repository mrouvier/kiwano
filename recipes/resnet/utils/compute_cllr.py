#!/usr/bin/env python3

import numpy as np
from scipy.special import logit, expit
import copy

from kiwano.utils import read_scores, read_keys
import argparse


def read_targets_and_nontargets(keys, scores):
    """
    arg1 keys: dictionary with key : tuple with the names of the pairs of audio files, value : labels (0 : not the same speaker, 1 : same speaker)
    arg2 scores: dictionary with key : tuple with the names of the pairs of audio files, value : the similarity between their two xvectors
    This function returns a ndarray for the targets and another for the nontargets, they separate the scores based on their labels
    """

    targets = []
    nontargets = []

    for vectors in keys :
        if keys[vectors] == "1":
            targets.append(float(scores[vectors]))
        else :
            nontargets.append(float(scores[vectors]))

    return np.array(targets), np.array(nontargets)

def pavx(y):
    """
    args1 y : ndarray, the input vector
    This function implements the PAV : Pool Adjacent Violators algorithm. With respect to an input vector y, it computes ghat, a nondecreasing vector such as sum((y - ghat).^2) is minimal
    This code is taken from : https://github.com/DigitalPhonetics/VoicePAT/blob/4b86f44d6e4dc4cb529892e6a51d10519a2273b4/evaluation/privacy/asv/metrics/helpers.py#L122
    """
    assert y.ndim == 1, 'Argument should be a 1-D array'
    assert y.shape[0] > 0, 'Input array is empty'
    n = y.shape[0]

    index = np.zeros(n, dtype=int)
    length = np.zeros(n, dtype=int)

    ghat = np.zeros(n)

    ci = 0
    index[ci] = 1
    length[ci] = 1
    ghat[ci] = y[0]

    for j in range(1, n):
        ci += 1
        index[ci] = j + 1
        length[ci] = 1
        ghat[ci] = y[j]
        while (ci >= 1) & (ghat[np.max(ci - 1, 0)] >= ghat[ci]):
            nw = length[ci - 1] + length[ci]
            ghat[ci - 1] = ghat[ci - 1] + (length[ci] / nw) * (ghat[ci] - ghat[ci - 1])
            length[ci - 1] = nw
            ci -= 1

    height = copy.deepcopy(ghat[:ci + 1])
    width = copy.deepcopy(length[:ci + 1])

    while n >= 0:
        for j in range(index[ci], n + 1):
            ghat[j - 1] = ghat[ci]
        n = index[ci] - 1
        ci -= 1

    return ghat, width, height

def optimal_llr(tar, non, laplace=False, monotonicity_epsilon=1e-6):
    """
    args1 tar: ndarray which contains the scores associated to target pairs
    args2 non: ndarray which contains the scores associated to non-target pairs
    args3 laplace: bool, use Laplace technique to avoid infinite values of LLRs
    args4 monotonicity_epsilon: float, unsures monoticity of the optimal LLRs
    This function uses PAV algorithm to optimally calibrate the score
    This code is taken from : https://github.com/DigitalPhonetics/VoicePAT/blob/4b86f44d6e4dc4cb529892e6a51d10519a2273b4/evaluation/privacy/asv/metrics/helpers.py#L122
    """

    scores = np.concatenate([non, tar])
    Pideal = np.concatenate([np.zeros(len(non)), np.ones(len(tar))])

    perturb = np.argsort(scores, kind='mergesort')
    Pideal = Pideal[perturb]

    if laplace:
        Pideal = np.hstack([1, 0, Pideal, 1, 0])

    Popt, width, foo = pavx(Pideal)

    if laplace:
        Popt = Popt[2:len(Popt) - 2]

    posterior_log_odds = logit(Popt)
    log_prior_odds = np.log(len(tar) / len(non))
    llrs = posterior_log_odds - log_prior_odds
    N = len(tar) + len(non)
    llrs = llrs + np.arange(N) * monotonicity_epsilon / N  # preserve monotonicity

    idx_reverse = np.zeros(len(scores), dtype=int)
    idx_reverse[perturb] = np.arange(len(scores))
    tar_llrs = llrs[idx_reverse][len(non):]
    nontar_llrs = llrs[idx_reverse][:len(non)]


    return tar_llrs, nontar_llrs


def compute_cllr(tar_llrs, nontar_llrs):
    """
    args1 tar_llrs: ndarray which contains the scores associated to target pairs
    args2 nontar_llrs: ndarray which contains the scores associated to non-target pairs
    This function calculates the value of the cllr based on tar_llrs and nontar_llrs
    A part of this code is taken from https://github.com/DigitalPhonetics/VoicePAT/blob/4b86f44d6e4dc4cb529892e6a51d10519a2273b4/evaluation/privacy/asv/metrics/cllr.py#L25
    """

    tar_posterior = expit(tar_llrs)  # sigmoid
    non_posterior = expit(-nontar_llrs)
    if any(tar_posterior == 0) or any(non_posterior == 0):
        return np.inf

    c1 = (-np.log(tar_posterior)).mean() / np.log(2)
    c2 = (-np.log(non_posterior)).mean() / np.log(2)
    cllr = (c1 + c2) / 2
    return cllr

def compute_score(keys, scores):
    """
     arg1 keys: dictionary with key : tuple with the names of the pairs of audio files, value : labels (0 : not the same speaker, 1 : same speaker)
     arg2 scores: dictionary with key : tuple with the names of the pairs of audio files, value : the similarity between their two xvectors
     This function calculates the cllr from all the scores
     """

    targets, nontargets = read_targets_and_nontargets(keys, scores)
    targets, nontargets = optimal_llr(targets, nontargets)

    return compute_cllr(targets, nontargets)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('keys', metavar='keys', type=str,
                        help='the path to the the file where the keys are stocked')
    parser.add_argument('scores', metavar='scores', type=str,
                        help='the path to the file where the scores are stocked')

    args = parser.parse_args()

    trials = read_keys(args.keys)

    scores = read_scores(args.scores)

    cllr = compute_score(trials, scores)
    print(cllr)




