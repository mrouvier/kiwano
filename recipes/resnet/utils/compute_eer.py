#!/usr/bin/env python3

import sys
from pathlib import Path
from typing import Optional, Union, List

import numpy as np
import sklearn.metrics
import torch
import time
from torch import nn

from kiwano.utils import Pathlike
from kiwano.features import Fbank
from kiwano.augmentation import Augmentation, Noise, Codec, Filtering, Normal, Sometimes, Linear, CMVN, Crop
from kiwano.dataset import Segment, SegmentSet
from kiwano.model import ResNet

import argparse



#trials

def read_xvector(file_path: str):
    h = {}
    f = open(file_path, "r")
    for line in f:
        line = line.strip().split(" ")
        arr = line[1:]
        my_list = [float(i) for i in arr]
        h[ line[0] ] = torch.Tensor(my_list)
    f.close
    return h

def read_keys(file_path: str):
    h = {}
    f = open(file_path, "r")
    for line in f:
        line = line.strip().split(" ")
        label = line[0]
        enrollmentName = line[1]
        testName = line[2]
        if file_path.split("/")[-1] == "voxceleb1_test_v2.txt" or file_path.split("/")[-1] == "list_test_hard2.txt" or file_path.split("/")[-1] == "list_test_all2.txt":
            enrollmentName = enrollmentName.replace("/", "_")[:-4]
            testName = testName.replace("/", "_").replace('\n', "")[:-4]
        h[(enrollmentName, testName)] = label
    f.close
    return h


def read_scores(file_path: str):
    h = {}
    f = open(file_path, "r")
    for line in f:
        line = line.strip().split(" ")
        enrollmentName = line[0]
        testName = line[1]
        score = line[2]
        h[(enrollmentName, testName)] = score
    f.close
    return h


def scoring_xvector(keys, xvectors_enrollment, xvectors_test, output_dir):
    """
    arg1 keys: dictionary with key : tuple with the names of the pairs of audio files, value labels (0 : not the same speaker, 1 : same speaker)
    arg2 xvectors_enrollment: dictionary with key : the name of the audio file, value : the corresponding xvector enrollment
    arg3 xvectors_test: dictionary with key : the name of the audio file, value : the corresponding xvector test
    arg4 output_dir : the path and the name of the file where the scores will be saved
    This function creates a file scores.txt which contains the names of each pairs of audio files and their cosine similarity
    """

    cos = torch.nn.CosineSimilarity(dim=0)

    for names in keys :
        enrollmentName = names[0]
        testName = names[1]

        xvectorEnrollment = xvectors_enrollment[enrollmentName]
        xvectorTest = xvectors_test[testName]
        score = cos(xvectorEnrollment, xvectorTest)

        with open(output_dir, "a") as outputFile:
            outputFile.write(enrollmentName + " " + testName + " " + str(score.item()) + "\n")


def compute_fpr_fnr(keys, scores):
    """
     arg1 keys: dictionary with key : tuple with the names of the pairs of audio files, value : labels (0 : not the same speaker, 1 : same speaker)
     arg2 scores: dictionary with key : tuple with the names of the pairs of audio files, value : the similarity between their two xvectors
     This function calculates and returns the fpr and the fnr
     A part of this code is taken from https://github.com/YuanGongND/python-compute-eer
     """

    labels = []
    distances = []

    for label in keys.values():
        labels.append(int(label))

    for distance in scores.values():
        distances.append(float(distance))

    positive_label = 1
    # all fpr, tpr, fnr, fnr, threshold are lists (in the format of np.array)
    fpr, tpr, threshold = sklearn.metrics.roc_curve(labels, distances, pos_label=positive_label)
    fnr = 1 - tpr

    return fpr, fnr


def compute_eer(fpr, fnr):
    """
     This function calculates the value of the EER on the values fpr and fnr
     A part of this code is taken from https://github.com/YuanGongND/python-compute-eer
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

    fpr, fnr = compute_fpr_fnr(keys, scores)

    return compute_eer(fpr, fnr)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('keys', metavar='keys', type=str,
                        help='the path to the the file where the keys are stocked')
    parser.add_argument('xvectorEnrollment', metavar='xvectorEnrollment', type=str,
                        help='the path to the the file where the xvector enrollment are stocked')
    parser.add_argument('xvectorTest', metavar='xvectorTest', type=str,
                        help='the path to the the file where the xvector test are stocked')
    parser.add_argument('output_dir', metavar='output_dir', type=str,
                        help='the path and the name of the file where the scores will be saved')

    args = parser.parse_args()

    trials = read_keys(args.keys)
    enrollment = read_xvector(args.xvectorEnrollment)
    test = read_xvector(args.xvectorTest)

    scoring_xvector(trials, enrollment, test, args.output_dir)

    scores = read_scores(args.output_dir)

    err = compute_score(trials, scores)
    print(err)




