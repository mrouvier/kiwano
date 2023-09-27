#!/usr/bin/env python3

import sys
from pathlib import Path
from typing import Optional, Union, List

import numpy as np
import torch
import time
from torch import nn

from kiwano.utils import Pathlike
from kiwano.features import Fbank
from kiwano.augmentation import Augmentation, Noise, Codec, Filtering, Normal, Sometimes, Linear, CMVN, Crop
from kiwano.dataset import Segment, SegmentSet
from kiwano.model import ResNet

import sklearn.metrics

#trials

def read_xvector(file_path: str):
    h = {}
    f = open(file_path, "r")
    for line in f:
        line = line.strip().split(" ")
        arr = line[1:-1]
        my_list = [float(i) for i in arr]
        h[ line[0] ] = torch.Tensor(my_list)
    f.close
    return h

def compute_score(keys, scores):
    """
    arg1 keys: path to the file with the key (0 : not the same speaker, 1 : same speaker) and the names of the pairs of audio files
    arg2 scores: path to the file with the names of the pairs of audio files and the similarity between their two xvectors
    This function calculates the value of the EER of all the scores
    A part of this code is taken from https://github.com/YuanGongND/python-compute-eer
    """

    labels = []
    distances = []

    with open(keys, "r") as file:
        for line in file :
            line = line.split(" ")
            labels.append(int(line[0]))
    with open(scores, "r") as file:
        for line in file:
            line = line.split(" ")
            distances.append(float(line[2]))

    positive_label = 1
    # all fpr, tpr, fnr, fnr, threshold are lists (in the format of np.array)
    fpr, tpr, threshold = sklearn.metrics.roc_curve(labels, distances, pos_label = positive_label)
    fnr = 1 - tpr

    # the threshold of fnr == fpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]

    # theoretically eer from fpr and eer from fnr should be identical but they can be slightly differ in reality
    eer_1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer_2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]

    # return the mean of eer from fpr and from fnr
    eer = (eer_1 + eer_2) / 2

    return eer

def scoring_xvector(keys, xvectors_enrollment, xvectors_test):
    """
    arg1 keys: path to the file with the key (0 : not the same speaker, 1 : same speaker) and the names of the pairs of audio files
    arg2 xvectors_enrollment: dictionary with key : the name of the audio file, value : the corresponding xvector enrollment
    arg3 xvectors_test: dictionary with key : the name of the audio file, value : the corresponding xvector test
    This function creates a file scores.txt which contains the names of each pairs of audio files and their cosine similarity
    """
    cos = torch.nn.CosineSimilarity(dim=0)

    with open(keys, "r") as file:
        for line in file :
            line = line.split(" ")
            enrollmentName = line[1]
            testName = line[2]
            if keys.split("/")[-1] == "voxceleb1_test_v2.txt":
                enrollmentName = enrollmentName.replace("/", "_")[:-4]
                testName = testName.replace("/", "_").replace('\n', "")[:-4]

            xvectorEnrollment = xvectors_enrollment[enrollmentName]
            xvectorTest = xvectors_test[testName]
            score = cos(xvectorEnrollment, xvectorTest)

            with open("../exp/scores.txt", "a") as outputFile :
                outputFile.write(enrollmentName+" "+testName+" "+str(score.item())+"\n")





if __name__ == '__main__':
    enrollment = read_xvector(sys.argv[1])
    test = read_xvector(sys.argv[2])

    scoring_xvector("../db/voxceleb1/voxceleb1_test_v2.txt", enrollment, test)
    err = compute_score("../db/voxceleb1/voxceleb1_test_v2.txt", "../exp/scores.txt")
    print(err)




