import argparse
from kiwano.embedding import EmbeddingSet, read_pkl
from recipes.resnet.utils.compute_snorm import SNorm, DotProductStrategy, CosineStrategy
from recipes.resnet.utils.scoring import read_keys
import torch
import numpy as np

import cProfile
import random
from math import dist

class ASNorm(SNorm):

    def __init__(self, trials, enrollment, test, impostors, computeStrategy, k):
        super().__init__(trials, enrollment, test, impostors, computeStrategy)
        self.vi = self.compute_all_vi()
        self.k = k

    def compute_ve_vt(self, xvectorEnrollment, xvectorTest, enrollmentName, testName):

        impostors_enrollment, ve = self.select_impostors(xvectorEnrollment, self.vi, enrollmentName)
        impostors_test, vt = self.select_impostors(xvectorTest, self.vi, testName)

        vt_e = [vt[i] for i in impostors_enrollment]
        ve_t = [ve[i] for i in impostors_test]

        return ve_t, vt_e

    def compute_all_vi(self):
        #vi list with a list for each impostor with all the scores between that impostor and each impostor of the set

        vi = []
        for i in range(len(self.impostors)):
            scores = []
            for k in range(0, i):
                scores.append(vi[k][i])
            for j in range(i, len(self.impostors)):
                scores.append(self.computeStrategy.scoring_xvector(self.impostors[i], self.impostors[j]))
            vi.append(scores)

        return vi


    def select_impostors(self, targetEmbedding, vi, targetName):
        #select the first K impostors which are closest to the vector with all the scores between the targetEmbedding and each impostor

        if targetName in self.v_embeddings :
            scores_target = self.v_embeddings[targetName]
        else:
            scores_target = self.compute_scores_among_all_impostors(targetEmbedding)

        vi_array = np.array(vi)
        distance_squared = np.sum((np.array(scores_target) - vi_array) ** 2, axis=1)
        cohort = np.argsort(distance_squared)[::-1][:self.k]

        return cohort.tolist(), scores_target


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('keys', metavar='keys', type=str,
                        help='the path to the file where the keys are stocked')
    parser.add_argument('xvectorEnrollment', metavar='xvectorEnrollment', type=str,
                        help='command to gather xvectors enrollment in pkl format')
    parser.add_argument('xvectorTest', metavar='xvectorTest', type=str,
                        help='command to gather xvectors test in pkl format')
    parser.add_argument('impostors', metavar='impostors', type=str,
                        help='command to gather xvectors for the impostor set in pkl format ')

    parser.add_argument('k', metavar='k', type=int,
                        help='numbers of impostors in the cohort')

    args = parser.parse_args()
    trials = read_keys(args.keys)
    enrollment = read_pkl(args.xvectorEnrollment)
    test = read_pkl(args.xvectorTest)
    impostors = read_pkl(args.impostors)

    computeStrategy = DotProductStrategy()

    asnorm = ASNorm(trials, enrollment, test, impostors, computeStrategy, args.k)
    asnorm.compute_norm()
