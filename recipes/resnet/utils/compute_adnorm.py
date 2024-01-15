import argparse
import cProfile

from kiwano.embedding import EmbeddingSet, read_pkl
from recipes.resnet.utils.compute_asnorm import ASNorm
from recipes.resnet.utils.compute_snorm import DotProductStrategy, CosineStrategy
from recipes.resnet.utils.scoring import read_keys
import torch
import numpy as np


class ADNorm(ASNorm):

    def __init__(self, trials, enrollment, test, impostors, computeStrategy, k):
        super().__init__(trials, enrollment, test, impostors, computeStrategy, k)

    def compute_score(self, xvectorEnrollment, xvectorTest, enrollmentName, testName):

        enrollment = self.v_embeddings.get(enrollmentName)
        if enrollment is None:
            cohort_e, _ = self.select_impostors(xvectorEnrollment, self.vi, enrollmentName)
            cohort_e = [self.impostors[i] for i in cohort_e]
            impostors_enrollment_mean_vector = torch.mean(torch.stack(cohort_e), dim=0)
            enrollment = xvectorEnrollment - impostors_enrollment_mean_vector
            self.v_embeddings[enrollmentName] = enrollment

        test = self.v_embeddings.get(testName)
        if test is None:
            cohort_t, _ = self.select_impostors(xvectorTest, self.vi, testName)
            cohort_t = [self.impostors[i] for i in cohort_t]
            impostors_test_mean_vector = torch.mean(torch.stack(cohort_t), dim=0)
            test = xvectorTest - impostors_test_mean_vector
            self.v_embeddings[testName] = test


        adnorm = self.computeStrategy.scoring_xvector(xvectorEnrollment, xvectorTest)
        return adnorm


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

    adnorm = ADNorm(trials, enrollment, test, impostors, computeStrategy, args.k)
    adnorm.compute_norm()


