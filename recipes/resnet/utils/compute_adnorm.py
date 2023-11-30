import argparse
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

        impostors_enrollment, _ = self.select_impostors(xvectorEnrollment, self.vi, enrollmentName)
        impostors_test, _ = self.select_impostors(xvectorTest, self.vi, testName)

        cohort_e = [self.impostors[i] for i in impostors_enrollment]
        cohort_t = [self.impostors[i] for i in impostors_test]

        impostors_enrollment_mean_vector = torch.mean(torch.stack(cohort_e), dim=0)
        impostors_test_mean_vector = torch.mean(torch.stack(cohort_t), dim=0)

        xvectorEnrollment = xvectorEnrollment - impostors_enrollment_mean_vector
        xvectorTest = xvectorTest - impostors_test_mean_vector

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
