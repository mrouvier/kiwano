import argparse
import cProfile

import numpy as np
import torch

from kiwano.embedding import EmbeddingSet, read_pkl
from recipes.resnet.utils.compute_asnorm import ASNorm
from recipes.resnet.utils.compute_snorm import CosineStrategy, DotProductStrategy
from recipes.resnet.utils.scoring import read_keys


class ADNorm(ASNorm):
    def __init__(self, trials, enrollment, test, impostors, computeStrategy, k):
        super().__init__(trials, enrollment, test, impostors, computeStrategy, k)
        # cohorts is here a dictionary which contains keys : the name of the embedding and value : the normalized vector

    def compute_score(self, xvectorEnrollment, xvectorTest, enrollmentName, testName):

        normalized_ve = self.cohorts.get(enrollmentName)
        if normalized_ve is None:
            ve = self.compute_v(xvectorEnrollment)
            cohort_ve = self.select_impostors(ve)
            cohort_ve = [self.impostors[i] for i in cohort_ve]
            normalized_ve = xvectorEnrollment - torch.mean(
                torch.stack(cohort_ve), dim=0
            )
            self.cohorts[enrollmentName] = normalized_ve

        normalized_vt = self.cohorts.get(testName)
        if normalized_vt is None:
            vt = self.compute_v(xvectorTest)
            cohort_vt = self.select_impostors(vt)
            cohort_vt = [self.impostors[i] for i in cohort_vt]
            normalized_vt = xvectorTest - torch.mean(torch.stack(cohort_vt), dim=0)
            self.cohorts[testName] = normalized_vt

        return self.computeStrategy.scoring_xvector(normalized_ve, normalized_vt)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "keys",
        metavar="keys",
        type=str,
        help="the path to the file where the keys are stocked",
    )
    parser.add_argument(
        "xvectorEnrollment",
        metavar="xvectorEnrollment",
        type=str,
        help="command to gather xvectors enrollment in pkl format",
    )
    parser.add_argument(
        "xvectorTest",
        metavar="xvectorTest",
        type=str,
        help="command to gather xvectors test in pkl format",
    )
    parser.add_argument(
        "impostors",
        metavar="impostors",
        type=str,
        help="command to gather xvectors for the impostor set in pkl format ",
    )
    parser.add_argument(
        "k", metavar="k", type=int, help="numbers of impostors in the cohort"
    )

    args = parser.parse_args()
    trials = read_keys(args.keys)
    enrollment = read_pkl(args.xvectorEnrollment)
    test = read_pkl(args.xvectorTest)
    impostors = read_pkl(args.impostors)

    computeStrategy = DotProductStrategy()

    adnorm = ADNorm(trials, enrollment, test, impostors, computeStrategy, args.k)
    adnorm.compute_norm()
    # cProfile.run('adnorm.compute_norm()')
