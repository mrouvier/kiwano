import argparse
import cProfile
import math

import numpy as np
from recipes.resnet.utils.compute_snorm import CosineStrategy, DotProductStrategy, SNorm
from recipes.resnet.utils.scoring import read_keys

from kiwano.embedding import EmbeddingSet, read_pkl


class ASNorm(SNorm):
    def __init__(self, trials, enrollment, test, impostors, computeStrategy, k):
        # cohorts : dictionary with keys : the name of the embedding and value : indices of the closest impostors
        super().__init__(trials, enrollment, test, impostors, computeStrategy)
        self.vi = self.compute_all_vi()
        self.k = k
        self.cohorts = {}
        self.v = {}

    def select_impostors(self, v):
        # select the first K impostors which are closest to the vector with all the scores between the targetEmbedding and each impostor

        distance_squared = [np.linalg.norm(vi - v) ** 2 for vi in self.vi]
        return np.argsort(distance_squared)[: self.k]

    def compute_all_vi(self):
        # vi list with a list for each impostor with all the scores between that impostor and each impostor of the set

        vi = np.zeros((len(self.impostors), len(self.impostors)))

        for i in range(len(self.impostors)):
            for j in range(i, len(self.impostors)):
                score = self.computeStrategy.scoring_xvector(
                    self.impostors[i], self.impostors[j]
                )
                vi[i, j] = score

        return vi + vi.T - np.diag(vi.diagonal())

    def compute_score(self, xvectorEnrollment, xvectorTest, enrollmentName, testName):

        ve = self.v.get(enrollmentName)
        if self.cohorts.get(enrollmentName) is None:
            ve = self.compute_v(xvectorEnrollment)
            self.v[enrollmentName] = ve
            self.cohorts[enrollmentName] = self.select_impostors(ve)

        vt = self.v.get(testName)
        if self.cohorts.get(testName) is None:
            vt = self.compute_v(xvectorTest)
            self.v[testName] = vt
            self.cohorts[testName] = self.select_impostors(vt)

        score = self.computeStrategy.scoring_xvector(xvectorEnrollment, xvectorTest)

        ve_t = ve[self.cohorts[testName]]
        vt_e = vt[self.cohorts[enrollmentName]]

        return 0.5 * (
            (score - np.mean(ve_t)) / (np.std(ve_t))
            + (score - np.mean(vt_e)) / (np.std(vt_e))
        )


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

    asnorm = ASNorm(trials, enrollment, test, impostors, computeStrategy, args.k)

    asnorm.compute_norm()
    # cProfile.run('asnorm.compute_norm()')
