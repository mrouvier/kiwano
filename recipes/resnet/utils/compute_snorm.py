import argparse
from kiwano.embedding import EmbeddingSet, read_pkl
from recipes.resnet.utils.scoring import read_keys
import torch
import numpy as np

import cProfile
import random
from math import dist

class ComputeStrategy:
    def scoring_xvector(self, targetEmbedding, impostor):
        pass

class CosineStrategy(ComputeStrategy):
    def scoring_xvector(self, targetEmbedding, impostor):
        cos = torch.nn.CosineSimilarity(dim=0)
        score = cos(targetEmbedding, impostor).item()
        return score

class DotProductStrategy(ComputeStrategy):
    def scoring_xvector(self, targetEmbedding, impostor):
        score = torch.dot(targetEmbedding, impostor).item()
        return score



class SNorm:

    def __init__(self, trials, enrollment, test, impostors, computeStrategy):
        #v_embeddings : dictionary with keys : the name of the embedding and value : ve or vt
        self.keys = trials
        self.xvectors_enrollment = enrollment
        self.xvectors_test = test
        self.impostors = self.prepare_impostors(impostors)
        self.computeStrategy = computeStrategy
        self.v_embeddings = {}

    def prepare_impostors(self, impostors):
        #averages vectors for the same speaker

        new_impostors = []

        #sort impostors by the spkid
        items_tries = sorted(impostors.h.items(), key=lambda x: (int(x[0][2:7]), x[0]))
        impostors.h = dict(items_tries)

        xvectors = []
        target_id = 0

        for key in impostors :

            current_id = key[2:7]
            if target_id == 0 : #init target_id
                target_id = current_id
            if current_id != target_id :
                target_id = current_id
                xvectors_concat = torch.stack(xvectors)
                mean_vector = torch.mean(xvectors_concat, dim=0)
                xvectors = []
                new_impostors.append(mean_vector)

            else :
                xvectors.append(impostors[key])
        xvectors_concat = torch.stack(xvectors)
        mean_vector = torch.mean(xvectors_concat, dim=0)
        new_impostors.append(mean_vector)

        return tuple(new_impostors)



    def compute_scores_among_all_impostors(self, targetEmbedding):
        # targetEmbedding : xvecteur, impostors : liste de xvecteur
        #compute all the scores between the targetEmbedding and each impostor

        scores = []
        for impostor in self.impostors:
            scores.append(self.computeStrategy.scoring_xvector(targetEmbedding, impostor))
        return scores

    def compute_ve_vt(self, xvectorEnrollment, xvectorTest, enrollmentName, testName):
        if enrollmentName in self.v_embeddings :
            ve = self.v_embeddings[enrollmentName]
        else:
            ve = self.compute_scores_among_all_impostors(xvectorEnrollment)
        if testName in self.v_embeddings:
            vt = self.v_embeddings[testName]
        else:
            vt = self.compute_scores_among_all_impostors(xvectorTest)
        return ve, vt

    def compute_score(self, xvectorEnrollment, xvectorTest, enrollmentName, testName):
        ve, vt = self.compute_ve_vt(xvectorEnrollment, xvectorTest, enrollmentName, testName)

        score = self.computeStrategy.scoring_xvector(xvectorEnrollment, xvectorTest)

        mean_enrollment = np.mean(ve)
        mean_test = np.mean(vt)

        standard_deviation_enrollment = np.std(ve)
        standard_deviation_test = np.std(vt)

        snorm = (score - mean_enrollment) / (2 * standard_deviation_enrollment) + (score - mean_test) / (2 * standard_deviation_test)
        return snorm

    def compute_norm(self):

        for names in self.keys:
            enrollmentName = names[0]
            testName = names[1]

            xvectorEnrollment = self.xvectors_enrollment[enrollmentName]
            xvectorTest = self.xvectors_test[testName]

            norm = self.compute_score(xvectorEnrollment, xvectorTest, enrollmentName, testName)

            print(enrollmentName + " " + testName + " " + str(norm))



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


    args = parser.parse_args()


    trials = read_keys(args.keys)
    enrollment = read_pkl(args.xvectorEnrollment)
    test = read_pkl(args.xvectorTest)
    impostors = read_pkl(args.impostors)

    computeStrategy = DotProductStrategy()

    snorm = SNorm(trials, enrollment, test, impostors, computeStrategy)
    snorm.compute_norm()
