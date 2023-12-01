import argparse
from kiwano.embedding import EmbeddingSet, read_pkl
from recipes.resnet.utils.scoring import read_keys
import torch
import numpy as np

import cProfile
import random
from math import dist

import statistics

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
        self.v_means = {}
        self.v_std = {}

    def prepare_impostors(self, impostors):
        #averages vectors for the same speaker

        new_impostors = []

        impostors_by_speaker = {}

        #sort impostors by the spkid
        for key, value in impostors.h.items():
            current_id = key[2:7]
            if impostors_by_speaker.get(current_id) is None:
                impostors_by_speaker[current_id] = []
            impostors_by_speaker[current_id].append(value)

        # Calculate mean vectors
        for xvectors in impostors_by_speaker.values():
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

        ve = self.v_embeddings.get(enrollmentName)
        if ve is None:
            ve = self.compute_scores_among_all_impostors(xvectorEnrollment)
            self.v_embeddings[enrollmentName] = ve

        vt = self.v_embeddings.get(testName)
        if vt is None:
            vt = self.compute_scores_among_all_impostors(xvectorTest)
            self.v_embeddings[testName] = vt


        return ve, vt

    def compute_score(self, xvectorEnrollment, xvectorTest, enrollmentName, testName):
        ve, vt = self.compute_ve_vt(xvectorEnrollment, xvectorTest, enrollmentName, testName)

        score = self.computeStrategy.scoring_xvector(xvectorEnrollment, xvectorTest)

        mean_enrollment = self.v_means.get(enrollmentName)
        if mean_enrollment is None:
            mean_enrollment = np.mean(ve)
            self.v_means[enrollmentName] = mean_enrollment

        mean_test = self.v_means.get(testName)
        if mean_test is None:
            mean_test = np.mean(vt)
            self.v_means[testName] = mean_test

        standard_deviation_enrollment = self.v_std.get(enrollmentName)
        if standard_deviation_enrollment is None:
            standard_deviation_enrollment = np.std(ve)
            self.v_std[enrollmentName] = standard_deviation_enrollment

        standard_deviation_test = self.v_std.get(testName)
        if standard_deviation_test is None:
            standard_deviation_test = np.std(vt)
            self.v_std[testName] = standard_deviation_test

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

    #computeStrategy = DotProductStrategy()
    computeStrategy = CosineStrategy()

    snorm = SNorm(trials, enrollment, test, impostors, computeStrategy)
    snorm.compute_norm()


