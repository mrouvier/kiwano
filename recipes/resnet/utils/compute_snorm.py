import argparse
from kiwano.embedding import EmbeddingSet, read_pkl
from recipes.resnet.utils.scoring import read_keys
import torch
import numpy as np

import cProfile

class ComputeStrategy:
    def scoring_xvector(self, targetEmbedding, impostor):
        pass

class CosineStrategy(ComputeStrategy):
    def scoring_xvector(self, targetEmbedding, impostor):
        cos = torch.nn.CosineSimilarity(dim=0)
        return cos(targetEmbedding, impostor).item()

class DotProductStrategy(ComputeStrategy):
    def scoring_xvector(self, targetEmbedding, impostor):
        return torch.dot(targetEmbedding, impostor).item()



class SNorm:

    def __init__(self, trials, enrollment, test, impostors, computeStrategy):
        #v_mean_std : dictionary with keys : the name of the embedding and value : tuple with the result of the mean and the std
        self.keys = trials
        self.xvectors_enrollment = enrollment
        self.xvectors_test = test
        self.impostors = self.prepare_impostors(impostors)
        self.computeStrategy = computeStrategy
        self.mean_std = {}

    def prepare_impostors(self, impostors):
        #averages vectors for the same speaker

        new_impostors = []

        impostors_by_speaker = {}

        #sort impostors by the spkid
        for key, value in impostors.h.items():
            id_position = key.find("id")
            current_id = key[id_position + 2:id_position + 7]
            #current_id = key[2:7]
            if impostors_by_speaker.get(current_id) is None:
                impostors_by_speaker[current_id] = []
            impostors_by_speaker[current_id].append(value)

        # Calculate mean vectors
        for xvectors in impostors_by_speaker.values():
            xvectors_concat = torch.stack(xvectors)
            mean_vector = torch.mean(xvectors_concat, dim=0)
            # next line require for dot product, but has to be removed for cosine similarity :
            mean_vector = mean_vector / np.linalg.norm(mean_vector, ord=2)
            new_impostors.append(mean_vector)

        return tuple(new_impostors)

    def compute_v(self, xvector):
        # targetEmbedding : xvecteur, impostors : liste de xvecteur
        # compute all the scores between the targetEmbedding and each impostor, return a numpy array

        return np.array([self.computeStrategy.scoring_xvector(xvector, impostor) for impostor in self.impostors])


    def compute_score(self, xvectorEnrollment, xvectorTest, enrollmentName, testName):

        mean_std_ve = self.mean_std.get(enrollmentName)
        if mean_std_ve is None :
            ve = self.compute_v(xvectorEnrollment)
            mean_std_ve = (np.mean(ve), np.std(ve))
            self.mean_std[enrollmentName] = mean_std_ve

        mean_std_vt = self.mean_std.get(testName)
        if mean_std_vt is None :
            vt = self.compute_v(xvectorTest)
            mean_std_vt = (np.mean(vt), np.std(vt))
            self.mean_std[testName] = mean_std_vt

        score = self.computeStrategy.scoring_xvector(xvectorEnrollment, xvectorTest)

        return ((score - self.mean_std[enrollmentName][0]) / (self.mean_std[enrollmentName][1]) + (score - self.mean_std[testName][0]) / (self.mean_std[testName][1]))*0.5

    def compute_norm(self):

        for names in self.keys:
            enrollmentName = names[0]
            testName = names[1]

            print(enrollmentName + " " + testName + " " + str(self.compute_score(self.xvectors_enrollment[enrollmentName], self.xvectors_test[testName], enrollmentName, testName)))



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
    #computeStrategy = CosineStrategy()

    snorm = SNorm(trials, enrollment, test, impostors, computeStrategy)
    snorm.compute_norm()
    #cProfile.run('snorm.compute_norm()')