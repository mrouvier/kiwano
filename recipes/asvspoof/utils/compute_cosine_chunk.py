import argparse
from recipes.resnet.utils.scoring import read_keys, read_xvector
import torch
from pathlib import Path
from kiwano.embedding import EmbeddingSet, read_pkl

def scoring_xvector(keys, xvectors_enrollment, xvectors_test):
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

        mean = []
        for i in range(0,10):
            for j in range(0, 10):
                xvectorEnrollment = xvectors_enrollment[enrollmentName+"#"+str(i)]
                xvectorTest = xvectors_test[testName+"#"+str(j)]
                score = cos(xvectorEnrollment, xvectorTest)
                mean.append( score.item() )

        print(enrollmentName + " " + testName + " " + str(torch.mean( torch.Tensor(mean) ).item() ) )





if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('keys', metavar='keys', type=str,
                        help='the path to the the file where the keys are stocked')
    parser.add_argument('xvectorEnrollment', metavar='xvectorEnrollment', type=str,
                        help='the path to the the file where the xvector enrollment are stocked')
    parser.add_argument('xvectorTest', metavar='xvectorTest', type=str,
                        help='the path to the the file where the xvector test are stocked')


    args = parser.parse_args()
    trials = read_keys(args.keys)
    enrollment = read_pkl(args.xvectorEnrollment)
    test = read_pkl(args.xvectorTest)

    scoring_xvector(trials, enrollment, test)


