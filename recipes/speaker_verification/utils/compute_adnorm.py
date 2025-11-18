import torch

import argparse

from kiwano.utils import read_keys
from kiwano.embedding import load_embeddings


cosine = torch.nn.CosineSimilarity(dim=0)
mean_std_cache = {}


def compute_v(xvector, impostors):
    """Compute cosine similarity between xvector and all impostors."""
    return torch.stack([cosine(xvector, impostors[imp]) for imp in impostors])


def compute_score_adnorm(xvectorEnrollment, xvectorTest, enrollmentName, testName, impostors, k):
    """
    Compute S-Norm score between enrollment and test xvectors.
    mean_std_cache: dict mapping embedding name -> (mean, std) of impostor scores.
    """

    impostor_keys = list(impostors.keys())

    mean_std_ve = mean_std_cache.get(enrollmentName)
    if mean_std_ve is None:
        ve = compute_v(xvectorEnrollment, impostors)
        top_vals, top_idx = torch.topk(ve, k)
        mean_std_cache[enrollmentName] = torch.mean( torch.stack( [impostors[impostor_keys[i.item()]] for i in top_idx] ) )

    mean_std_vt = mean_std_cache.get(testName)
    if mean_std_vt is None:
        vt = compute_v(xvectorTest, impostors)
        top_vals, top_idx = torch.topk(vt, k)
        mean_std_cache[testName] = torch.mean( torch.stack( [impostors[impostor_keys[i.item()]] for i in top_idx] ) )

    score = cosine(xvectorEnrollment-mean_std_cache[enrollmentName], xvectorTest-mean_std_cache[testName])

    return score


def adnorm(trials, xvector_enrollment, xvector_test, xvector_impostor, k):

    for enrollment_name, test_name in trials:
        score = compute_score_adnorm(xvector_enrollment[enrollment_name], xvector_test[test_name], enrollment_name, test_name, xvector_impostor, k)
        print(f"{enrollment_name} {test_name} {score}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "keys",
        metavar="keys",
        type=str,
        help="the path to the file where the keys are stocked",
    )
    parser.add_argument(
        "xvector_enrollment",
        metavar="xvector_enrollment",
        type=str,
        help="command to gather xvectors enrollment in pkl format",
    )
    parser.add_argument(
        "xvector_test",
        metavar="xvector_test",
        type=str,
        help="command to gather xvectors test in pkl format",
    )
    parser.add_argument(
        "xvector_impostor",
        metavar="xvector_impostor",
        type=str,
        help="command to gather xvectors for the impostor set in pkl format ",
    )
    parser.add_argument(
        "k", metavar="k", type=int, help="numbers of impostors in the cohort"
    )


    args = parser.parse_args()

    trials = read_keys(args.keys)

    enrollment = load_embeddings(args.xvector_enrollment)
    test = load_embeddings(args.xvector_test)
    impostor = load_embeddings(args.xvector_impostor)

    adnorm(trials, enrollment, test, impostor, args.k)
