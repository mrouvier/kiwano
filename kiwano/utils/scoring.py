import sklearn.metrics
import torch


def read_xvector(file_path: str):
    h = {}
    f = open(file_path, "r")
    for line in f:
        line = line.strip().split(" ")
        arr = line[1:]
        my_list = [float(i) for i in arr]
        h[line[0]] = torch.Tensor(my_list)
    f.close
    return h


def read_keys(file_path: str):
    h = {}
    f = open(file_path, "r")
    for line in f:
        line = line.strip().split(" ")

        enrollmentName = line[0]
        testName = line[1]
        label = line[2]

        if label == "target":
            label = "1"
        else:
            label = "0"

        """
        if file_path.split("/")[-1] == "voxceleb1_test_v2.txt" or file_path.split("/")[-1] == "list_test_hard2.txt" or file_path.split("/")[-1] == "list_test_all2.txt":
            enrollmentName = enrollmentName.replace("/", "_")[:-4]
            testName = testName.replace("/", "_").replace('\n', "")[:-4]
        """
        h[(enrollmentName, testName)] = label
    f.close
    return h


def read_scores(file_path: str):
    h = {}
    f = open(file_path, "r")
    for line in f:
        line = line.strip().split(" ")
        enrollmentName = line[0]
        testName = line[1]
        score = line[2]
        h[(enrollmentName, testName)] = score
    f.close
    return h


def compute_fpr_fnr_threshold(keys, scores):
    """
    arg1 keys: dictionary with key : tuple with the names of the pairs of audio files, value : labels (0 : not the same speaker, 1 : same speaker)
    arg2 scores: dictionary with key : tuple with the names of the pairs of audio files, value : the similarity between their two xvectors
    This function calculates and returns the fpr and the fnr
    A part of this code is taken from https://github.com/YuanGongND/python-compute-eer
    """

    labels = []
    distances = []

    for x in keys:
        labels.append(int(keys[x]))
        distances.append(float(scores[x]))

    """
    for label in keys.values():
        labels.append(int(label))

    for distance in scores.values():
        distances.append(float(distance))
    """

    positive_label = 1
    # all fpr, tpr, fnr, fnr, threshold are lists (in the format of np.array)
    fpr, tpr, threshold = sklearn.metrics.roc_curve(
        labels, distances, pos_label=positive_label
    )
    fnr = 1 - tpr

    return fpr, fnr, threshold
