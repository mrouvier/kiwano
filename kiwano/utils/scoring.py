import sklearn.metrics
import torch


def read_xvector(file_path: str):
    """
    Read a text-based x-vector file into a Python dictionary.

    Each line is expected to have the format:
        <utt_id> <dim1> <dim2> ... <dimN>

    where `<utt_id>` is a string (utterance ID) and the rest are floats forming
    an embedding vector. The function converts each embedding to a
    `torch.Tensor`.

    Parameters
    ----------
    file_path : str
        Path to the text file containing embeddings.

    Returns
    -------
    dict[str, torch.Tensor]
        Mapping `{utt_id -> embedding_tensor}`.

    Examples
    --------
    Example content of `xvector.txt`:
    ```
    spk1_utt1 0.12 0.33 -0.91
    spk2_utt1 0.09 -0.05 0.88
    ```

    >>> h = read_xvector("xvector.txt")
    >>> list(h.keys())[:1]
    ['spk1_utt1']
    >>> h['spk1_utt1'].shape
    torch.Size([3])
    """
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
    """
    Parse a key file defining trial pairs and their labels (target/non-target).

    Each line is expected to have the format:
        <enrollment_utt> <test_utt> <label>

    where `<label>` is either `'target'` or `'nontarget'`. The function converts
    them to binary strings `"1"` (same speaker) and `"0"` (different speaker).

    Parameters
    ----------
    file_path : str
        Path to the trial key list (e.g., `voxceleb1_test_v2.txt`).

    Returns
    -------
    dict[tuple[str, str], str]
        Mapping `{(enrollment_utt, test_utt) -> label}`, where label is `'1'` or `'0'`.

    Examples
    --------
    Example content of `keys.txt`:
    ```
    spk1_utt1 spk1_utt2 target
    spk2_utt1 spk3_utt4 nontarget
    ```

    >>> keys = read_keys("keys.txt")
    >>> list(keys.items())[0]
    (('spk1_utt1', 'spk1_utt2'), '1')
    >>> list(keys.items())[1][1]
    '0'
    """
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
    """
    Read a scoring file mapping enrollment-test pairs to similarity scores.

    Each line is expected to have the format:
        <enrollment_utt> <test_utt> <score>

    where `<score>` is a float (cosine similarity, PLDA, etc.).

    Parameters
    ----------
    file_path : str
        Path to the score file.

    Returns
    -------
    dict[tuple[str, str], str]
        Mapping `{(enrollment_utt, test_utt) -> score}`.

    Examples
    --------
    Example content of `scores.txt`:
    ```
    spk1_utt1 spk1_utt2 0.913
    spk2_utt1 spk3_utt4 -0.087
    ```

    >>> scores = read_scores("scores.txt")
    >>> list(scores.values())[:2]
    ['0.913', '-0.087']
    """
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
    Compute FPR, FNR, and threshold arrays for verification evaluation.

    Given a trial list (`keys`) and similarity scores (`scores`), this function
    aligns them by pair keys and computes the ROC curve via
    `sklearn.metrics.roc_curve`. It outputs the corresponding **false positive
    rates (FPR)**, **false negative rates (FNR)**, and thresholds.

    Intended for fine-grained analysis or computing the EER threshold.

    A part of this code is taken from https://github.com/YuanGongND/python-compute-eer

    Parameters
    ----------
    keys : dict[tuple[str, str], str]
        Mapping `{(enroll, test) -> label}`, where label is `'1'` (same speaker)
        or `'0'` (different speaker).
    scores : dict[tuple[str, str], str]
        Mapping `{(enroll, test) -> score}` (float or string-cast float).

    Returns
    -------
    fpr : numpy.ndarray
        False positive rate for each threshold.
    fnr : numpy.ndarray
        False negative rate for each threshold.
    threshold : numpy.ndarray
        Score thresholds corresponding to each (fpr, fnr) pair.

    Examples
    --------
    >>> keys = {('e1','t1'):'1', ('e2','t2'):'0'}
    >>> scores = {('e1','t1'): '0.9', ('e2','t2'): '0.1'}
    >>> fpr, fnr, thr = compute_fpr_fnr_threshold(keys, scores)
    >>> round(fpr[0], 2), round(fnr[-1], 2)
    (0.0, 0.0)
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
