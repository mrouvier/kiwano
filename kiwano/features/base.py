import numpy as np


class CMVN:
    """
    Apply cepstral mean and (optionnaly) variance normalization

    arr = [ [1, 2, 3], [3, 4, 5], [5, 6, 7] ]
    CMVN(arr)

    """

    def __new__(self, arr: np.ndarray):
        return arr - np.mean(arr, axis=0)
