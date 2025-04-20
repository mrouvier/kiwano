import sys

import torch

from kiwano.embedding import EmbeddingSet, read_pkl, write_pkl

if __name__ == "__main__":
    tmp = EmbeddingSet()
    arr = read_pkl(sys.argv[1])
    for v in arr:
        tmp[v] = arr[v] / arr[v].norm(2)
    write_pkl(sys.argv[2], tmp)
