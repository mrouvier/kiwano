from kiwano.embedding import EmbeddingSet, write_pkl, read_pkl
import sys
import torch

if __name__ == '__main__':
    tmp = EmbeddingSet()
    arr = read_pkl( sys.argv[1] )
    for v in arr:
        tmp[v] = arr[v] / arr[v].norm(2)
    write_pkl( sys.argv[2], tmp )

