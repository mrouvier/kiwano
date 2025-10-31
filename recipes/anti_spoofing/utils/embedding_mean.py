import sys

import torch

from kiwano.embedding import EmbeddingSet, read_pkl, write_pkl

if __name__ == "__main__":
    model = {}
    with open(sys.argv[1], "r") as f:
        for line in f:
            line = line.strip()
            line = line.split(" ")
            if line[1] not in model:
                model[line[1]] = []
            model[line[1]].append(line[0])

    tmp = EmbeddingSet()
    arr = read_pkl(sys.argv[2])

    for m in model:
        for k in model[m]:
            if m not in tmp:
                tmp[m] = arr[k] / len(model[m])
            else:
                tmp[m] = tmp[m] + arr[k] / len(model[m])

    write_pkl(sys.argv[3], tmp)
