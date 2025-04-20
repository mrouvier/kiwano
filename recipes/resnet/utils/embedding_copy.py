import sys

from kiwano.embedding import EmbeddingSet, read_pkl, write_pkl

if __name__ == "__main__":
    arr = read_pkl(sys.argv[1])
    write_pkl(sys.argv[2], arr)
