from kiwano.embedding import EmbeddingSet, write_pkl, read_pkl
import sys

if __name__ == '__main__':
    arr = read_pkl( sys.argv[1] )
    write_pkl( sys.argv[2], arr )

