#!/usr/bin/env python3
import argparse
import sys

import torch

from kiwano.embedding import open_input_reader, open_output_writer


def length_normalize(vec: torch.Tensor) -> torch.Tensor:
    vec = vec.to(torch.float32)
    norm = torch.linalg.norm(vec)

    if norm == 0:
        return vec

    return vec / norm


def main():
    parser = argparse.ArgumentParser(
        description="Apply L2 length normalization on embeddings"
    )
    parser.add_argument(
        "input_spec", help="pkl:input.pkl or pkl:- or command ending with |"
    )
    parser.add_argument("output_spec", help="pkl:output.pkl or pkl:-")

    args = parser.parse_args()

    reader, proc = open_input_reader(args.input_spec)
    writer = open_output_writer(args.output_spec)

    try:
        for utt_id, emb in reader:
            writer.write(utt_id, length_normalize(emb))
    finally:
        writer.close()
        reader.close()
        if proc is not None:
            proc.wait()


if __name__ == "__main__":
    main()
