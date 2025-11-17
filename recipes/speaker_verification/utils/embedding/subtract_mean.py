#!/usr/bin/env python3
import sys
import argparse
from kiwano.embedding import open_input_reader, open_output_writer


def main():
    parser = argparse.ArgumentParser(
        description="Compute mean embeddings.\n\n"
            "Usage:\n"
            "  prog input_spec output_spec\n"
            "    -> global mean over all embeddings in input_spec, written as 'mean' in output_spec.\n\n"
            "  prog spk2utt input_spec output_spec\n"
            "    -> per-speaker means using spk2utt mapping, written with keys = speaker IDs."
    )
    parser.add_argument("arg1", help="arg1")
    parser.add_argument("arg2", help="arg2")
    parser.add_argument("arg3", help="arg3")

    args = parser.parse_args()


    spk2utt_path = args.arg1
    input_spec = args.arg2
    output_spec = args.arg3

    reader, proc = open_input_reader(input_spec)

    for utt, emb in reader:
        emb_dict[utt] = emb

    reader.close()
    if proc:
        proc.wait()



if __name__ == "__main__":
    main()

