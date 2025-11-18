#!/usr/bin/env python3
import argparse
import sys

from kiwano.embedding import open_input_reader, open_output_writer


def main():
    parser = argparse.ArgumentParser(
        description="Copy speaker embeddings from an input PKL spec to an output PKL spec."
    )
    parser.add_argument(
        "input_spec",
        help="Input spec (e.g., 'pkl:file.pkl', 'pkl:-', 'pkl,t:file.txt', 'pkl:python normalize.py ... |')",
    )
    parser.add_argument(
        "output_spec",
        help="Output spec (e.g., 'pkl:file.pkl', 'pkl:-', 'pkl,t:file.txt', 'pkl,t:-')",
    )

    args = parser.parse_args()

    reader, proc = open_input_reader(args.input_spec)
    writer = open_output_writer(args.output_spec)

    try:
        for utt_id, emb in reader:
            writer.write(utt_id, emb)
    finally:
        writer.close()
        reader.close()
        if proc is not None:
            proc.wait()


if __name__ == "__main__":
    main()
