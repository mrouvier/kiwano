#!/usr/bin/env python3
import sys
import argparse
from kiwano.embedding import open_input_reader, open_output_writer, load_embeddings


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

    mean_spec = args.arg1
    input_spec = args.arg2
    output_spec = args.arg3

    mean = load_embeddings(mean_spec)
    reader, proc = open_input_reader(input_spec)
    writer = open_output_writer(output_spec)

    try:
        for utt_id, emb in reader:
            writer.write(utt_id, emb-mean["global-all"])

    finally:
        writer.close()
        reader.close()
        if proc is not None:
            proc.wait()



if __name__ == "__main__":
    main()

