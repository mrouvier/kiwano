#!/usr/bin/env python3
import argparse
import sys

import numpy as np

from kiwano.embedding import open_input_reader


def main():
    parser = argparse.ArgumentParser(
        description="Check that all embeddings have the same (or expected) dimension."
    )
    parser.add_argument(
        "input_spec",
        help="Input PKL spec (e.g. 'pkl:emb.pkl', 'pkl:-', 'pkl,t:emb.txt')",
    )
    parser.add_argument(
        "--expected-dim",
        type=int,
        default=None,
        help="If set, fail if any embedding has a different dimension.",
    )
    parser.add_argument(
        "--max-utt",
        type=int,
        default=None,
        help="Optional maximum number of embeddings to check.",
    )

    args = parser.parse_args()

    reader, proc = open_input_reader(args.input_spec)

    dims = set()
    n_vecs = 0
    status_ok = True

    try:
        for utt_id, emb in reader:
            emb = np.asarray(emb)
            dim = emb.shape[0]
            dims.add(dim)
            n_vecs += 1

            if args.expected_dim is not None and dim != args.expected_dim:
                status_ok = False
                print(
                    f"[ERROR] utt_id={utt_id}: dim={dim} != expected_dim={args.expected_dim}",
                    file=sys.stderr,
                )

            if args.max_utt is not None and n_vecs >= args.max_utt:
                break
    finally:
        reader.close()
        if proc:
            proc.wait()

    if n_vecs == 0:
        print("No embeddings found.", file=sys.stderr)
        sys.exit(1)

    dims_list = sorted(dims)
    print("----- Dimension check -----")
    print(f"#utterances   : {n_vecs}")
    print(f"dimensions    : {dims_list}")

    if args.expected_dim is not None:
        print(f"expected_dim  : {args.expected_dim}")
        print(f"status        : {'OK' if status_ok else 'FAILED'}")
        sys.exit(0 if status_ok else 1)
    else:
        # Just informational
        sys.exit(0)


if __name__ == "__main__":
    main()
