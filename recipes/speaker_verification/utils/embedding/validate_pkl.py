#!/usr/bin/env python3
import argparse
import sys

import torch

from kiwano.embedding import open_input_reader


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Validate an embeddings PKL spec (dimension consistency, "
            "NaN/Inf checks, duplicate utt_ids, etc.)."
        )
    )
    parser.add_argument(
        "input_spec",
        help="Input PKL spec (e.g. 'pkl:emb.pkl', 'pkl:-', 'pkl,t:emb.txt', 'pkl:python ... |')",
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
        help="Optional max number of utterances to read (for quick checks).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only exit code is meaningful, suppress detailed messages.",
    )

    args = parser.parse_args()

    reader, proc = open_input_reader(args.input_spec)

    seen_ids = set()
    dims = set()
    n_vecs = 0
    n_nan = 0
    n_inf = 0
    n_zero_dim = 0
    n_duplicates = 0

    status_ok = True

    try:
        for utt_id, emb in reader:
            n_vecs += 1

            if not isinstance(emb, torch.Tensor):
                emb = torch.tensor(emb)
            emb = emb.to(torch.float32)

            # Duplicate IDs
            if utt_id in seen_ids:
                n_duplicates += 1
                status_ok = False
                if not args.quiet:
                    print(f"[ERROR] Duplicate utt_id: {utt_id}", file=sys.stderr)
            else:
                seen_ids.add(utt_id)

            # Dimension
            if emb.ndim != 1:
                if not args.quiet:
                    print(
                        f"[ERROR] utt_id={utt_id}: embedding ndim={emb.ndim} != 1",
                        file=sys.stderr,
                    )
                status_ok = False
            dim = emb.shape[0]
            dims.add(dim)
            if dim == 0:
                n_zero_dim += 1
                status_ok = False
                if not args.quiet:
                    print(
                        f"[ERROR] utt_id={utt_id}: zero dimension embedding",
                        file=sys.stderr,
                    )

            if args.expected_dim is not None and dim != args.expected_dim:
                status_ok = False
                if not args.quiet:
                    print(
                        f"[ERROR] utt_id={utt_id}: dim={dim} != expected_dim={args.expected_dim}",
                        file=sys.stderr,
                    )

            # NaN / Inf
            # torch.isfinite / isnan / isinf
            finite_mask = torch.isfinite(emb)
            if not torch.all(finite_mask):
                nan_count = int(torch.isnan(emb).sum().item())
                inf_count = int(torch.isinf(emb).sum().item())
                n_nan += nan_count
                n_inf += inf_count
                status_ok = False
                if not args.quiet:
                    print(
                        f"[ERROR] utt_id={utt_id}: contains NaN/Inf values "
                        f"(NaN={nan_count}, Inf={inf_count})",
                        file=sys.stderr,
                    )

            if args.max_utt is not None and n_vecs >= args.max_utt:
                break

    finally:
        reader.close()
        if proc:
            proc.wait()

    if not args.quiet:
        print("----- Validation summary -----", file=sys.stderr)
        print(f"#utterances   : {n_vecs}", file=sys.stderr)
        print(f"dimensions    : {sorted(dims)}", file=sys.stderr)
        print(f"duplicates    : {n_duplicates}", file=sys.stderr)
        print(f"zero-dim      : {n_zero_dim}", file=sys.stderr)
        print(f"NaN count     : {n_nan}", file=sys.stderr)
        print(f"Inf count     : {n_inf}", file=sys.stderr)
        print(f"status        : {'OK' if status_ok else 'FAILED'}", file=sys.stderr)

    sys.exit(0 if status_ok else 1)


if __name__ == "__main__":
    main()
