#!/usr/bin/env python3
import argparse
import sys

import torch

from kiwano.embedding import open_input_reader


def main():
    parser = argparse.ArgumentParser(
        description="Compute statistics on embeddings (dims, norms, values)."
    )
    parser.add_argument(
        "input_spec",
        help="Input PKL spec (e.g. 'pkl:emb.pkl', 'pkl:-', 'pkl,t:emb.txt', 'pkl:python ... |')",
    )
    parser.add_argument(
        "--max-utt",
        type=int,
        default=None,
        help="Optional maximum number of embeddings to consider.",
    )

    args = parser.parse_args()

    reader, proc = open_input_reader(args.input_spec)

    dims = set()
    n_vecs = 0

    # On stocke juste pour stats globales (sans tout garder en mémoire si possible)
    norms = []
    sum_vals = 0.0
    sum_sq_vals = 0.0
    total_count_vals = 0
    global_min = None
    global_max = None

    try:
        for utt_id, emb in reader:
            emb = emb.to(torch.float64)

            n_vecs += 1

            # dimensions
            dims.add(emb.shape[0])

            # norme
            norm = torch.linalg.norm(emb).item()
            norms.append(norm)

            # valeur min/max
            vmin = torch.min(emb).item()
            vmax = torch.max(emb).item()

            global_min = vmin if global_min is None else min(global_min, vmin)
            global_max = vmax if global_max is None else max(global_max, vmax)

            # somme / somme des carrés
            s = torch.sum(emb).item()
            ss = torch.sum(emb * emb).item()

            sum_vals += s
            sum_sq_vals += ss
            total_count_vals += emb.numel()

            if args.max_utt is not None and n_vecs >= args.max_utt:
                break
    finally:
        reader.close()
        if proc:
            proc.wait()

    if n_vecs == 0:
        print("No embeddings found.", file=sys.stderr)
        sys.exit(1)

    # Convert norms to tensor for statistics
    norms_t = torch.tensor(norms, dtype=torch.float64)

    # stats sur normes
    norm_min = torch.min(norms_t).item()
    norm_max = torch.max(norms_t).item()
    norm_mean = torch.mean(norms_t).item()
    norm_std = torch.std(norms_t, unbiased=False).item()  # np.std equivalent

    # stats sur valeurs globales
    mean_val = sum_vals / total_count_vals
    var_val = sum_sq_vals / total_count_vals - mean_val * mean_val
    var_val = max(var_val, 0.0)  # éviter -0.0
    std_val = var_val**0.5

    print("----- Embedding statistics -----")
    print(f"#utterances       : {n_vecs}")
    print(f"dimensions        : {sorted(dims)}")
    print("")
    print("Norm statistics:")
    print(f"  min norm        : {norm_min:.6f}")
    print(f"  max norm        : {norm_max:.6f}")
    print(f"  mean norm       : {norm_mean:.6f}")
    print(f"  std norm        : {norm_std:.6f}")
    print("")
    print("Value statistics:")
    print(f"  global min      : {global_min:.6f}")
    print(f"  global max      : {global_max:.6f}")
    print(f"  mean value      : {mean_val:.6f}")
    print(f"  std value       : {std_val:.6f}")


if __name__ == "__main__":
    main()
