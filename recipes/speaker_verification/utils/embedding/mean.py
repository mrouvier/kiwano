#!/usr/bin/env python3
import sys
import argparse
import torch
from kiwano.embedding import open_input_reader, open_output_writer

def compute_mean(emb_dict):
    arr = np.stack(list(emb_dict.values()))
    return arr.mean(axis=0)


def compute_mean(emb_dict: dict) -> torch.Tensor:
    if not emb_dict:
        raise RuntimeError("No embeddings to average")

    tensors = [emb.to(torch.float32) for emb in emb_dict.values()]
    arr = torch.stack(tensors, dim=0)
    return arr.mean(dim=0)


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
    parser.add_argument("arg3", help="arg3", nargs="?")

    args = parser.parse_args()


    if args.arg3 is None:
        input_spec = args.arg1
        output_spec = args.arg2

        reader, proc = open_input_reader(input_spec)

        emb_dict = {}
        for utt, emb in reader:
            emb_dict[utt] = emb

        reader.close()
        if proc:
            proc.wait()

        # Compute mean
        mean_vec = compute_mean(emb_dict)

        writer = open_output_writer(output_spec)

        try:
            writer.write("global-all", mean_vec)
        finally:
            writer.close()

    else:
        spk2utt_path = args.arg1
        input_spec = args.arg2
        output_spec = args.arg3

        reader, proc = open_input_reader(input_spec)
        writer = open_output_writer(output_spec)

        emb_dict = {}
        for utt, emb in reader:
            emb_dict[utt] = emb

        model = {}
        with open(spk2utt_path, "r") as f:
            for line in f:
                line = line.strip()
                line = line.split(" ")
                if line[1] not in model:
                    model[line[1]] = []
                model[line[1]].append(line[0])


        tmp_dict = {}
        for m in model:
            for k in model[m]:
                if m not in tmp_dict:
                    tmp_dict[m] = emb_dict[k] / len(model[m])
                else:
                    tmp_dict[m] = tmp_dict[m] + emb_dict[k] / len(model[m])


        for u in tmp_dict:
            writer.write(u, tmp_dict[u])


        reader.close()
        writer.close()

        if proc:
            proc.wait()





if __name__ == "__main__":
    main()

