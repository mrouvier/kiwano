#!/usr/bin/python3

import sys
from kiwano.utils import Pathlike
from pathlib import Path
import torchaudio
from tqdm import tqdm

import argparse

def get_duration(file_path: str):
   info = torchaudio.info(file_path)
   return info.num_frames/info.sample_rate


def prepare_rirs_noises(in_data: Pathlike, out_data: Pathlike):
    in_data = Path(in_data)

    out_data = Path(out_data)
    out_data.mkdir(parents=True, exist_ok=True)

    liste = open(out_data / "liste", "w")

    for segment in Path(in_data).rglob("*.wav"):
        duration = str(round(float(get_duration(segment)),2))
        name = str(segment).split("/")[-1].split(".")[0]
        room = str(segment).split("/")[-3]
        if room == "mediumroom" or room == "smallroom":
            liste.write(f"{room}_{name} {room} {duration} {segment}\n")

    liste.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('in_data', metavar='in_data', type=str,
                        help='the path to the target directory where the wav files are stored')
    parser.add_argument('out_data', metavar="out_data", type=str,
                        help='the path to the target directory where the liste will be stored')

    args = parser.parse_args()

    prepare_rirs_noises(args.in_data, args.out_data)
