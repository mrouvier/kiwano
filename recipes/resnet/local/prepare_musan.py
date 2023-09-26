#!/usr/bin/python3

import sys
from kiwano.utils import Pathlike
from pathlib import Path
import librosa
from tqdm import tqdm

import argparse

def get_duration_librosa(file_path: str):
   audio_data, sample_rate = librosa.load(file_path)
   duration = librosa.get_duration(y=audio_data, sr=sample_rate)
   return duration


def prepare_musan(in_data: Pathlike, out_data: Pathlike):
    in_data = Path(in_data)

    out_data = Path(out_data)
    out_data.mkdir(parents=True, exist_ok=True)

    liste = open(out_data / "liste", "w")

    for segment in Path(in_data).rglob("*.wav"):
        duration = str(round(float(get_duration_librosa(segment)),2))
        name = str(segment).split("/")[-1].split(".")[0]
        spkid = str(segment).split("/")[-3]
        liste.write(f"{name} {spkid} {duration} {segment}\n")

    liste.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('in_data', metavar='in_data', type=str,
                        help='the path to the target directory where the wav files are stored')
    parser.add_argument('out_data', metavar="out_data", type=str,
                        help='the path to the target directory where the liste will be stored')

    args = parser.parse_args()

    prepare_musan(args.in_data, args.out_data)
