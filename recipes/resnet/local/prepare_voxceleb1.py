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


def prepare_voxceleb1(in_data: Pathlike, out_data: Pathlike):
    in_data = Path(in_data)

    out_data = Path(out_data)
    out_data.mkdir(parents=True, exist_ok=True)

    liste = open(out_data / "liste", "w")

    with open(in_data / "vox1_meta.csv", "r") as f:
        next(f)
        for line in tqdm(f):
            spkid, name, gender, nationality, split = line.strip().split("\t")

            for segment in (in_data / "wav" / spkid).rglob("*.wav"):
                name = "_".join(str(segment).split("/")[-3:]).split(".")[0]
                duration = str(round(float(get_duration_librosa(segment)),2))
                liste.write(f"{name} {spkid} {duration} {segment}\n")

    liste.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('in_data', metavar='in_data', type=str,
                        help='the path to vox1_meta.csv')
    parser.add_argument('out_data', metavar="out_data", type=str,
                        help='the path to the target directory where the liste will be stored')

    args = parser.parse_args()

    prepare_voxceleb1(args.in_data, args.out_data)

