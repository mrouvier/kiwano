#!/usr/bin/python3

import sys
from kiwano.utils import Pathlike
from pathlib import Path
import librosa
from tqdm import tqdm


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

    if len(sys.argv) == 3:
        prepare_voxceleb1(sys.argv[1], sys.argv[2])
    else:
        print("Erreur, usage correct : prepare_voxceleb1.py in_data out_data ")

