#!/usr/bin/python3

import sys
from kiwano.utils import Pathlike, get_all_files
from pathlib import Path
import torchaudio
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

from subprocess import PIPE, run

import argparse
import os

def get_duration(file_path: str):
   info = torchaudio.info(file_path)
   return info.num_frames/info.sample_rate

def process_file(segment: Pathlike, spkid: str, in_data: Pathlike):
    name = str(segment).split("/")[-1]
    duration = str(round(float(get_duration(segment)),2))


    return name, spkid, duration, segment



def prepare_sre20_enrollment(in_data: Pathlike, out_data: Pathlike, jobs: int):
    in_data = Path(in_data)

    out_data = Path(out_data)
    out_data.mkdir(parents=True, exist_ok=True)

    model = {}
    with open(in_data / "docs" / "2020_cts_challenge_enrollment.tsv", "r") as f:
        for line in f:
            line = line.rstrip()
            line = line.split("\t")
            model[ line[1] ] = line[0]


    nameListe = "liste"

    liste = open(out_data / nameListe, "w")

    wav_lst = get_all_files(in_data / "data" / "enrollment", match_and=[".sph"])

    with ProcessPoolExecutor(jobs) as ex:
        futures = []

        for segment in wav_lst:
            if str(segment).split("/")[-1] in model:
                futures.append(ex.submit(process_file, segment, model[ str(segment).split("/")[-1] ], in_data))

        for future in tqdm(futures, desc="Processing SRE20 Enrollment"):
            name, spkid, duration, segment = future.result()
            liste.write(f"{name} {spkid} {duration} {segment}\n")

    liste.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--thread', type=int, default=10,
                    help='Number of parallel jobs (default: 10)')
    parser.add_argument('in_data', type=str,
                    help='Path to the directory containing the "wav" directory')
    parser.add_argument('out_data', type=str,
                    help='Path to the target directory where the list will be stored')

    args = parser.parse_args()

    prepare_sre20_enrollment(args.in_data, args.out_data, args.thread)

