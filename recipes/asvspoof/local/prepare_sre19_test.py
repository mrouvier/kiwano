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

def process_file(segment: Pathlike, in_data: Pathlike):
    name = str(segment).split("/")[-1]
    spkid = name
    duration = str(round(float(get_duration(segment)),2))

    return name, spkid, duration, segment



def prepare_sre19_test(in_data: Pathlike, out_data: Pathlike, jobs: int):
    in_data = Path(in_data)

    out_data = Path(out_data)
    out_data.mkdir(parents=True, exist_ok=True)

    nameListe = "liste"

    liste = open(out_data / nameListe, "w")

    wav_lst = get_all_files(in_data / "data" / "test", match_and=[".sph"])

    with ProcessPoolExecutor(jobs) as ex:
        futures = []

        for segment in wav_lst:
            futures.append(ex.submit(process_file, segment, in_data))

        for future in tqdm(futures, desc="Processing SRE19 Test"):
            name, spkid, duration, segment = future.result()
            liste.write(f"{name} {spkid} {duration} {segment}\n")

    liste.close()


    trials = open(out_data / "trials", "w")
    counter = 0

    with open(in_data / "docs" / "sre19_cts_challenge_trial_key.tsv", "r") as f:
        for line in f:
            if counter > 0:
                line = line.rstrip()
                line = line.split("\t")
                enrollment = line[0]
                test = line[1]
                key = line[3]
                trials.write(f"{enrollment} {test} {key}\n")
            counter += 1

    trials.close()



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--thread', type=int, default=10,
                    help='Number of parallel jobs (default: 10)')
    parser.add_argument('in_data', type=str,
                    help='Path to the directory containing the "wav" directory')
    parser.add_argument('out_data', type=str,
                    help='Path to the target directory where the list will be stored')

    args = parser.parse_args()

    prepare_sre19_test(args.in_data, args.out_data, args.thread)

