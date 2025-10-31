#!/usr/bin/env python3

import argparse
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from subprocess import PIPE, run

import torchaudio
from tqdm.auto import tqdm

from kiwano.utils import Pathlike, get_all_files


def get_duration(file_path: str):
    info = torchaudio.info(file_path)
    return info.num_frames / info.sample_rate


def read_csv(file_path: str):
    h = {}
    counter = 0
    with open(file_path, "r") as infile:
        for line in infile:
            columns = line.strip().split(",")
            if counter > 0:
                if columns[2] == "bona-fide":
                    columns[2] = "bonafide"
                h[columns[0].split(".")[0]] = columns[2]
            counter += 1
    return h


def process_file(segment: Pathlike, out_data: Pathlike):
    name = str(segment).split("/")[-1].split(".")[0]
    output = str(out_data) + "/wav/" + "/" + name + ".wav"

    o = str(out_data) + "/wav/"
    Path(o).mkdir(parents=True, exist_ok=True)

    cmd = (
        "ffmpeg -y -threads 1 -i "
        + str(segment)
        + " -acodec pcm_s16le -ac 1 -ar 16000 -ab 48 -threads 1 "
        + str(output)
    )

    proc = run(cmd, shell=True, stdout=PIPE, stderr=PIPE)

    duration = str(round(float(get_duration(output)), 2))

    return name, duration, output


def prepare_itw(in_data: Pathlike = ".", out_data: Pathlike = ".", num_jobs: int = 20):
    out_data = Path(out_data)
    out_data.mkdir(parents=True, exist_ok=True)

    h = read_csv(in_data / "meta.csv")

    liste = open(out_data / "liste", "w")

    # LA Part
    wav_lst = get_all_files(in_data, match_and=[".wav"])

    print(len(wav_lst))

    with ProcessPoolExecutor(num_jobs) as ex:
        futures = []

        for segment in wav_lst:
            futures.append(ex.submit(process_file, segment, out_data))

        for future in tqdm(futures, total=len(futures), desc="Processing ITW..."):
            name, duration, segment = future.result()
            liste.write(f"{name} {h[name]} {duration} {segment}\n")

    liste.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "in_data",
        metavar="in_data",
        type=str,
        help='the path to the directory where the directory "dev" is stored',
    )
    parser.add_argument(
        "out_data",
        metavar="out_data",
        type=str,
        help="the path to the target directory where the liste will be stored",
    )

    args = parser.parse_args()

    prepare_itw(Path(args.in_data), Path(args.out_data), 30)
