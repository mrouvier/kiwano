#!/usr/bin/python3

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from subprocess import PIPE, run

import torchaudio
from tqdm import tqdm

from kiwano.utils import Pathlike, get_all_files


def get_duration(file_path: str):
    info = torchaudio.info(file_path)
    return info.num_frames / info.sample_rate


def process_file(segment: Pathlike, sampling_frequency: int, out_data: Pathlike):

    new_segment = segment

    if sampling_frequency != 16000:
        out = Path(
            out_data
            / str(segment).split("/")[-5]
            / str(segment).split("/")[-4]
            / str(segment).split("/")[-3]
            / str(segment).split("/")[-2]
        )
        out.mkdir(parents=True, exist_ok=True)

        output = Path(
            out_data
            / str(segment).split("/")[-5]
            / str(segment).split("/")[-4]
            / str(segment).split("/")[-3]
            / str(segment).split("/")[-2]
            / str(segment).split("/")[-1]
        )
        cmd = (
            "ffmpeg -y -threads 1 -i "
            + str(segment)
            + " -acodec pcm_s16le -ac 1 -ar "
            + str(sampling_frequency)
            + " -ab 48 -threads 1 "
            + str(output)
        )
        proc = run(cmd, shell=True, stdout=PIPE, stderr=PIPE)
        new_segment = output

    duration = str(round(float(get_duration(str(new_segment))), 2))
    name = str(segment).split("/")[-1].split(".")[0]
    spkid = str(segment).split("/")[-3]

    return spkid + "_" + name, spkid, duration, new_segment


def prepare_rirs_noises(
    in_data: Pathlike, out_data: Pathlike, sampling_frequency: int, num_jobs: int
):
    in_data = Path(in_data)

    out_data = Path(out_data)
    out_data.mkdir(parents=True, exist_ok=True)

    liste = open(out_data / "liste", "w")

    wav_lst = get_all_files(
        in_data, match_and=[".wav"], match_or=["mediumroom", "smallroom"]
    )

    with ProcessPoolExecutor(num_jobs) as ex:
        futures = []

        for segment in wav_lst:
            room = str(segment).split("/")[-3]
            if room == "mediumroom" or room == "smallroom":
                futures.append(
                    ex.submit(process_file, segment, sampling_frequency, out_data)
                )

        for future in tqdm(futures, total=len(futures), desc=f"Processing MUSAN..."):
            name, spkid, duration, segment = future.result()
            liste.write(f"{name} {spkid} {duration} {segment}\n")

    liste.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "in_data",
        metavar="in_data",
        type=str,
        help="the path to the target directory where the wav files are stored",
    )
    parser.add_argument(
        "out_data",
        metavar="out_data",
        type=str,
        help="the path to the target directory where the liste will be stored",
    )
    parser.add_argument(
        "--thread", type=int, default=20, help="Number of parallel jobs (default: 20)"
    )
    parser.add_argument(
        "--downsampling",
        type=int,
        default=16000,
        help="the value of sampling frequency (default: 16000)",
    )

    args = parser.parse_args()

    prepare_rirs_noises(args.in_data, args.out_data, args.downsampling, args.thread)
