#!/usr/bin/env python3

import argparse
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from subprocess import PIPE, run

import soundfile as sf
import torch
import torchaudio
from silero_vad import (
    collect_chunks,
    get_speech_timestamps,
    load_silero_vad,
    read_audio,
)
from tqdm import tqdm

from kiwano.utils import Pathlike, get_all_files

model = load_silero_vad()

"""
def process_file(segment: Pathlike, out_data: Pathlike, vad: bool):
    wav, sr = torchaudio.load(str(segment))

    speech_timestamps = get_speech_timestamps(
        wav, model, sampling_rate=sr, threshold=0.7
    )

    return name, speech_timestamps


def prepare_vad(in_data: Pathlike = "."):

    nameListe = "liste"

    liste = open(out_data / nameListe, "r")

    with ProcessPoolExecutor(num_jobs) as ex:
        futures = []

        for segment in wav_lst:
            futures.append(ex.submit(process_file, segment, out_data, vad))

        for future in tqdm(
            futures, total=len(futures), desc=f"Processing CTS Superset..."
        ):
            name, spkid, duration, segment = future.result()
            liste.write(f"{name} {spkid} {duration} {segment}\n")

    liste.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--thread", type=int, default=20, help="Number of parallel jobs (default: 20)"
    )
    parser.add_argument(
        "--vad", action="store_true", default=False, help="Apply VAD (default: False)"
    )
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

    prepare_cts_superset(Path(args.in_data), Path(args.out_data), args.thread, args.vad)
"""
