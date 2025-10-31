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


def get_duration(file_path: str):
    info = torchaudio.info(file_path)
    return info.num_frames / info.sample_rate


def process_file(segment: Pathlike, out_data: Pathlike, vad: bool):
    new_segment = segment
    duration = 0.0

    if vad:
        out = Path(out_data / str(segment).split("/")[-3] / str(segment).split("/")[-2])
        out.mkdir(parents=True, exist_ok=True)

        wav, sr = torchaudio.load(str(segment))
        wav = wav[0]

        output = (
            str(
                Path(
                    out_data
                    / str(segment).split("/")[-3]
                    / str(segment).split("/")[-2]
                    / str(segment).split("/")[-1].split(".")[0]
                )
            )
            + ".wav"
        )

        speech_timestamps = get_speech_timestamps(
            wav, model, sampling_rate=sr, threshold=0.7
        )

        if len(speech_timestamps) > 0:
            wav = collect_chunks(speech_timestamps, wav)

        sf.write(str(output), wav, sr)

        new_segment = output
        duration = wav.shape[0] / sr
    else:
        duration = str(round(float(get_duration(str(segment))), 2))

    name = "_".join(str(segment).split("/")[-2:]).split(".")[0]
    spkid = str(segment).split("/")[-2]

    return name, spkid, duration, new_segment


def prepare_cts_superset(
    in_data: Pathlike = ".",
    out_data: Pathlike = ".",
    num_jobs: int = 20,
    vad: bool = False,
):

    out_data = Path(out_data)
    out_data.mkdir(parents=True, exist_ok=True)

    nameListe = "liste"

    liste = open(out_data / nameListe, "w")

    wav_lst = get_all_files(in_data / "data", match_and=[".sph"])

    print(len(wav_lst))
    print(vad)

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
