#!/usr/bin/env python3

import argparse
import glob
import os
import sys
from concurrent.futures import as_completed
from concurrent.futures.process import ProcessPoolExecutor
from pathlib import Path
from subprocess import PIPE, run

import numpy as np
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


def ffmpeg(file_path: Pathlike, sampling_frequency: int):
    cmd = (
        "ffmpeg -y -threads 1 -i "
        + str(file_path)
        + " -acodec pcm_s16le -ac 1 -ar "
        + str(sampling_frequency)
        + " -ab 48 -threads 1 -f wav -"
    )
    proc = run(cmd, shell=True, stdout=PIPE, stderr=PIPE)
    raw_audio = proc.stdout
    audio_array = np.frombuffer(raw_audio, dtype=np.int16)
    wav = torch.tensor(audio_array.astype(np.float32) / 32768.0)

    return wav


def process_file(
    segment: Pathlike, out_data: Pathlike, sampling_frequency: int, vad: bool
):
    name = "_".join(str(segment).split("/")[-3:]).split(".")[0]
    spkid = str(segment).split("/")[-3]
    emission = str(segment).split("/")[-2]
    n = str(segment).split("/")[-1].split(".")[0]

    out = Path(out_data / "wav" / spkid / emission)
    out.mkdir(parents=True, exist_ok=True)

    output = str(Path(out_data / "wav" / spkid / emission / n)) + ".wav"

    wav = ffmpeg(segment, sampling_frequency)

    if vad:
        speech_timestamps = get_speech_timestamps(
            wav, model, sampling_rate=sampling_frequency, threshold=0.6
        )
        if len(speech_timestamps) > 0:
            wav = collect_chunks(speech_timestamps, wav)

    sf.write(str(output), wav, sampling_frequency)

    duration = round(wav.shape[0] / sampling_frequency, 2)

    return name, spkid, duration, output


def prepare_voxceleb2(
    in_data: Pathlike = ".",
    out_data: Pathlike = ".",
    sampling_frequency: int = 16000,
    delete_zip: bool = False,
    num_jobs: int = 30,
    vad: bool = False,
):
    out_data = Path(out_data)
    out_data.mkdir(parents=True, exist_ok=True)

    liste = open(out_data / "liste", "w")

    wav_lst = get_all_files(in_data / "dev" / "aac", match_and=[".m4a"])

    with ProcessPoolExecutor(num_jobs) as ex:
        futures = []

        for segment in wav_lst:
            futures.append(
                ex.submit(process_file, segment, out_data, sampling_frequency, vad)
            )

        for future in tqdm(
            futures, total=len(futures), desc=f"Processing VoxCeleb2..."
        ):
            name, spkid, duration, segment = future.result()
            liste.write(f"{name} {spkid} {duration} {segment}\n")

    liste.close()

    if delete_zip:
        for file in sorted(in_data.glob("vox2_dev_aac_part*")):
            os.remove(file)
        os.remove(in_data / "vox2_aac.zip")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "in_data",
        metavar="in_data",
        type=str,
        help='Path to the directory where the directory "dev" is stored',
    )
    parser.add_argument(
        "out_data",
        metavar="out_data",
        type=str,
        help="Path to the target directory where the liste will be stored",
    )
    parser.add_argument(
        "--num_jobs", type=int, default=30, help="Number of parallel jobs (default: 30)"
    )
    parser.add_argument(
        "--vad", action="store_true", default=False, help="Apply VAD (default: False)"
    )
    parser.add_argument(
        "--downsampling",
        type=int,
        default=16000,
        help="Downsampling frequency value (default: 16000)",
    )
    parser.add_argument(
        "--delete_zip",
        action="store_true",
        default=False,
        help="Delete the already extracted ZIP files (default: False)",
    )

    args = parser.parse_args()

    prepare_voxceleb2(
        Path(args.in_data),
        Path(args.out_data),
        args.downsampling,
        args.delete_zip,
        args.num_jobs,
        args.vad,
    )
