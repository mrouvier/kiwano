#!/usr/bin/env python3

import argparse
import glob
import os
import random
import sys
from concurrent.futures import as_completed
from concurrent.futures.process import ProcessPoolExecutor
from pathlib import Path
from subprocess import PIPE, run

import numpy as np
import soundfile as sf
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


def process_file(
    segment: Pathlike, in_data: Pathlike, sampling_frequency: int, vad: bool
):
    name = "_".join(str(segment).split("/")[-3:]).split(".")[0]
    spkid = str(segment).split("/")[-3]
    emission = str(segment).split("/")[-2]
    n = str(segment).split("/")[-1].split(".")[0]

    nameDir = "wav"

    out = Path(in_data / nameDir / spkid / emission)
    out.mkdir(parents=True, exist_ok=True)

    output = str(Path(in_data / nameDir / spkid / emission / n)) + ".wav"

    audio = []

    if not Path(output).exists():
        audio = _process_file(segment, Path(output), sampling_frequency, vad)
        print(audio)

    """
    if vad:
        speech_timestamps = get_speech_timestamps(audio, model, sampling_rate=sampling_frequency, threshold=0.7)
        if len(speech_timestamps) > 0:
            wav = collect_chunks(speech_timestamps, audio)
    """

    sf.write(str(output), audio, sampling_frequency)

    duration = str(round(float(get_duration(output)), 2))

    toolkitPath = Path("db") / "voxceleb2" / "wav" / spkid / emission / (n + ".wav")

    return name, spkid, duration, toolkitPath


def prepare_voxceleb2(
    sampling_frequency: int,
    canDeleteZIP: bool,
    in_data: Pathlike = ".",
    out_data: Pathlike = ".",
    vad: bool = False,
    num_jobs: int = 20,
):

    print(in_data)

    out_data = Path(out_data)
    out_data.mkdir(parents=True, exist_ok=True)

    nameListe = "liste"

    liste = open(out_data / nameListe, "w")

    if str(in_data) == "/gpfsdswork/dataset/VoxCeleb2":
        wav_lst = get_all_files(in_data / "dev" / "aac", match_and=[".wav"])
    else:
        wav_lst = get_all_files(in_data / "dev" / "aac", match_and=[".m4a"])

    print(len(wav_lst))

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

    if canDeleteZIP:
        for file in sorted(in_data.glob("vox2_dev_aac_part*")):
            os.remove(file)
        os.remove(in_data / "vox2_aac.zip")


def _process_file(
    file_path: Pathlike, output: Pathlike, sampling_frequency: int, vad: bool
):
    # ffmpeg -v 8 -i {segment} -f wav -acodec pcm_s16le - |
    # cmd = "ffmpeg -y -threads 1 -i "+str(file_path)+" -acodec pcm_s16le -ac 1 -ar "+str(sampling_frequency)+" -ab 48 -threads 1 "+str(output)
    # proc = run(cmd, shell=True, stdout=PIPE, stderr=PIPE)

    nombre = random.randint(0, 1)

    if nombre == 0:
        codec = "gsm"
    else:
        codec = "amr-nb"

    if sampling_frequency == 8000:
        cmd = "ffmpeg -threads 1 -i {file_path} -acodec pcm_s16le -ac 1 -ar {sampling_frequency} -ab 48 -threads 1 -f wav - | sox -t wav - -r 8000 -t {codec} - | sox -t {codec} - -t wav -e signed-integer -b 16 -"
        print(cmd)
        proc = run(cmd, shell=True, stdout=PIPE, stderr=PIPE)
        raw_audio = proc.stdout
        audio = np.frombuffer(raw_audio, dtype=np.float32)
        return audio
    else:
        cmd = "ffmpeg -threads 1 -i {file_path} -acodec pcm_s16le -ac 1 -ar {sampling_frequency} -ab 48 -threads 1 -f wav -"
        proc = run(cmd, shell=True, stdout=PIPE, stderr=PIPE)
        raw_audio = proc.stdout
        audio = np.frombuffer(raw_audio, dtype=np.float32)
        return audio

    # audio = np.frombuffer(raw_audio, dtype=np.float32)


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
    parser.add_argument(
        "--downsampling",
        type=int,
        default=16000,
        help="the value of sampling frequency (default: 16000)",
    )
    parser.add_argument(
        "--vad",
        action="store_true",
        default=False,
        help="to delete the ZIP files already extracted (default: False)",
    )
    parser.add_argument(
        "--deleteZIP",
        action="store_true",
        default=False,
        help="to delete the ZIP files already extracted (default: False)",
    )

    args = parser.parse_args()

    prepare_voxceleb2(
        args.downsampling,
        args.deleteZIP,
        Path(args.in_data),
        Path(args.out_data),
        args.vad,
        20,
    )
