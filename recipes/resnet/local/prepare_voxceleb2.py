#!/usr/bin/env python3

import sys
import glob
from kiwano.utils import Pathlike
from pathlib import Path
import torchaudio
import argparse
from concurrent.futures import as_completed
from concurrent.futures.process import ProcessPoolExecutor
from tqdm import tqdm
from subprocess import PIPE, run
import os


def get_duration(file_path: str):
    info = torchaudio.info(file_path)
    return info.num_frames / info.sample_rate


def process_file(segment: Pathlike, in_data: Pathlike, sampling_frequency: int):
    name = "_".join(str(segment).split("/")[-3:]).split(".")[0]
    spkid = str(segment).split("/")[-3]
    emission = str(segment).split("/")[-2]
    n = str(segment).split("/")[-1].split(".")[0]

    nameDir = "wav"
    if sampling_frequency != 16000:
        nameDir = nameDir + "_" + str(sampling_frequency)

    out = Path(in_data / nameDir / spkid / emission)
    out.mkdir(parents=True, exist_ok=True)

    output = str(Path(in_data / nameDir / spkid / emission / n)) + ".wav"

    if not Path(output).exists():
        _process_file(segment, Path(output), sampling_frequency)

    duration = str(round(float(get_duration(output)), 2))

    toolkitPath = Path("db") / "voxceleb2" / "wav" / spkid / emission / (n + ".wav")

    return name, spkid, duration, toolkitPath


def prepare_voxceleb2(sampling_frequency: int, canDeleteZIP: bool, in_data: Pathlike = ".", out_data: Pathlike = ".",
                      num_jobs: int = 20):
    out_data = Path(out_data)
    out_data.mkdir(parents=True, exist_ok=True)

    nameListe = "liste"
    if sampling_frequency != 16000:
        nameListe = nameListe + "_" + str(sampling_frequency)

    liste = open(out_data / nameListe, "w")

    with ProcessPoolExecutor(num_jobs) as ex:
        futures = []

        for segment in Path(in_data / "dev" / "aac").rglob("*.m4a"):
            futures.append(ex.submit(process_file, segment, out_data, sampling_frequency))

        for future in tqdm(futures, total=len(futures), desc=f"Processing VoxCeleb2..."):
            name, spkid, duration, segment = future.result()
            liste.write(f"{name} {spkid} {duration} {segment}\n")

    liste.close()

    if canDeleteZIP:
        for file in sorted(in_data.glob("vox2_dev_aac_part*")):
            os.remove(file)

        os.remove(in_data / "vox2_aac.zip")


def _process_file(file_path: Pathlike, output: Pathlike, sampling_frequency: int):
    # ffmpeg -v 8 -i {segment} -f wav -acodec pcm_s16le - |
    cmd = "ffmpeg -y -threads 1 -i " + str(file_path) + " -acodec pcm_s16le -ac 1 -ar " + str(
        sampling_frequency) + " -ab 48 -threads 1 " + str(output)
    proc = run(cmd, shell=True, stdout=PIPE, stderr=PIPE)
    print(cmd)

    # audio = np.frombuffer(raw_audio, dtype=np.float32)


def get_number_speaker(in_data: Pathlike, fname: str):
    speaker_ids = []
    in_data = Path(in_data)
    with open(in_data / fname, "r") as f:
        for line in f:
            line = line.strip().split()
            spkId = line[0].strip()
            speaker_ids.append(spkId)

    speaker_ids = set(speaker_ids)
    print(f"Voxceleb2 - Number of speaker in {fname}: {len(speaker_ids)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_data', metavar='in_data', type=str,
                        help='the path to the directory where the directory "dev" is stored')
    parser.add_argument('--out_data', metavar="out_data", type=str,
                        help='the path to the target directory where the liste will be stored')
    parser.add_argument('--downsampling', type=int, default=16000,
                        help='the value of sampling frequency (default: 16000)')
    parser.add_argument('--deleteZIP', action="store_true", default=False,
                        help='to delete the ZIP files already extracted (default: False)')
    parser.add_argument('--old_file', metavar="old_file", type=str,
                        help='old file name')

    args = parser.parse_args()

    # prepare_voxceleb2(args.downsampling, args.deleteZIP, Path(args.in_data), Path(args.out_data), 20)
    get_number_speaker(args.in_data, args.old_file)
