#!/usr/bin/python3

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from subprocess import PIPE, run

import torchaudio
from tqdm import tqdm

from kiwano.utils import Pathlike, get_all_files


def get_duration(file_path: str):
    info = torchaudio.info(file_path)
    return info.num_frames / info.sample_rate


def process_file(segment: Pathlike, sampling_frequency: int, in_data: Pathlike):
    name = "_".join(str(segment).split("/")[-3:]).split(".")[0]
    spkid = str(segment).split("/")[-3]
    ref = str(segment).split("/")[-2]
    n = str(segment).split("/")[-1]
    duration = str(round(float(get_duration(segment)), 2))

    if sampling_frequency != 16000:
        name_dir = "wav"
        out = Path(in_data / name_dir / spkid / ref)
        out.mkdir(parents=True, exist_ok=True)

        output = str(Path(in_data / name_dir / spkid / ref / n))
        output = Path(output)

        cmd = (
            "ffmpeg -threads 1 -i "
            + str(segment)
            + " -acodec pcm_s16le -ac 1 -ar "
            + str(sampling_frequency)
            + " -ab 48 -threads 1 "
            + str(output)
        )
        print(cmd)
        proc = run(cmd, shell=True, stdout=PIPE, stderr=PIPE)
        segment = output

    return name, spkid, duration, segment


def prepare_voxceleb1(
    in_data: Pathlike,
    out_data: Pathlike,
    jobs: int,
    sampling_frequency: int,
    delete_zip: bool,
):
    in_data = Path(in_data)

    out_data = Path(out_data)
    out_data.mkdir(parents=True, exist_ok=True)

    nameListe = "liste"

    liste = open(out_data / nameListe, "w")

    wav_lst = get_all_files(in_data / "wav", match_and=[".wav"])

    with ProcessPoolExecutor(jobs) as ex:
        futures = []

        for segment in wav_lst:
            futures.append(
                ex.submit(process_file, segment, sampling_frequency, in_data)
            )

        for future in tqdm(futures, desc="Processing VoxCeleb1"):
            name, spkid, duration, segment = future.result()
            liste.write(f"{name} {spkid} {duration} {segment}\n")

    liste.close()

    for txt, trials in zip(
        ["veri_test2.txt", "list_test_hard2.txt", "list_test_all2.txt"],
        [
            "voxceleb1-o-cleaned.trials",
            "voxceleb1-h-cleaned.trials",
            "voxceleb1-e-cleaned.trials",
        ],
    ):

        r_txt = open(in_data / txt, "r")
        w_trials = open(out_data / trials, "w")

        for line in r_txt:
            line = line.strip().split(" ")

            if line[0] == "0":
                line[0] = "nontarget"
            else:
                line[0] = "target"

            w_trials.write(
                line[1].replace("/", "_").split(".")[0]
                + " "
                + line[2].replace("/", "_").split(".")[0]
                + " "
                + line[0]
                + "\n"
            )

        w_trials.close()
        r_txt.close()

    if delete_zip:
        for file in sorted(in_data.glob("vox1_dev_wav_part*")):
            os.remove(file)

        os.remove(in_data / "vox1_test_wav.zip")
        os.remove(in_data / "vox1_dev_wav.zip")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--thread", type=int, default=10, help="Number of parallel jobs (default: 10)"
    )
    parser.add_argument(
        "in_data", type=str, help='Path to the directory containing the "wav" directory'
    )
    parser.add_argument(
        "out_data",
        type=str,
        help="Path to the target directory where the list will be stored",
    )
    parser.add_argument(
        "--downsampling",
        type=int,
        default=16000,
        help="Downsampling frequency value (default: 16000)",
    )
    parser.add_argument(
        "--deleteZIP",
        action="store_true",
        default=False,
        help="Delete the already extracted ZIP files (default: False)",
    )

    args = parser.parse_args()

    prepare_voxceleb1(
        args.in_data, args.out_data, args.thread, args.downsampling, args.deleteZIP
    )
