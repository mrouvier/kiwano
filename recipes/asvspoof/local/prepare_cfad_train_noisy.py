#!/usr/bin/env python3

import sys
import glob
from kiwano.utils import Pathlike, get_all_files
from pathlib import Path
import torchaudio
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from tqdm.auto import tqdm
from subprocess import PIPE, run
import os


def get_duration(file_path: str):
   info = torchaudio.info(file_path)
   return info.num_frames/info.sample_rate


def process_file(segment: Pathlike):
    name = str(segment).split("/")[-1].split(".")[0]
    s = str(segment).split("/")[-3].split(".")[0]

    spkid = "bonafide"

    if s == "fake_noise":
        spkid = "spoof"

    duration = str(round(float(get_duration(segment)),2))

    return name, spkid, duration, segment


def prepare_cfad(in_data: Pathlike = ".", out_data: Pathlike = ".", num_jobs: int = 20):
    out_data = Path(out_data)
    out_data.mkdir(parents=True, exist_ok=True)


    wav_lst = get_all_files(in_data / "noisy_version" / "train_noise" , match_and=[".wav"])

    print(len(wav_lst))

    liste = open(out_data / "liste", "w")

    with ProcessPoolExecutor(num_jobs) as ex:
        futures = []

        for segment in wav_lst:
            futures.append( ex.submit(process_file, segment) )

        for future in tqdm( futures, total=len(futures), desc="Processing CFAD..."):
            name, spkid, duration, segment = future.result()
            liste.write(f"{name} {spkid} {duration} {segment}\n")


    liste.close()



def _process_file(file_path: Pathlike, output: Pathlike):
    #ffmpeg -v 8 -i {segment} -f wav -acodec pcm_s16le - |
    cmd = f"ffmpeg -y -threads 1 -i {file_path} -acodec pcm_s16le -ac 1 -ar 16000 -ab 48 -threads 1 {output}"
    proc = run(cmd, shell=True, stdout=PIPE, stderr=PIPE)
    
    #audio = np.frombuffer(raw_audio, dtype=np.float32)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('in_data', metavar='in_data', type=str,
                        help='the path to the directory where the directory "dev" is stored')
    parser.add_argument('out_data', metavar="out_data", type=str,
                        help='the path to the target directory where the liste will be stored')

    args = parser.parse_args()

    prepare_cfad(Path(args.in_data), Path(args.out_data), 30)
