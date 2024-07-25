#!/usr/bin/python3

import sys
from kiwano.utils import Pathlike, get_all_files
from pathlib import Path
import torchaudio
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor


import argparse

def get_duration(file_path: str):
   info = torchaudio.info(file_path)
   return info.num_frames/info.sample_rate

def process_file(segment):
    duration = str(round(float(get_duration(segment)),2))
    name = str(segment).split("/")[-1].split(".")[0]
    spkid = str(segment).split("/")[-3]

    return name, spkid, duration, segment

def prepare_musan(in_data: Pathlike, out_data: Pathlike, num_jobs: int):
    in_data = Path(in_data)

    out_data = Path(out_data)
    out_data.mkdir(parents=True, exist_ok=True)

    liste = open(out_data / "liste", "w")

    wav_lst = get_all_files(in_data, match_and=[".wav"])

    with ProcessPoolExecutor(num_jobs) as ex:
        futures = []

        for segment in wav_lst:
            futures.append( ex.submit(process_file, segment) )

        for future in tqdm(futures, total=len(futures), desc=f"Processing MUSAN..."):
            name, spkid, duration, segment = future.result()
            liste.write(f"{name} {spkid} {duration} {segment}\n")

    liste.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('in_data', metavar='in_data', type=str,
                        help='the path to the target directory where the wav files are stored')
    parser.add_argument('out_data', metavar="out_data", type=str,
                        help='the path to the target directory where the liste will be stored')

    args = parser.parse_args()

    prepare_musan(args.in_data, args.out_data, 20)
