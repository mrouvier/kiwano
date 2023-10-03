#!/usr/bin/python3

import sys
from kiwano.utils import Pathlike
from pathlib import Path
import torchaudio
from tqdm import tqdm
from concurrent.futures.process import ProcessPoolExecutor

import argparse

def get_duration(file_path: str):
   info = torchaudio.info(file_path)
   return info.num_frames/info.sample_rate

def process_file(segment: Pathlike):
    name = "_".join(str(segment).split("/")[-3:]).split(".")[0]
    spkid = str(segment).split("/")[-3]
    duration = str(round(float(get_duration(segment)),2))
    return name, spkid, duration, segment




def prepare_voxceleb1(in_data: Pathlike, out_data: Pathlike):
    in_data = Path(in_data)

    out_data = Path(out_data)
    out_data.mkdir(parents=True, exist_ok=True)

    liste = open(out_data / "liste", "w")

    with ProcessPoolExecutor(20) as ex:
        futures = []

        for segment in (in_data / "wav").rglob("*.wav"):
            futures.append(ex.submit(process_file, segment))
        for future in tqdm(futures, desc="Processing VoxCeleb1"):

            name, spkid, duration, segment = future.result()
            liste.write(f"{name} {spkid} {duration} {segment}\n")

            #print(f"{name} {spkid} {duration} {segment}\n")

    liste.close()






if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('in_data', metavar='in_data', type=str,
                        help='the path to the directory where the directory "wav" is stored')
    parser.add_argument('out_data', metavar="out_data", type=str,
                        help='the path to the target directory where the liste will be stored')

    args = parser.parse_args()

    prepare_voxceleb1(args.in_data, args.out_data)

