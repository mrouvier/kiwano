#!/usr/bin/env python3

import sys
import glob
from kiwano.utils import Pathlike
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


def process_file(segment: Pathlike, in_data: Pathlike ):
    name = str(segment).split("/")[-1].split(".")[0]

    output = Path(in_data  / "wav" / (name + ".wav" ))

    _process_file(segment, output)

    duration = str(round(float(get_duration(output)),2))

    return name, duration, output


def prepare_asvspoof5(in_data: Pathlike = ".", out_data: Pathlike = ".", num_jobs: int = 20):
    out_data = Path(out_data)
    out_data.mkdir(parents=True, exist_ok=True)

    spkid = {}

    rliste = open(in_data / "ASVspoof5.dev.metadata.txt" )
    for line in rliste:
        line = line.strip().split(" ")
        spkid[ line[1] ] = line[5]
    rliste.close()


    liste = open(out_data / "liste", "w")

    with ProcessPoolExecutor(num_jobs) as ex:
        futures = []

        for segment in os.scandir( Path(in_data / "flac_D") ):
            if segment.is_file() == True:
                if segment.name.endswith(".flac"):
                    if Path(segment).stem in spkid:
                        futures.append( ex.submit(process_file, segment.path, out_data) )

        for future in tqdm( futures, total=len(futures), desc="Processing ASVSpoof5..."):
            name, duration, segment = future.result()
            liste.write(f"{name} {spkid[name]} {duration} {segment}\n")


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

    prepare_asvspoof5(Path(args.in_data), Path(args.out_data), 30)
