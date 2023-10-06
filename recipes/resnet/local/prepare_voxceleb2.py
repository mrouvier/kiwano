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


def get_duration(file_path: str):
   info = torchaudio.info(file_path)
   return info.num_frames/info.sample_rate


def process_file(segment: Pathlike, in_data: Pathlike):
    name = "_".join(str(segment).split("/")[-3:]).split(".")[0]
    spkid = str(segment).split("/")[-3]
    emission = str(segment).split("/")[-2]
    n = str(segment).split("/")[-1].split(".")[0]

    out = Path(in_data / "data" / spkid / emission)
    out.mkdir(parents=True, exist_ok=True)

    output = str(Path(in_data / "data" / spkid / emission / n ) )+".wav"

    _process_file(segment, Path(output))
    duration = str(round(float(get_duration(output)),2))

    return name, spkid, duration, output






def prepare_voxceleb2(in_data: Pathlike = ".", out_data: Pathlike = ".", num_jobs: int = 20):
    out_data = Path(out_data)
    out_data.mkdir(parents=True, exist_ok=True)

    liste = open(out_data / "liste", "w")

    with ProcessPoolExecutor(num_jobs) as ex:
        futures = []

        for segment in Path(in_data / "dev" / "aac").rglob("*.m4a"):
            futures.append( ex.submit(process_file, segment, out_data) )

        for future in tqdm( futures, total=len(futures), desc=f"Processing VoxCeleb2..."):
            name, spkid, duration, segment = future.result()
            liste.write(f"{name} {spkid} {duration} {segment}\n")


    liste.close()






def _process_file(file_path: Pathlike, output: Pathlike):
    #ffmpeg -v 8 -i {segment} -f wav -acodec pcm_s16le - |
    cmd = "ffmpeg -y -threads 1 -i "+str(file_path)+" -acodec pcm_s16le -ac 1 -ar 16000 -ab 48 -threads 1 "+str(output)
    proc = run(cmd, shell=True, stdout=PIPE, stderr=PIPE)

    #audio = np.frombuffer(raw_audio, dtype=np.float32)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('in_data', metavar='in_data', type=str, help='the path to vox1_meta.csv')
    parser.add_argument('out_data', metavar="out_data", type=str, help='the path to the target directory where the liste will be stored')

    args = parser.parse_args()

    prepare_voxceleb2(Path(args.in_data), Path(args.out_data), 20)
