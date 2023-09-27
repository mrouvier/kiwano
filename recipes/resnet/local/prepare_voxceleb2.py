#!/usr/bin/env python3

import sys
import glob
from kiwano.utils import Pathlike
from pathlib import Path
import torchaudio

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
    out.mkdir()

    output = Path(in_data / "data" / spkid / emission / n ) 
    print(segment)
    print(output)

    _process_file(segment, str(output)+".wav")
    duration = str(round(float(get_duration(output)),2))

    return name, spkid, duration, segment






def prepare_voxceleb2(in_data: Pathlike = ".", out_data: Pathlike = ".", num_jobs: int = 20):
    out_data = Path(out_data)
    out_data.mkdir(parents=True, exist_ok=True)

    liste = open(out_data / "liste", "w")

    with ProcessPoolExecutor(20) as ex:
        futures = []

        for segment in Path(in_data / "dev" / "aac").rglob("*.m4a"):
            futures.append( ex.submit(process_file, segment, in_data) )

        for future in tqdm( futures, desc=f"Processing VoxCeleb2..."):
            name, spkid, duration, segment = future.result()
            liste.write(f"{name} {spkid} {duration} {segment}\n")


    liste.close()






def _process_file(file_path: Pathlike):
    #ffmpeg -v 8 -i {segment} -f wav -acodec pcm_s16le - |
    cmd = "ffmpeg -threads 1 -i {file_path} -acodec pcm_s16le -ac 1 -ar 16000 -ab 48 -threads 1 -pipe:1 > {output}"
    proc = run(cmd, shell=True, stdout=PIPE, stderr=PIPE)
    #raw_audio = proc.stdout
    #audio = np.frombuffer(raw_audio, dtype=np.float32)

if __name__ == '__main__':
    prepare_voxceleb2(Path(sys.argv[1]), Path(sys.argv[2]), 10)

