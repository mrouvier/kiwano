#!/usr/bin/env python3

import sys
import glob
from kiwano.utils import Pathlike
from pathlib import Path
import librosa

from concurrent.futures import as_completed
from concurrent.futures.process import ProcessPoolExecutor
from tqdm import tqdm
from subprocess import PIPE, run


def get_duration_librosa(file_path: str):
   audio_data, sample_rate = librosa.load(file_path)
   duration = librosa.get_duration(y=audio_data, sr=sample_rate)
   return duration




def prepare_voxceleb2(in_data: Pathlike = ".", out_data: Pathlike = ".", num_jobs: int = 20):
    out_data = Path(out_data)
    out_data.mkdir(parents=True, exist_ok=True)

    liste = open(out_data / "liste", "w")


    with ProcessPoolExecutor(num_jobs) as ex:
        futures = []

        with open(in_data / "vox2_meta.csv", "r") as f:
            for line in f:
                name, spkid, vggfaceid, gender, split = line.strip().split(" \t")

                d = Path(in_data / "dev" / "wav" / spkid)
                d.mkdir(parents=True, exist_ok=True)

                for segment in Path(in_data / "dev" / "aac" /spkid ).rglob("*.m4a"):
                    futures.append( ex.submit(_process_file,  segment) )

        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing VoxCeleb2...", leave=False):
            future.result()


    with open(in_data / "vox2_meta.csv", "r") as f:
        next(f)
        for line in f:
            name, spkid, vggfaceid, gender, split = line.strip().split(" \t")

            for segment in Path(in_data / "dev" / "wav" /spkid ).rglob("*.wav"):
                name = "_".join(str(segment).split("/")[-3:]).split(".")[0]
                duration = str(round(float(get_duration_librosa(segment)),2))
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

