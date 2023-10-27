#!/usr/bin/python3

import sys
from kiwano.utils import Pathlike
from pathlib import Path
import torchaudio
from tqdm import tqdm
from concurrent.futures.process import ProcessPoolExecutor

from subprocess import PIPE, run

import argparse
import os

def get_duration(file_path: str):
   info = torchaudio.info(file_path)
   return info.num_frames/info.sample_rate

def process_file(segment: Pathlike, sampling_frequency: int, in_data: Pathlike):
    name = "_".join(str(segment).split("/")[-3:]).split(".")[0]
    spkid = str(segment).split("/")[-3]
    ref = str(segment).split("/")[-2]
    n = str(segment).split("/")[-1]
    duration = str(round(float(get_duration(segment)),2))

    if sampling_frequency != 16000:
        #print(n)
        nameDir = "wav_"+str(sampling_frequency)
        out = Path(in_data / nameDir / spkid / ref)
        out.mkdir(parents=True, exist_ok=True)

        output = str(Path(in_data / nameDir / spkid / ref / n))
        output = Path(output)

        cmd = "ffmpeg -threads 1 -i " + str(segment) + " -acodec pcm_s16le -ac 1 -ar " + str(sampling_frequency) + " -ab 48 -threads 1 " + str(output)
        print(cmd)
        proc = run(cmd, shell=True, stdout=PIPE, stderr=PIPE)
        segment = output

    return name, spkid, duration, segment



def prepare_voxceleb1(in_data: Pathlike, out_data: Pathlike, jobs: int, sampling_frequency: int, canDeleteZIP: bool):
    in_data = Path(in_data)

    out_data = Path(out_data)
    out_data.mkdir(parents=True, exist_ok=True)

    nameListe = "liste"
    if sampling_frequency != 16000:
        nameListe = nameListe+"_"+str(sampling_frequency)

    liste = open(out_data / nameListe, "w")

    with ProcessPoolExecutor(jobs) as ex:
        futures = []

        for segment in (in_data / "wav").rglob("*.wav"):
            futures.append(ex.submit(process_file, segment, sampling_frequency, in_data))
        for future in tqdm(futures, desc="Processing VoxCeleb1"):

            name, spkid, duration, segment = future.result()
            liste.write(f"{name} {spkid} {duration} {segment}\n")


    liste.close()


    for txt, trials in zip( ["veri_test2.txt", "list_test_hard2.txt", "list_test_all2.txt"], ["voxceleb1-o-cleaned.trials", "voxceleb1-h-cleaned.trials", "voxceleb1-e-cleaned.trials"] ):

        r_txt = open(in_data / txt, "r")
        w_trials = open(out_data / trials, "w")

        for line in r_txt:
            line = line.strip().split(" ")

            if line[0] == "0":
                line[0] = "nontarget"
            else:
                line[0] = "target"

            w_trials.write(line[1].replace("/", "_").split(".")[0]+" "+line[2].replace("/", "_").split(".")[0]+" "+line[0]+"\n")

        w_trials.close()
        r_txt.close()

    if canDeleteZIP :
        for file in sorted(in_data.glob("vox1_dev_wav_part*")):

            os.remove(file)

        os.remove(in_data / "vox1_test_wav.zip")
        os.remove(in_data / "vox1_dev_wav.zip")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--thread', type=int, default=10,
                        help='number of parallel jobs')
    parser.add_argument('in_data', metavar='in_data', type=str,
                        help='the path to the directory where the directory "wav" is stored')
    parser.add_argument('out_data', metavar="out_data", type=str,
                        help='the path to the target directory where the liste will be stored')
    parser.add_argument('--downsampling', type=int, default=16000,
                        help='the value of sampling frequency (default: 16000)')
    parser.add_argument('--deleteZIP', action="store_true", default=False,
                        help='to delete the ZIP files already extracted (default: False)')

    args = parser.parse_args()

    prepare_voxceleb1(args.in_data, args.out_data, args.thread, args.downsampling, args.deleteZIP)

