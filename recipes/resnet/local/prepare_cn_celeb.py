#!/usr/bin/python3

from kiwano.utils import Pathlike
from pathlib import Path
import torchaudio
from tqdm import tqdm
from concurrent.futures.process import ProcessPoolExecutor

import argparse
import logging
from subprocess import PIPE, run

import os

def get_duration(file_path: str):
   info = torchaudio.info(file_path)
   return info.num_frames/info.sample_rate

def process_audio_file(file_path: Pathlike, output: Pathlike):
    cmd = f"ffmpeg -threads 1 -i {file_path} -acodec pcm_s16le -ac 1 -ar 16000 -ab 48 {output}"
    
    proc = run(cmd, shell=True, stdout=PIPE, stderr=PIPE)
    
    if proc.returncode != 0:
        error_message = proc.stderr.decode('utf-8')
        raise RuntimeError(f"FFmpeg command failed with error code {proc.returncode}: {error_message}")

def process_file(segment: Pathlike, out_data: Pathlike, spkid: str):
    name = "_".join(str(segment).split("/")[-3:]).split(".")[0]
    n = str(segment).split("/")[-1].split(".")[0]

    out_data.mkdir(parents=True, exist_ok=True)

    output_path = str(Path(out_data / n)) + ".wav"
    if not Path(output_path).exists():
        process_audio_file(segment, Path(output_path))

    duration = str(round(float(get_duration(output_path)), 2))

    return name, spkid, duration, output_path


def prepare_trials(in_data: Pathlike, out_data: Pathlike):
    # change trials.lst to have the correct format
    dictEnroll = {}

    with open(in_data / "CN-Celeb_flac" / "eval" / "lists" / "enroll.lst", 'r') as enroll:
        for line in enroll:
            col1, col2 = line.strip().split(' ')
            dictEnroll[col1] = col2
    trial_dir = Path(out_data / "CN-Celeb_flac" / "eval" / "lists")
    trial_dir.mkdir(parents=True, exist_ok=True)
    
    with open(in_data / "CN-Celeb_flac" / "eval" / "lists" / "trials.lst", 'r') as trials:
        with open(out_data / "CN-Celeb_flac" / "eval" / "lists" / "new_trials.lst", 'w') as new_trials:
            for line in trials:
                col1, col2, col3 = line.strip().split(' ')
                if col3 == "0":
                    col3 = "nontarget"
                else:
                    col3 = "target"
                col1 = dictEnroll[col1]
                col1 = "eval_enroll_" + col1[7:-4]
                col2 = "eval_test_" + col2[5:-4]
                new_trials.write(col1 + " " + col2 + " " + col3 + "\n")

def prepare_cn_celeb(canDeleteZIP: bool, in_data: Pathlike, out_data: Pathlike, jobs: int):
    in_data = Path(in_data)

    out_data = Path(out_data)
    out_data.mkdir(parents=True, exist_ok=True)

    listeTrain = open(out_data / "listeTrain", "w")
    listeDev = open(out_data / "listeDev", "w")
    listeTest = open(out_data / "listeTest", "w")

    dev_files_list = open(in_data / "CN-Celeb_flac" / "dev" / "dev.lst", "r").read().splitlines()

    with ProcessPoolExecutor(jobs) as ex:
        futuresTrain = []
        futuresDev = []
        futuresTest = []

        for segment in (in_data / "CN-Celeb2_flac" / "data").rglob("*.flac"):
            spkid = str(segment).split("/")[-2]
            out_data_path = Path(out_data / "CN-Celeb2_flac" / "train" / "wav" / spkid)
            futuresTrain.append(ex.submit(process_file, segment, out_data_path, spkid))
            
        for segment in (in_data / "CN-Celeb_flac" / "data").rglob("*.flac"):
            if segment.parts[4] in dev_files_list:
                futures = futuresDev
                spkid = spkid = str(segment).split("/")[4]
                out_data_path = Path(out_data/ "CN-Celeb_flac" / "dev" / "wav" / spkid)
            else:
                futures = futuresTrain
                spkid = spkid = str(segment).split("/")[-2]
                out_data_path = Path(out_data / "CN-Celeb_flac" / "train" / "wav" / spkid)
            futures.append(ex.submit(process_file, segment, out_data_path, spkid))

        for segment in (in_data / "CN-Celeb_flac" / "eval").rglob("*.flac"):
            spkid = spkid = str(segment).split("/")[-1].split("-")[0]
            out_data_path = Path(out_data/ "CN-Celeb_flac" / "eval" / "wav")
            futuresTest.append(ex.submit(process_file, segment, out_data_path, spkid))

        def process_futures(futures, output_file, desc):
            for future in tqdm(futures, desc=desc):
                name, spkid, duration, segment = future.result()
                output_file.write(f"{name} {spkid} {duration} {segment}\n")

        process_futures(futuresTrain, listeTrain, "Processing Cn Celeb 2 (Train)")
        process_futures(futuresDev, listeDev, "Processing Cn Celeb (Dev)")
        process_futures(futuresTest, listeTest, "Processing Cn Celeb (Test)")


    listeTrain.close()
    listeDev.close()
    listeTest.close()

    prepare_trials(in_data, out_data)

    if canDeleteZIP :
        for file in sorted(in_data.glob("cn-celeb2_v2.tar*")):

            os.remove(file)

        os.remove(in_data / "cn-celeb2.tar.gz")
        os.remove(in_data / "cn-celeb_v2.tar.gz")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--thread', type=int, default=16,
                    help='Number of parallel jobs (default: 16)')
    parser.add_argument('in_data', metavar='in_data', type=str,
                        help='the path to the directory where CN-Celeb2_flac and CN-Celeb_flac are stored')
    parser.add_argument('out_data', metavar="out_data", type=str,
                        help='the path to the target directory where the liste will be stored')
    parser.add_argument('--deleteZIP', action="store_true", default=False,
                        help='to delete the ZIP files already extracted (default: False)')

    args = parser.parse_args()

    prepare_cn_celeb(args.deleteZIP, args.in_data, args.out_data, args.thread)

