#!/usr/bin/python3

import sys
from kiwano.utils import Pathlike, get_all_files, download_from_github, check_md5
from pathlib import Path
import torchaudio
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

from subprocess import PIPE, run

import argparse
import os
import json
import datetime
import logging

ARAB_CELEB_URL = [
    ["https://github.com/CeLuigi/ArabCeleb/blob/main/veri_test.txt", "6416a29369920d76910231a8e90293ef"]
]

def prepare_trials(in_dir: Pathlike, out_dir: Pathlike):
    with open(in_dir / "veri_test.txt") as f:
        trials = f.readlines()

    with open(out_dir / "trials.lst", "w") as f:
        for trial in trials:
            values = trial.split()
            line = f"{values[1]} {values[2]} {'target' if values[0] == '1' else 'nontarget'}\n"
            f.write(line)


def process_audio_file(output_celeb_dir: Pathlike, input_celeb_dir: Pathlike, id_video: str, utterance: list, sampling_frequency: int):
    filename = str(id_video) + '.mp4'
    utterance_fn = utterance['name']
    utterance_from = utterance['from']
    utterance_to = utterance['to']
    output_wav_filename = output_celeb_dir / utterance_fn
    input_mp4_filename = input_celeb_dir / filename
    cmd = f"ffmpeg -y -threads 1 -i {input_mp4_filename} -acodec pcm_s16le -ac 1 -ar {sampling_frequency} -ab 48 -ss {str(datetime.timedelta(seconds=utterance_from))} -to {str(datetime.timedelta(seconds=utterance_to))} {output_wav_filename}"
    output = run(cmd, shell=True, stdout=PIPE, stderr=PIPE)

def prepare_arab_celeb(in_data: Pathlike, out_data: Pathlike, jobs: int, sampling_frequency: int):
    in_data = Path(in_data)

    out_data = Path(out_data)
    out_data.mkdir(parents=True, exist_ok=True)

    utterance_info_file = in_data / "utterance_info.json"

    with open(utterance_info_file) as f:
        utterances = json.load(f)

    logging.info("Processing audio files...")

    with ProcessPoolExecutor(max_workers=jobs) as executor:
        futures = []
        for key, value in utterances.items():
            output_celeb_dir = out_data / key
            output_celeb_dir.mkdir(parents=True, exist_ok=True)

            input_celeb_dir = in_data / key

            for id_video, (url, utterance_list) in enumerate(value['utterances'].items()):
                for utterance in utterance_list:
                    futures.append(executor.submit(process_audio_file, output_celeb_dir, input_celeb_dir, id_video, utterance, sampling_frequency))

        for future in tqdm(futures, total=len(futures), desc="Processing audio files"):
            future.result()
    logging.info("Processing audio files done")

    download_from_github(ARAB_CELEB_URL[0][0], in_data / "veri_test.txt")
    prepare_trials(in_data, out_data)

    logging.info("Processing ArabCeleb done")
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare ArabCeleb dataset')
    parser.add_argument('in_data', type=str, 
                        help='Input data directory')
    parser.add_argument('out_data', type=str, 
                        help='Output data directory')
    parser.add_argument('--jobs', type=int, default=8, 
                        help='Number of jobs to run in parallel')
    parser.add_argument('--sampling-frequency', type=int, default=16000, 
                        help='Sampling frequency')

    args = parser.parse_args()

    prepare_arab_celeb(args.in_data, args.out_data, args.jobs, args.sampling_frequency)




