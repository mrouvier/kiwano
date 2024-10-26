#!/usr/bin/python3

import sys
from kiwano.utils import Pathlike, get_all_files
from pathlib import Path
import torchaudio
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
import soundfile as sf


from silero_vad import load_silero_vad, read_audio, get_speech_timestamps, collect_chunks


from subprocess import PIPE, run

import argparse
import os

model = load_silero_vad()


def get_duration(file_path: str):
   info = torchaudio.info(file_path)
   return info.num_frames/info.sample_rate


def ffmpeg(file_path: Pathlike, sampling_frequency: int):
    cmd = "ffmpeg -y -threads 1 -i "+str(file_path)+" -acodec pcm_s16le -ac 1 -ar "+str(sampling_frequency)+" -ab 48 -threads 1 -f wav -"
    proc = run(cmd, shell=True, stdout=PIPE, stderr=PIPE)
    raw_audio = proc.stdout
    audio_array = np.frombuffer(raw_audio, dtype=np.int16)
    wav = torch.tensor(audio_array.astype(np.float32) / 32768.0)

    return wav



def process_file(segment: Pathlike, out_data: Pathlike, sampling_frequency: int, vad: bool):
    name = "_".join(str(segment).split("/")[-3:]).split(".")[0]
    spkid = str(segment).split("/")[-3]
    emission = str(segment).split("/")[-2]
    n = str(segment).split("/")[-1]

    output = str(Path(out_data / "wav" / spkid / emission / n))
    duration = -1
    wav = torch.tensor([])

    if sampling_frequency != 16000:
        wav = ffmpeg(segment, sampling_frequency)

    if vad:
        if sampling_frequency == 16000:
            wav, sr = torchaudio.load(segment)
            wav = wav[0]

        speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=sampling_frequency, threshold=0.6)

        if len(speech_timestamps) > 0:
            wav = collect_chunks(speech_timestamps, wav)

    if sampling_frequency != 16000 or vad:
        out = Path(out_data / "wav" / spkid / emission)
        out.mkdir(parents=True, exist_ok=True)

        sf.write(str(output), wav, sampling_frequency)
        segment = output
    else:
        duration = str(round(float(get_duration(segment)),2))

    return name, spkid, duration, segment



def prepare_voxceleb1(in_data: Pathlike = ".", out_data: Pathlike = ".", sampling_frequency: int = 16000, delete_zip: bool = False, num_jobs: int = 30, vad: bool = False):
    out_data = Path(out_data)
    out_data.mkdir(parents=True, exist_ok=True)

    liste = open(out_data / "liste", "w")

    wav_lst = get_all_files(in_data / "wav", match_and=[".wav"])

    with ProcessPoolExecutor(num_jobs) as ex:
        futures = []

        for segment in wav_lst:
            futures.append(ex.submit(process_file, segment, out_data, sampling_frequency, vad))

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

    if delete_zip:
        for file in sorted(in_data.glob("vox1_dev_wav_part*")):
            os.remove(file)

        os.remove(in_data / "vox1_test_wav.zip")
        os.remove(in_data / "vox1_dev_wav.zip")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('in_data', type=str,
                    help='Path to the directory containing the "wav" directory')
    parser.add_argument('out_data', type=str,
                    help='Path to the target directory where the list will be stored')
    parser.add_argument('--num_jobs', type=int, default=30,
                    help='Number of parallel jobs (default: 30)')
    parser.add_argument('--vad', action='store_true', default=False,
                    help='Apply VAD (default: False)')
    parser.add_argument('--downsampling', type=int, default=16000,
                    help='Downsampling frequency value (default: 16000)')
    parser.add_argument('--delete_zip', action='store_true', default=False,
                    help='Delete the already extracted ZIP files (default: False)')

    args = parser.parse_args()

    prepare_voxceleb1(Path(args.in_data), Path(args.out_data), args.downsampling, args.delete_zip, args.num_jobs, args.vad)

