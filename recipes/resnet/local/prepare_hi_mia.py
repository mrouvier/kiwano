#!/usr/bin/python3

from kiwano.utils import Pathlike, get_all_files
from pathlib import Path
import torchaudio
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

from subprocess import PIPE, run

import argparse
import logging
import os

def get_duration(file_path: str):
   info = torchaudio.info(file_path)
   return info.num_frames/info.sample_rate

def process_file(segment: Pathlike, sampling_frequency: int, in_data: Pathlike, out_data: Pathlike):
    filename = str(segment).split("/")[-1]
    spkid = filename.split("_")[0]
    n = filename.split("_")[-1]
    duration = str(round(float(get_duration(segment)),2))

    spkid_dir = out_data / spkid
    spkid_dir.mkdir(parents=True, exist_ok=True)

    output_file = spkid_dir / filename
    if not output_file.exists():
        cmd = f"ffmpeg -threads 1 -i {segment} -acodec pcm_s16le -ac 1 -ar {sampling_frequency} -ab 48 {output_file}"
        proc = run(cmd, shell=True, stdout=PIPE, stderr=PIPE)

    return filename, spkid, duration, spkid_dir / filename

def prepare_trials(in_dir: Pathlike, out_dir: Pathlike):
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trials_file = in_dir / "trials_mic"
    output_file = out_dir / "trials"

    processed_pairs = set()
    
    with open(trials_file, "r") as f_in, open(output_file, "w") as f_out:
        for line in f_in:
            file1, file2, label = line.strip().split()
            
            file2 = file2.replace("_{}_", "_7_01_") # Update the trials to only consider the HiFi mono audios
            file_pair = tuple(sorted([file1, file2])) # Remove duplicates
            
            if file_pair not in processed_pairs:
                processed_pairs.add(file_pair)
                
                file1_path = out_dir / f"test/{file1.split('_')[0]}/{file1}"
                file2_path = out_dir / f"test/{file2.split('_')[0]}/{file2}"
                
                f_out.write(f"{file1_path} {file2_path} {label}\n")

def prepare_hi_mia(in_data: Pathlike, out_data: Pathlike, jobs: int, sampling_frequency: int):
    in_data = Path(in_data)

    out_data = Path(out_data)
    out_data.mkdir(parents=True, exist_ok=True)

    nameListe = "liste"
    liste = open(out_data / nameListe, "w")

    in_dirs = ["train/SPEECHDATA", "dev/SPEECHDATA", "test"]

    with ProcessPoolExecutor(jobs) as ex:
        futures = []
        logging.info("Preparing jobs...")
        for in_dir in in_dirs:
            wav_lst = get_all_files(in_data / in_dir, match_and=[".wav"])
            wav_lst = [wav for wav in wav_lst if "_7_01_" in wav and wav.endswith(".wav")]
            for segment in wav_lst:
                out_dir = out_data / in_dir.split("/")[0]
                futures.append(ex.submit(process_file, segment, sampling_frequency, in_data, out_dir))

        for future in tqdm(futures, desc="Processing audio files"):
            filename, spkid, duration, segment = future.result()
            liste.write(f"{filename} {spkid} {duration} {segment}\n")

    liste.close()

    prepare_trials(in_data / "test", out_data)

    train_tar = in_data / "train.tar.gz"
    dev_tar = in_data / "dev.tar.gz"
    test_tar = in_data / "test_v2.tar.gz"

    if os.path.exists(train_tar):
        os.remove(train_tar)
    if os.path.exists(dev_tar):
        os.remove(dev_tar)
    if os.path.exists(test_tar):
        os.remove(test_tar)
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("in_data", type=str, metavar="IN_DATA", 
                        help="Path to the input directory")
    parser.add_argument("out_data", type=str, metavar="OUT_DATA", 
                        help="Path to the output directory")
    parser.add_argument("sampling_frequency", type=int, metavar="SAMPLING_FREQUENCY", 
                        help="Sampling frequency of the audio files")
    parser.add_argument("--jobs", type=int, default=8, 
                        help="Number of parallel jobs (default: 8)")

    args = parser.parse_args()

    prepare_hi_mia(args.in_data, args.out_data, args.jobs, args.sampling_frequency)