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
import os


def get_duration(file_path: str):
    info = torchaudio.info(file_path)
    return info.num_frames / info.sample_rate


def process_file(segment: Pathlike, in_data: Pathlike, sampling_frequency: int):
    # db/nist/nist-sre-test2004/xeot.sph
    name = segment.split('/')[-1].split('.')[0]
    nameDir = "wav"
    if sampling_frequency != 16000:
        nameDir = nameDir + "_" + str(sampling_frequency)

    output = str(Path(in_data / name)) + f"_{nameDir}_.wav"

    if not Path(output).exists():
        _process_file(segment, Path(output), sampling_frequency)


def convert_sph_to_wav_nist(sampling_frequency: int, canDeleteZIP: bool, in_data: Pathlike = ".",
                            out_data: Pathlike = ".",
                            num_jobs: int = 20):
    out_data = Path(out_data)
    out_data.mkdir(parents=True, exist_ok=True)

    with ProcessPoolExecutor(num_jobs) as ex:
        futures = []

        for segment in Path(in_data).rglob("*.sph"):
            futures.append(ex.submit(process_file, segment, out_data, sampling_frequency))

        for future in tqdm(futures, total=len(futures), desc=f"Processing Nist Train..."):
            future.result()

    if canDeleteZIP:
        for file in sorted(in_data.glob("*.sph")):
            os.remove(file)


def _process_file(file_path: Pathlike, output: Pathlike, sampling_frequency: int):
    # ffmpeg -v 8 -i {segment} -f wav -acodec pcm_s16le - |
    cmd = "ffmpeg -y -threads 1 -i " + str(file_path) + " -acodec pcm_s16le -ac 1 -ar " + str(
        sampling_frequency) + " -ab 48 -threads 1 " + str(output)
    proc = run(cmd, shell=True, stdout=PIPE, stderr=PIPE)
    print(cmd)

    # audio = np.frombuffer(raw_audio, dtype=np.float32)


def get_number_speaker(in_data: Pathlike, fname: str):
    speaker_ids = []
    in_data = Path(in_data)
    with open(in_data / fname, "r") as f:
        for line in f:
            line = line.strip().split()
            spkId = line[0].strip()
            speaker_ids.append(spkId)

    speaker_ids = set(speaker_ids)
    print(f"Voxceleb2 - Number of speaker in {fname}: {len(speaker_ids)}")


def create_new_train_list(in_data: Pathlike, out_data: Pathlike, oldfile: str):
    in_data = Path(in_data)
    out_data = Path(out_data)
    out_data.mkdir(parents=True, exist_ok=True)

    listeTrain = open(out_data / f"{oldfile}.edited", "w")
    with open(in_data / oldfile, "r") as f:
        lines = f.readlines()
        for line in tqdm(lines):
            line = line.strip().split()
            if len(line) == 8:
                idspk = line[1].strip()
                fname = f"{line[2].strip()}.wav"
                full_path = in_data / fname
                if full_path.exists():
                    listeTrain.write(f"{idspk} {fname}\n")
                else:
                    print(full_path, flush=True)

    listeTrain.close()


def create_new_eval_list(in_data: Pathlike, out_data: Pathlike, oldfile: str):
    in_data = Path(in_data)
    out_data = Path(out_data)
    out_data.mkdir(parents=True, exist_ok=True)

    listeEval = open(out_data / f"{oldfile}.edited", "w")
    with open(in_data / oldfile, "r") as f:
        lines = f.readlines()
        for i, line in tqdm(enumerate(lines)):
            line = line.strip().split()
            if len(line) == 8:
                spkId1 = f"{line[1].strip()}.wav"
                path1 = in_data / spkId1
                others = lines[(i + 1):]
                for line2 in others:
                    line2 = line2.strip().split()
                    if len(line2) == 8:
                        spkId2 = f"{line2[1].strip()}.wav"
                        path2 = in_data / spkId2
                        label = 1 if spkId1 == spkId2 else 0
                        if path1.exists() and path2.exists():
                            listeEval.write(f"{label} {spkId1} {spkId2}\n")
                        else:
                            print(f"{label} {path1} {path2}", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_data', metavar='in_data', type=str,
                        help='the path to the directory where the directory "dev" is stored')
    parser.add_argument('--out_data', metavar="out_data", type=str,
                        help='the path to the target directory where the liste will be stored')
    parser.add_argument('--downsampling', type=int, default=16000,
                        help='the value of sampling frequency (default: 16000)')
    parser.add_argument('--deleteZIP', action="store_true", default=False,
                        help='to delete the ZIP files already extracted (default: False)')
    parser.add_argument('--old_file', metavar="old_file", type=str,
                        help='old file name')

    args = parser.parse_args()

    # prepare_voxceleb2(args.downsampling, args.deleteZIP, Path(args.in_data), Path(args.out_data), 20)
    convert_sph_to_wav_nist(args.downsampling, args.deleteZIP, Path(args.in_data), Path(args.out_data), 20)
    # get_number_speaker(args.in_data, args.old_file)
    # create_new_train_list(args.in_data, args.out_data, args.old_file)
    # create_new_eval_list(args.in_data, args.in_data, args.old_file)