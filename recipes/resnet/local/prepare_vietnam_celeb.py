#!/usr/bin/python3
import os

from kiwano.utils import Pathlike
from pathlib import Path
import torchaudio
from tqdm import tqdm
from concurrent.futures.process import ProcessPoolExecutor

import argparse


def get_duration(file_path: str):
    info = torchaudio.info(file_path)
    return info.num_frames / info.sample_rate


def process_file(segment: Pathlike):
    name = "_".join(str(segment).split("/")[-3:]).split(".")[0]
    spkid = str(segment).split("/")[-2]
    duration = str(round(float(get_duration(segment)), 2))
    return name, spkid, duration, segment


def prepare_vietnam_celeb(in_data: Pathlike, out_data: Pathlike):
    in_data = Path(in_data)

    out_data = Path(out_data)
    out_data.mkdir(parents=True, exist_ok=True)

    listeTrain = open(out_data / "liste", "w")
    listeTest = open(out_data / "listeTest", "w")

    trainFiles = []

    with open(in_data / "vietnam-celeb-t.txt", "r") as f:
        for line in f:
            line = line.strip().split("\t")
            dir = line[0]
            nameFile = line[1]
            trainFiles.append(dir + "/" + nameFile)

    with ProcessPoolExecutor(20) as ex:
        futuresTrain = []
        futuresTest = []

        for segment in tqdm((in_data / "data").rglob("*.wav"), desc="Creating Jobs"):
            part = segment.parts
            endOfPart = part[-2:]
            endOfPath = endOfPart[0] + "/" + endOfPart[1]
            if endOfPath not in trainFiles:
                futuresTest.append(ex.submit(process_file, segment))
            else:
                futuresTrain.append(ex.submit(process_file, segment))

        for future in tqdm(futuresTest, desc="Processing Vietnam_Celeb Test"):
            name, spkid, duration, segment = future.result()
            listeTest.write(f"{name} {spkid} {duration} {segment}\n")

        for future in tqdm(futuresTrain, desc="Processing Vietnam_Celeb Train"):
            name, spkid, duration, segment = future.result()
            listeTrain.write(f"{name} {spkid} {duration} {segment}\n")

            # print(f"{name} {spkid} {duration} {segment}\n")

    listeTest.close()
    listeTrain.close()


def create_new_train_list(in_data: Pathlike, out_data: Pathlike, oldfile: str):
    in_data = Path(in_data)
    out_data = Path(out_data)
    out_data.mkdir(parents=True, exist_ok=True)

    listeTrain = open(out_data / f"{oldfile}.edited", "w")
    with open(in_data / oldfile, "r") as f:
        for line in f:
            line = line.strip().split()
            dir = line[0].strip()
            fname = line[1].strip()
            path = f"{dir}/{fname}"
            full_path = f"db/vietnam_celeb/data/{path}"
            if os.path.exists(full_path):
                listeTrain.write(f"{dir} {path}\n")

    listeTrain.close()


def create_new_eval_list(in_data: Pathlike, out_data: Pathlike, oldfile: str):
    in_data = Path(in_data)
    out_data = Path(out_data)
    out_data.mkdir(parents=True, exist_ok=True)

    listeEval = open(out_data / f"{oldfile}.edited", "w")
    with open(in_data / oldfile, "r") as f:
        for line in f:
            line = line.strip().split("\t")
            speaker1 = line[1].strip()
            speaker2 = line[-1].strip()
            label = line[0].strip()
            path1 = f"db/vietnam_celeb/data/{speaker1}"
            path2 = f"db/vietnam_celeb/data/{speaker2}"
            if os.path.exists(path1) and os.path.exists(path2):
                listeEval.write(f"{label} {speaker1} {speaker2}\n")

    listeEval.close()


def get_number_speaker(in_data: Pathlike, fname: str):
    speaker_ids = []
    in_data = Path(in_data)
    with open(in_data / fname, "r") as f:
        for line in f:
            line = line.strip().split()
            spkId = line[0].strip()
            speaker_ids.append(spkId)

    speaker_ids = set(speaker_ids)
    print(f"Vietnam_celeb - Number of speaker in {fname}: {len(speaker_ids)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_data', metavar='in_data', type=str,
                        help='the path to the directory "data", where the wav files are stored')
    parser.add_argument('--out_data', metavar="out_data", type=str,
                        help='the path to the target directory where the liste will be stored')
    parser.add_argument('--old_file', metavar="old_file", type=str,
                        help='old file name')

    args = parser.parse_args()

    # prepare_vietnam_celeb(args.in_data, args.out_data)
    # create_new_train_list(args.in_data, args.out_data, args.old_file)
    create_new_eval_list(args.in_data, args.in_data, args.old_file)
    # get_number_speaker(args.in_data, args.old_file)
