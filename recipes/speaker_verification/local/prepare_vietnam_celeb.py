#!/usr/bin/python3

import argparse
from concurrent.futures.process import ProcessPoolExecutor
from pathlib import Path

import torchaudio
from tqdm import tqdm

from kiwano.utils import Pathlike


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

    listeTrain = open(out_data / "listeTrain", "w")
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

        for segment in (in_data / "data").rglob("*.wav"):
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "in_data",
        metavar="in_data",
        type=str,
        help='the path to the directory "data", where the wav files are stored',
    )
    parser.add_argument(
        "out_data",
        metavar="out_data",
        type=str,
        help="the path to the target directory where the liste will be stored",
    )

    args = parser.parse_args()

    prepare_vietnam_celeb(args.in_data, args.out_data)
