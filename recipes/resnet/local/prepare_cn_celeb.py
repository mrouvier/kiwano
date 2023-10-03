#!/usr/bin/python3

from kiwano.utils import Pathlike
from pathlib import Path
import torchaudio
from tqdm import tqdm
from concurrent.futures.process import ProcessPoolExecutor

import argparse
from subprocess import PIPE, run

def get_duration(file_path: str):
   info = torchaudio.info(file_path)
   return info.num_frames/info.sample_rate


def _process_file(file_path: Pathlike, output: Pathlike):
    cmd = "ffmpeg -threads 1 -i " + str(file_path) + " -acodec pcm_s16le -ac 1 -ar 16000 -ab 48 -threads 1 " + str(output)
    proc = run(cmd, shell=True, stdout=PIPE, stderr=PIPE)


def process_file_test(segment: Pathlike, in_data: Pathlike):
    name = "_".join(str(segment).split("/")[-3:]).split(".")[0]
    spkid = str(segment).split("/")[-1].split("-")[0]
    n = str(segment).split("/")[-1].split(".")[0]

    out = Path(in_data/ "CN-Celeb_flac" / "eval" / "wav")
    out.mkdir(parents=True, exist_ok=True)

    output = str(Path(in_data/ "CN-Celeb_flac" / "eval" / "wav"/ n)) + ".wav"

    _process_file(segment, Path(output))
    duration = str(round(float(get_duration(output)), 2))

    return name, spkid, duration, segment

def process_file_train(segment: Pathlike, in_data: Pathlike):
    name = "_".join(str(segment).split("/")[-3:]).split(".")[0]
    spkid = str(segment).split("/")[-2]
    n = str(segment).split("/")[-1].split(".")[0]

    out = Path(in_data / "CN-Celeb2_flac" / "wav" / spkid)
    out.mkdir(parents=True, exist_ok=True)

    output = str(Path(in_data / "CN-Celeb2_flac" / "wav" /spkid / n)) + ".wav"

    _process_file(segment, Path(output))
    duration = str(round(float(get_duration(output)), 2))


    return name, spkid, duration, segment


def prepare_cn_celeb(in_data: Pathlike, out_data: Pathlike):
    in_data = Path(in_data)

    out_data = Path(out_data)
    out_data.mkdir(parents=True, exist_ok=True)

    listeTrain = open(out_data / "listeTrain", "w")
    listeTest = open(out_data / "listeTest", "w")



    with ProcessPoolExecutor(20) as ex:
        futuresTrain = []
        futuresTest = []

        for segment in (in_data / "CN-Celeb2_flac" / "data").rglob("*.flac"):
            futuresTrain.append(ex.submit(process_file_train, segment, out_data))
        for segment in (in_data / "CN-Celeb_flac" / "eval").rglob("*.flac"):
            futuresTest.append(ex.submit(process_file_test, segment, out_data))

        for future in tqdm(futuresTrain, desc="Processing Cn Celeb 2"):
            name, spkid, duration, segment = future.result()
            listeTrain.write(f"{name} {spkid} {duration} {segment}\n")
        for future in tqdm(futuresTest, desc="Processing Cn Celeb"):
            name, spkid, duration, segment = future.result()
            listeTest.write(f"{name} {spkid} {duration} {segment}\n")


    listeTrain.close()
    listeTest.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('in_data', metavar='in_data', type=str,
                        help='the path to the directory where CN-Celeb2_flac and CN-Celeb_flac are stored')
    parser.add_argument('out_data', metavar="out_data", type=str,
                        help='the path to the target directory where the liste will be stored')

    args = parser.parse_args()

    prepare_cn_celeb(args.in_data, args.out_data)

