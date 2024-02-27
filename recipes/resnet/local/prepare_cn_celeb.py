#!/usr/bin/python3

from kiwano.utils import Pathlike
from pathlib import Path
import torchaudio
from tqdm import tqdm
from concurrent.futures.process import ProcessPoolExecutor

import argparse
from subprocess import PIPE, run

import os


def get_duration(file_path: str):
    info = torchaudio.info(file_path)
    return info.num_frames / info.sample_rate


def _process_file(file_path: Pathlike, output: Pathlike):
    cmd = "ffmpeg -threads 1 -i " + str(file_path) + " -acodec pcm_s16le -ac 1 -ar 16000 -ab 48 -threads 1 " + str(
        output)
    proc = run(cmd, shell=True, stdout=PIPE, stderr=PIPE)


def process_file_test(segment: Pathlike, in_data: Pathlike):
    name = "_".join(str(segment).split("/")[-3:]).split(".")[0]
    spkid = str(segment).split("/")[-1].split("-")[0]
    n = str(segment).split("/")[-1].split(".")[0]

    out = Path(in_data / "CN-Celeb_flac" / "eval" / "wav")
    out.mkdir(parents=True, exist_ok=True)

    output = str(Path(in_data / "CN-Celeb_flac" / "eval" / "wav" / n)) + ".wav"

    if not Path(output).exists():
        _process_file(segment, Path(output))
    duration = str(round(float(get_duration(output)), 2))

    toolkitPath = Path("db") / "cnceleb1" / "wav" / (n + ".wav")

    return name, spkid, duration, toolkitPath


def process_file_dev(segment: Pathlike, in_data: Pathlike):
    name = "_".join(str(segment).split("/")[-3:]).split(".")[0]
    spkid = str(segment).split("/")[4]
    n = str(segment).split("/")[-1].split(".")[0]

    out = Path(in_data / "CN-Celeb_flac" / "dev" / "wav" / spkid)
    out.mkdir(parents=True, exist_ok=True)

    output = str(Path(in_data / "CN-Celeb_flac" / "dev" / "wav" / spkid / n)) + ".wav"

    if not Path(output).exists():
        _process_file(segment, Path(output))
    duration = str(round(float(get_duration(output)), 2))

    toolkitPath = Path("db") / "cnceleb1_dev" / "wav" / spkid / (n + ".wav")

    return name, spkid, duration, toolkitPath


def process_file_train(segment: Pathlike, in_data: Pathlike):
    name = "_".join(str(segment).split("/")[-3:]).split(".")[0]
    spkid = str(segment).split("/")[-2]
    n = str(segment).split("/")[-1].split(".")[0]

    out = Path(in_data / "CN-Celeb2_flac" / "wav" / spkid)
    out.mkdir(parents=True, exist_ok=True)

    output = str(Path(in_data / "CN-Celeb2_flac" / "wav" / spkid / n)) + ".wav"

    if not Path(output).exists():
        _process_file(segment, Path(output))
    duration = str(round(float(get_duration(output)), 2))

    toolkitPath = Path("db") / "cnceleb2" / "wav" / spkid / (n + ".wav")

    return name, spkid, duration, toolkitPath


def prepare_trials(in_data: Pathlike):
    # change trials.lst to have the correct format

    dictEnroll = {}

    with open(in_data / "CN-Celeb_flac" / "eval" / "lists" / "enroll.lst", 'r') as enroll:
        for line in enroll:
            col1, col2 = line.strip().split(' ')
            dictEnroll[col1] = col2

    with open(in_data / "CN-Celeb_flac" / "eval" / "lists" / "trials.lst", 'r') as trials:
        with open(in_data / "CN-Celeb_flac" / "eval" / "lists" / "new_trials.txt", 'w') as new_trials:
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


def prepare_cn_celeb(canDeleteZIP: bool, in_data: Pathlike, out_data: Pathlike):
    in_data = Path(in_data)

    out_data = Path(out_data)
    out_data.mkdir(parents=True, exist_ok=True)

    listeTrain = open(out_data / "liste", "w")
    listeDev = open(out_data / "listeDev", "w")
    listeTest = open(out_data / "listeTest", "w")

    with ProcessPoolExecutor(20) as ex:
        futuresTrain = []
        futuresDev = []
        futuresTest = []

        for segment in (in_data / "CN-Celeb2_flac" / "data").rglob("*.flac"):
            futuresTrain.append(ex.submit(process_file_train, segment, out_data))
        for segment in (in_data / "CN-Celeb_flac" / "data").rglob("*.flac"):
            if int(segment.parts[4][2:]) < 800:
                futuresDev.append(ex.submit(process_file_dev, segment, out_data))

        for segment in (in_data / "CN-Celeb_flac" / "eval").rglob("*.flac"):
            futuresTest.append(ex.submit(process_file_test, segment, out_data))

        for future in tqdm(futuresTrain, desc="Processing Cn Celeb 2"):
            name, spkid, duration, segment = future.result()
            listeTrain.write(f"{name} {spkid} {duration} {segment}\n")

        for future in tqdm(futuresDev, desc="Processing Cn Celeb"):
            name, spkid, duration, segment = future.result()
            listeDev.write(f"{name} {spkid} {duration} {segment}\n")

        for future in tqdm(futuresTest, desc="Processing Cn Celeb"):
            name, spkid, duration, segment = future.result()
            listeTest.write(f"{name} {spkid} {duration} {segment}\n")

    listeTrain.close()
    listeDev.close()
    listeTest.close()

    prepare_trials(in_data)

    if canDeleteZIP:
        for file in sorted(in_data.glob("cn-celeb2_v2.tar*")):
            os.remove(file)

        os.remove(in_data / "cn-celeb2.tar.gz")
        os.remove(in_data / "cn-celeb_v2.tar.gz")


def create_new_eval_list(in_data: Pathlike, out_data: Pathlike, oldfile: str):
    in_data = Path(in_data)
    out_data = Path(out_data)
    out_data.mkdir(parents=True, exist_ok=True)

    listeEval = open(out_data / f"{oldfile}.edited", "w")
    with open(in_data / oldfile, "r") as f:
        for line in f:
            line = line.strip().split()
            if len(line) == 3:
                part0 = line[0].strip()
                part1 = line[1].strip()
                part2 = line[2].strip()

                speaker1 = f"{part0}.wav"
                speaker2 = part1.split("/")[1]
                label = part2

                listeEval.write(f"{label} {speaker1} {speaker2}\n")

    listeEval.close()


def create_new_train_list(in_data: Pathlike, out_data: Pathlike, oldfile: str):
    in_data = Path(in_data)
    out_data = Path(out_data)
    out_data.mkdir(parents=True, exist_ok=True)

    listeTrain = open(out_data / f"{oldfile}.edited", "w")
    with open(in_data / oldfile, "r") as f:
        for line in f:
            line = line.strip().split()
            if len(line) == 4:
                part1 = line[1].strip()
                part3 = line[-1].strip()
                speaker = "/".join(part3.split('/')[-2:])
                if os.path.exists(part3):
                    listeTrain.write(f"{part1} {speaker}\n")

    listeTrain.close()


def get_number_speaker(in_data: Pathlike, fname: str):
    speaker_ids = []
    in_data = Path(in_data)
    with open(in_data / fname, "r") as f:
        for line in f:
            line = line.strip().split()
            spkId = line[0].strip()
            speaker_ids.append(spkId)

    speaker_ids = set(speaker_ids)
    print(f"Cn_celeb - Number of speaker in {fname}: {len(speaker_ids)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_data', metavar='in_data', type=str,
                        help='the path to the directory where CN-Celeb2_flac and CN-Celeb_flac are stored')
    parser.add_argument('--out_data', metavar="out_data", type=str,
                        help='the path to the target directory where the liste will be stored')
    parser.add_argument('--deleteZIP', action="store_true", default=False,
                        help='to delete the ZIP files already extracted (default: False)')
    parser.add_argument('--old_file', metavar="old_file", type=str,
                        help='old file name')

    args = parser.parse_args()

    # prepare_cn_celeb(args.deleteZIP, args.in_data, args.out_data)
    # create_new_eval_list(args.in_data, args.in_data, args.old_file)
    create_new_train_list(args.in_data, args.out_data, args.old_file)
    # get_number_speaker(args.in_data, args.old_file)
