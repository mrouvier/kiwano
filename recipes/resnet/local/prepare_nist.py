#!/usr/bin/env python3
import shutil
import sys
import glob

from torch.utils.data import Dataset, DataLoader

from kiwano.utils import Pathlike
from pathlib import Path
import soundfile as sf
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


def process_file(segment: Pathlike, out_data: Pathlike, sampling_frequency: int):
    # db/nist/nist-sre-test2004/xeot.sph
    # name = segment.name.replace(".sph", ".wav")

    output = Path(out_data) / segment.name

    if not output.exists():
        _process_file(segment, output, sampling_frequency)


def convert_sph_to_wav_nist(sampling_frequency: int, canDeleteZIP: bool, in_data: Pathlike = ".",
                            out_data: Pathlike = ".",
                            num_jobs: int = 20):
    out_data = Path(out_data)
    out_data.mkdir(parents=True, exist_ok=True)

    with ProcessPoolExecutor(num_jobs) as ex:
        futures = []

        for segment in Path(in_data).rglob("*.wav"):
            futures.append(ex.submit(process_file, segment, out_data, sampling_frequency))

        for future in tqdm(futures, total=len(futures), desc=f"Processing Nist ... "):
            future.result()

    if canDeleteZIP:
        for file in tqdm(sorted(in_data.glob("*.wav")), desc="Deletion "):
            os.remove(file)


def _process_file(file_path: Pathlike, output: Pathlike, sampling_frequency: int):
    # ffmpeg -v 8 -i {segment} -f wav -acodec pcm_s16le - |
    cmd = "ffmpeg -y -threads 1 -i " + str(file_path) + " -acodec pcm_s16le -ac 1 -ar " + str(
        sampling_frequency) + " -ab 48 -threads 1 " + str(output)
    proc = run(cmd, shell=True, stdout=PIPE, stderr=PIPE)
    print(cmd, flush=True)

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
    print(f"NIST - Number of speaker in {fname}: {len(speaker_ids)}", flush=True)


def change_sph_ext_to_wav(in_data: Pathlike, fname: str):
    in_data = Path(in_data)
    with open(in_data / fname, "r") as f:
        lines = f.readlines()

    lines = [line.replace('.sph', '.wav') for line in lines if len(line.strip()) > 0]
    with open(in_data / fname, "w") as f:
        f.writelines(lines)


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
                fname = f"{line[2].strip()}.sph"
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
        for i, line in tqdm(enumerate(lines), total=len(lines)):
            line = line.strip().split()
            if len(line) == 8:
                idspk1 = line[1].strip()
                file1 = f"{line[2].strip()}.sph"
                path1 = in_data / file1
                others = lines[(i + 1):]
                for line2 in others:
                    line2 = line2.strip().split()
                    if len(line2) == 8:
                        idspk2 = line2[1].strip()
                        file2 = f"{line2[2].strip()}.sph"
                        path2 = in_data / file2
                        label = 1 if idspk1 == idspk2 else 0
                        if path1.exists() and path2.exists():
                            listeEval.write(f"{label} {file1} {file2}\n")
                        else:
                            print(f"{label} {path1} {path2}", flush=True)

    listeEval.close()


class ConvertDataset(Dataset):
    def __init__(self, files, out_data):
        self.files = files
        self.out_data = Path(out_data)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        in_path = self.files[i]
        old_name = in_path.name
        file = str(in_path)
        data, sr = sf.read(file)
        new_name = old_name.replace('.sph', '.wav')
        out_path = self.out_data / new_name
        sf.write(out_path, data, samplerate=sr)
        return file


def custom_convert_sph_to_wav(in_data: Pathlike, out_data: Pathlike):
    print(f"Path: {in_data}", flush=True)
    shutil.copy(Path(in_data) / "MASTER", Path(out_data) / "MASTER")
    files = list(Path(in_data).rglob("*.sph"))
    dataset = ConvertDataset(files, out_data)
    loader = DataLoader(dataset, batch_size=32, drop_last=False, num_workers=8)
    n_batch = len(loader)
    for i, batch in tqdm(enumerate(loader), total=n_batch):
        print(f"Batch  [{i + 1}/{n_batch}] done", flush=True)


class ExtractDataset(Dataset):
    def __init__(self, lines, in_data, out_data):
        self.lines = lines
        self.in_data = Path(in_data)
        self.out_data = Path(out_data)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, i):
        line = self.lines[i]
        parts = line.strip().split()
        if len(parts) != 8:
            return ""
        fname = parts[2].strip()
        path = self.in_data / f"{fname}.wav"
        channel = parts[3].strip()
        data, sr = sf.read(path)
        nchannels = data.shape[1]
        int_channel = 0 if channel == 'a' else 1
        if 0 <= int_channel < nchannels:
            channel_data = data[:, int_channel]
            new_fname = f"{fname}_{channel}.wav"
            new_path = self.out_data / new_fname
            sf.write(new_path, channel_data, sr)
        else:
            print("Ce cannal n'existe pas dans l'audio", flush=True)
        return str(path)


class DeleteDataset(Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        file = self.files[i]
        os.remove(file)
        return file


def extract_channel(in_data: Pathlike, out_data: Pathlike):
    print(f"Path: {in_data}", flush=True)
    in_data = Path(in_data)
    out_data = Path(out_data)
    out_data.mkdir(parents=True, exist_ok=True)
    master = in_data / "MASTER"
    to_delete = []
    with open(master, mode="r") as mfile:
        lines = mfile.readlines()
        dataset = ExtractDataset(lines, in_data, out_data)
        loader = DataLoader(dataset, num_workers=8, drop_last=False, batch_size=32)
        n_batch = len(loader)
        for i, batch in tqdm(enumerate(loader), total=n_batch, desc="Channel: "):
            print(f"Batch  [{i + 1}/{n_batch}] done", flush=True)
            to_delete.extend(list(batch))

        to_delete = list(set(to_delete))
        to_delete = [f.strip() for f in to_delete if len(f.strip()) > 0]
        dataset = DeleteDataset(to_delete)
        loader = DataLoader(dataset, num_workers=8, drop_last=False, batch_size=32)
        n_batch = len(loader)
        for i, batch in tqdm(enumerate(loader), total=n_batch, desc="Delete: "):
            print(f"Batch  [{i + 1}/{n_batch}] done", flush=True)

        with open(in_data / "liste.txt", mode="w") as lfile:
            for line in tqdm(lines, total=len(lines), desc="Liste: "):
                parts = line.strip().split()
                if len(parts) != 8:
                    continue
                speaker = parts[1].strip()
                fname = parts[2].strip()
                channel = parts[3].strip()
                fdir = in_data.parts[-1]
                fname = f"{fdir}/wav/{fname}_{channel}.wav"
                lfile.write(f"{speaker} {fname}\n")


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
    # convert_sph_to_wav_nist(args.downsampling, args.deleteZIP, Path(args.in_data), Path(args.out_data), 8)
    # get_number_speaker(args.in_data, args.old_file)
    # create_new_train_list(args.in_data, args.out_data, args.old_file)
    # create_new_eval_list(args.in_data, args.out_data, args.old_file)
    # change_sph_ext_to_wav(args.in_data, args.old_file)
    custom_convert_sph_to_wav(args.in_data, args.out_data)
    # extract_channel(args.in_data, args.out_data)
