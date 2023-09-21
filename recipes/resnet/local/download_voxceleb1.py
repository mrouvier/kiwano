#!/usr/bin/env python3

import logging
import zipfile
import shutil
import sys
from kiwano.utils import Pathlike, urlretrieve_progress
from pathlib import Path
from typing import Optional

import hashlib
import os

VOXCELEB1_PARTS_URL = [
    ["https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partaa", "e395d020928bc15670b570a21695ed96"],
    ["https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partab", "bbfaaccefab65d82b21903e81a8a8020"],
    ["https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partac", "017d579a2a96a077f40042ec33e51512"],
    ["https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partad", "7bb1e9f70fddc7a678fa998ea8b3ba19"],
    ["https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_test_wav.zip", "185fdc63c3c739954633d50379a3d102"]
]


VOXCELEB1_TRIALS_URL = [
    ["http://www.openslr.org/resources/49/voxceleb1_test_v2.txt", "29fc7cc1c5d59f0816dc15d6e8be60f7"]
]

VOXCELEB1_META_URL = [
    ["https://www.openslr.org/resources/49/vox1_meta.csv", "f5067cae5157ca27904c4b0a926e2661"]
]


def check_md5(dir, liste):
    """
    arg1 dir: the directory where the files in the liste are stocked
    arg2 liste: the name of the liste
    This function check if all the files in the list are downloaded correctly,
    if not, there are 3 attempts to re-download the file
    """

    for url in liste:
        fname = dir / url[0].split("/")[-1]

        for i in range(3):
            try:
                md5 = hashlib.md5(fname.read_bytes()).hexdigest()
                if md5 != url[1]:
                    raise ValueError()
                else:
                    print("File ", fname, " correctly downloaded")
                    break
            except ValueError:
                print("error downloading file ", fname)
                os.remove(fname)
                urlretrieve_progress(url[0], filename=dir / url[0].split("/")[-1], desc=f"Downloading VoxCeleb1 {url[0].split('/')[-1]}")

        else:
            if hashlib.md5(fname.read_bytes()).hexdigest() != url[1]:
                print("Download failed for file ", fname)
                os.remove(fname)
            else:
                print("File ", fname," finally correctly downloaded")



def download_voxceleb1(target_dir: Pathlike = ".", force_download: Optional[bool] = False):
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    zip_name = "vox1_dev_wav.zip"
    zip_path = target_dir / zip_name

    if zip_path.exists() and not force_download:
        logging.info(f"Skipping {zip_name} because file exists.")
    else:
        for url in VOXCELEB1_PARTS_URL:
            fname=target_dir / url[0].split("/")[-1]
            if not fname.exists() and not force_download:
                urlretrieve_progress(url[0], filename=target_dir / url[0].split("/")[-1], desc=f"Downloading VoxCeleb1 {url[0].split('/')[-1]}")

        with open(zip_path, "wb") as outFile:
            for file in sorted(target_dir.glob("vox1_dev_wav_part*")):
                with open(file, "rb") as inFile:
                    shutil.copyfileobj(inFile, outFile)

        logging.info(f"Unzipping dev...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(target_dir)

        logging.info(f"Unzipping test...")
        with zipfile.ZipFile(target_dir / "vox1_test_wav.zip") as zf:
            zf.extractall(target_dir)

    check_md5(target_dir, VOXCELEB1_PARTS_URL)

    for url in VOXCELEB1_TRIALS_URL:
        fname=target_dir / url[0].split("/")[-1]
        if not fname.exists() and not force_download:
            urlretrieve_progress(url[0], filename=target_dir / url[0].split("/")[-1], desc=f"Downloading VoxCeleb1 {url[0].split('/')[-1]}")

    check_md5(target_dir, VOXCELEB1_TRIALS_URL)

    for url in VOXCELEB1_META_URL:
        fname=target_dir / url[0].split("/")[-1]
        if not fname.exists() and not force_download:
            urlretrieve_progress(url[0], filename=target_dir / url[0].split("/")[-1], desc=f"Downloading VoxCeleb1 {url[0].split('/')[-1]}")

    check_md5(target_dir, VOXCELEB1_META_URL)

if __name__ == '__main__':
    download_voxceleb1(sys.argv[1])
