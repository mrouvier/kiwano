#!/usr/bin/env python3

import logging
import zipfile
import shutil
import sys
from kiwano.utils import Pathlike, urlretrieve_progress
from pathlib import Path
from typing import Optional


VOXCELEB1_PARTS_URL = [
    ["https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partaa", "md5"],
    ["https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partab", "md5"],
    ["https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partac", "md5"],
    ["https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partad", "md5"],
    ["https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_test_wav.zip", "md5"],
    ["https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/vox1_meta.csv", "md5"]
]


VOXCELEB1_TRIALS_URL = [
    ["http://www.openslr.org/resources/49/voxceleb1_test_v2.txt", "md5"]
]

VOXCELEB1_META_URL = [
    ["https://www.openslr.org/resources/49/vox1_meta.csv", "md5"]
]




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

    for url in VOXCELEB1_TRIALS_URL:
        fname=target_dir / url[0].split("/")[-1]
        if not fname.exists() and not force_download:
            urlretrieve_progress(url[0], filename=target_dir / url[0].split("/")[-1], desc=f"Downloading VoxCeleb1 {url[0].split('/')[-1]}")

    for url in VOXCELEB1_META_URL:
        fname=target_dir / url[0].split("/")[-1]
        if not fname.exists() and not force_download:
            urlretrieve_progress(url[0], filename=target_dir / url[0].split("/")[-1], desc=f"Downloading VoxCeleb1 {url[0].split('/')[-1]}")



if __name__ == '__main__':
    download_voxceleb1(sys.argv[1])
