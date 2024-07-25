#!/usr/bin/env python3

import logging
import zipfile
import shutil
import sys
import os
from tqdm import tqdm
from kiwano.utils import Pathlike, urlretrieve_progress, check_md5, parallel_unzip, copy_files
from pathlib import Path
from typing import Optional

import argparse



VOXCELEB1_PARTS_URL = [
    ["http://drbenchmark.univ-avignon.fr/corpus/voxceleb1/vox1a/vox1_dev_wav_partaa", "e395d020928bc15670b570a21695ed96"],
    ["http://drbenchmark.univ-avignon.fr/corpus/voxceleb1/vox1a/vox1_dev_wav_partab", "bbfaaccefab65d82b21903e81a8a8020"],
    ["http://drbenchmark.univ-avignon.fr/corpus/voxceleb1/vox1a/vox1_dev_wav_partac", "017d579a2a96a077f40042ec33e51512"],
    ["http://drbenchmark.univ-avignon.fr/corpus/voxceleb1/vox1a/vox1_dev_wav_partad", "7bb1e9f70fddc7a678fa998ea8b3ba19"],
    ["http://drbenchmark.univ-avignon.fr/corpus/voxceleb1/vox1a/vox1_test_wav.zip", "185fdc63c3c739954633d50379a3d102"]
]


VOXCELEB1_TRIALS_URL = [
    ["http://drbenchmark.univ-avignon.fr/corpus/voxceleb1/meta/veri_test2.txt", "b73110731c9223c1461fe49cb48dddfc"], #voxceleb1-o cleaned
    ["http://drbenchmark.univ-avignon.fr/corpus/voxceleb1/meta/list_test_hard2.txt", "857790e09d579a68eb2e339a090343c8"], #voxceleb1-h cleaned
    ["http://drbenchmark.univ-avignon.fr/corpus/voxceleb1/meta/list_test_all2.txt", "a53e059deb562ffcfc092bf5d90d9f3a"] #voxceleb1-e cleaned
]

VOXCELEB1_META_URL = [
    ["https://www.openslr.org/resources/49/vox1_meta.csv", "f5067cae5157ca27904c4b0a926e2661"]
]



def download_voxceleb1(target_dir: Pathlike = ".", force_download: Optional[bool] = False, check_md5: Optional[bool] = False, jobs: int = 10):
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
            elif force_download :
                urlretrieve_progress(url[0], filename=target_dir / url[0].split("/")[-1], desc=f"Downloading VoxCeleb1 {url[0].split('/')[-1]}")


        logging.info(f"Concatenating files...")
        copy_files(zip_path, target_dir, "vox1_dev_wav_part*")

        logging.info(f"Extracting zip...")
        parallel_unzip(zip_path, target_dir, jobs)

        logging.info(f"Extracting zip...")
        parallel_unzip(target_dir / "vox1_test_wav.zip", target_dir, jobs)


    if check_md5:
        check_md5(target_dir, VOXCELEB1_PARTS_URL)


    for url in VOXCELEB1_TRIALS_URL:
        fname=target_dir / url[0].split("/")[-1]
        if not fname.exists() and not force_download:
            urlretrieve_progress(url[0], filename=target_dir / url[0].split("/")[-1], desc=f"Downloading VoxCeleb1 {url[0].split('/')[-1]}")
        elif force_download:
            urlretrieve_progress(url[0], filename=target_dir / url[0].split("/")[-1], desc=f"Downloading VoxCeleb1 {url[0].split('/')[-1]}")

    if check_md5:
        check_md5(target_dir, VOXCELEB1_TRIALS_URL)

    for url in VOXCELEB1_META_URL:
        fname=target_dir / url[0].split("/")[-1]
        if not fname.exists() and not force_download:
            urlretrieve_progress(url[0], filename=target_dir / url[0].split("/")[-1], desc=f"Downloading VoxCeleb1 {url[0].split('/')[-1]}")
        elif force_download:
            urlretrieve_progress(url[0], filename=target_dir / url[0].split("/")[-1], desc=f"Downloading VoxCeleb1 {url[0].split('/')[-1]}")

    if check_md5:
        check_md5(target_dir, VOXCELEB1_META_URL)




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--thread', type=int, default=10,
            help='Number of parallel jobs (default: 10)')
    parser.add_argument('--force_download', action='store_true', default=False,
            help='Force the download, overwriting existing files (default: False)')
    parser.add_argument('--check_md5', action='store_true', default=False,
            help='Verify MD5 checksums of the files (default: False)')
    parser.add_argument('target_dir', type=str, metavar='TARGET_DIR',
            help='Path to the target directory where the data will be stored')

    args = parser.parse_args()

    download_voxceleb1(args.target_dir, args.force_download, args.check_md5, args.thread)

