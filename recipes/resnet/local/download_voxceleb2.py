#!/usr/bin/env python3

import sys
import logging
import shutil
from kiwano.utils import Pathlike, urlretrieve_progress, check_md5, parallel_unzip, copy_files
from pathlib import Path
from typing import Optional

import argparse

VOXCELEB2_PARTS_URL = [
    ["http://drbenchmark.univ-avignon.fr/corpus/voxceleb2/vox1a/vox2_dev_aac_partaa", "da070494c573e5c0564b1d11c3b20577"],
    ["http://drbenchmark.univ-avignon.fr/corpus/voxceleb2/vox1a/vox2_dev_aac_partab", "17fe6dab2b32b48abaf1676429cdd06f"],
    ["http://drbenchmark.univ-avignon.fr/corpus/voxceleb2/vox1a/vox2_dev_aac_partac", "1de58e086c5edf63625af1cb6d831528"],
    ["http://drbenchmark.univ-avignon.fr/corpus/voxceleb2/vox1a/vox2_dev_aac_partad", "5a043eb03e15c5a918ee6a52aad477f9"],
    ["http://drbenchmark.univ-avignon.fr/corpus/voxceleb2/vox1a/vox2_dev_aac_partae", "cea401b624983e2d0b2a87fb5d59aa60"],
    ["http://drbenchmark.univ-avignon.fr/corpus/voxceleb2/vox1a/vox2_dev_aac_partaf", "fc886d9ba90ab88e7880ee98effd6ae9"],
    ["http://drbenchmark.univ-avignon.fr/corpus/voxceleb2/vox1a/vox2_dev_aac_partag", "d160ecc3f6ee3eed54d55349531cb42e"],
    ["http://drbenchmark.univ-avignon.fr/corpus/voxceleb2/vox1a/vox2_dev_aac_partah", "6b84a81b9af72a9d9eecbb3b1f602e65"]
]


VOXCELEB2_META_URL = [
    ["https://www.openslr.org/resources/49/vox2_meta.csv", "6090d767f8334733dfe4c6578fa725c2"]
]

def download_voxceleb2(target_dir: Pathlike = ".", force_download: Optional[bool] = False, check_md5: Optional[bool] = False, jobs: int = 10):
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    zip_name = "vox2_aac.zip"
    zip_path = target_dir / zip_name


    if zip_path.exists() and not force_download:
        logging.info(f"Skipping {zip_name} because file exists.")
    else:
        for url in VOXCELEB2_PARTS_URL:
            fname = target_dir / url[0].split("/")[-1]
            if not fname.exists() and not force_download:
                urlretrieve_progress(url[0], filename=target_dir / url[0].split("/")[-1], desc=f"Downloading VoxCeleb2 {url[0].split('/')[-1]}")


        copy_files(zip_path, target_dir, "vox2_dev_aac_part*")

        logging.info(f"Unzipping dev...")
        parallel_unzip(zip_path, target_dir, jobs)

        if check_md5:
            check_md5(target_dir, VOXCELEB2_PARTS_URL, "VoxCeleb2")

    for url in VOXCELEB2_META_URL:
        fname=target_dir / url[0].split("/")[-1]
        if not fname.exists() or force_download:
            urlretrieve_progress(url[0], filename=target_dir / url[0].split("/")[-1], desc=f"Downloading VoxCeleb2 {url[0].split('/')[-1]}")


    if check_md5:
        check_md5(target_dir, VOXCELEB2_META_URL)


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

    download_voxceleb2(args.target_dir, args.force_download, args.check_md5, args.thread)

