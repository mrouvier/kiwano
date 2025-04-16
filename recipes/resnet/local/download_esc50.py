#!/usr/bin/env python3

import logging
import shutil
import sys
from kiwano.utils import Pathlike, urlretrieve_progress, check_md5, parallel_unzip
from pathlib import Path
from typing import Optional

import argparse

ESC50_PARTS_URL = [
    ["https://github.com/karoldvl/ESC-50/archive/master.zip", "629e8e9ebc1592bcceb06da6bec40275"],
]

def download_esc50(target_dir: Pathlike = ".", force_download: Optional[bool] = False, check_md5: Optional[bool] = False, num_jobs: int = 10):
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    zip_name = "master.zip"
    zip_path = target_dir / zip_name


    if zip_path.exists() and not force_download:
        logging.info(f"Skipping {zip_name} because file exists.")
    else:
        for url in ESC50_PARTS_URL:
            fname=target_dir / url[0].split("/")[-1]
            if not fname.exists() and not force_download:
                urlretrieve_progress(url[0], filename=target_dir / url[0].split("/")[-1], desc=f"Downloading ESC50 {url[0].split('/')[-1]}")
            elif force_download :
                urlretrieve_progress(url[0], filename=target_dir / url[0].split("/")[-1], desc=f"Downloading ESC50 {url[0].split('/')[-1]}")

        logging.info(f"Unzipping ESC50...")
        parallel_unzip(zip_path, target_dir, num_jobs)


    if check_md5:
        check_md5(target_dir, ESC50_PARTS_URL)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_jobs', type=int, default=30,
            help='Number of parallel jobs (default: 30)')
    parser.add_argument('--force_download', action='store_true', default=False,
            help='Force the download, overwriting existing files (default: False)')
    parser.add_argument('--check_md5', action='store_true', default=False,
            help='Verify MD5 checksums of the files (default: False)')
    parser.add_argument('target_dir', type=str, metavar='TARGET_DIR',
            help='Path to the target directory where the data will be stored')


    args = parser.parse_args()

    download_esc50(args.target_dir, args.force_download, args.check_md5, args.num_jobs)
