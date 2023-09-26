#!/usr/bin/env python3

import logging
import tarfile
import shutil
import sys
from kiwano.utils import Pathlike, urlretrieve_progress, check_md5
from pathlib import Path
from typing import Optional

import argparse

MUSAN_PARTS_URL = [
    ["https://www.openslr.org/resources/17/musan.tar.gz", "0c472d4fc0c5141eca47ad1ffeb2a7df"],
]

def download_musan(target_dir: Pathlike = ".", force_download: Optional[bool] = False):
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    zip_name = "musan.tar.gz"
    zip_path = target_dir / zip_name

    if zip_path.exists() and not force_download:
        logging.info(f"Skipping {zip_name} because file exists.")
    else:
        for url in MUSAN_PARTS_URL:
            fname=target_dir / url[0].split("/")[-1]
            if not fname.exists() and not force_download:
                urlretrieve_progress(url[0], filename=target_dir / url[0].split("/")[-1], desc=f"Downloading MUSAN {url[0].split('/')[-1]}")
            elif force_download :
                urlretrieve_progress(url[0], filename=target_dir / url[0].split("/")[-1], desc=f"Downloading MUSAN {url[0].split('/')[-1]}")

        logging.info(f"Unzipping MUSAN...")
        with tarfile.open(zip_path) as zf:
            zf.extractall(target_dir)

    check_md5(target_dir, MUSAN_PARTS_URL)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('target_dir', metavar='target_dir', type=str,
                        help='the path to the target directory where the data will be stored')
    parser.add_argument('--force_download', action="store_true", default=False,
                        help='force the download, overwrites files (default: False)')

    args = parser.parse_args()

    download_musan(args.target_dir, args.force_download)

