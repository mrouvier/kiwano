#!/usr/bin/env python3

import logging
import zipfile
import shutil
import sys
from kiwano.utils import Pathlike, urlretrieve_progress, check_md5
from pathlib import Path
from typing import Optional

import argparse

RIRS_NOISES_PARTS_URL = [
    ["http://www.openslr.org/resources/28/rirs_noises.zip", "e6f48e257286e05de56413b4779d8ffb"],
]

def download_rirs_noises(target_dir: Pathlike = ".", force_download: Optional[bool] = False):
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    zip_name = "rirs_noises.zip"
    zip_path = target_dir / zip_name

    if zip_path.exists() and not force_download:
        logging.info(f"Skipping {zip_name} because file exists.")
    else:
        for url in RIRS_NOISES_PARTS_URL:
            fname=target_dir / url[0].split("/")[-1]
            if not fname.exists() and not force_download:
                urlretrieve_progress(url[0], filename=target_dir / url[0].split("/")[-1], desc=f"Downloading RIRS NOISES {url[0].split('/')[-1]}")
            elif force_download :
                urlretrieve_progress(url[0], filename=target_dir / url[0].split("/")[-1], desc=f"Downloading RIRS NOISES {url[0].split('/')[-1]}")

        logging.info(f"Unzipping RIRS NOISES...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(target_dir)

    check_md5(target_dir, RIRS_NOISES_PARTS_URL)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('target_dir', metavar='target_dir', type=str,
                        help='the path to the target directory where the data will be stored')
    parser.add_argument('--force_download', action="store_true", default=False,
                        help='force the download, overwrites files (default: False)')

    args = parser.parse_args()

    download_rirs_noises(args.target_dir, args.force_download)

