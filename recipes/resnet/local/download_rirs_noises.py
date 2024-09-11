#!/usr/bin/env python3

import logging
import shutil
import sys
from kiwano.utils import Pathlike, urlretrieve_progress, check_md5, parallel_unzip
from pathlib import Path
from typing import Optional

import argparse

RIRS_NOISES_PARTS_URL = [
        ["https://openslr.elda.org/resources/28/rirs_noises.zip", "e6f48e257286e05de56413b4779d8ffb"],
]

def download_rirs_noises(target_dir: Pathlike = ".", force_download: Optional[bool] = False, do_check_md5: Optional[bool] = False, jobs: int = 10):
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    zip_name = "rirs_noises.zip"
    zip_path = target_dir / zip_name

    if zip_path.exists() and not force_download:
        logging.info(f"Skipping {zip_name} because file exists.")
    else:
        for url, md5 in RIRS_NOISES_PARTS_URL:
            fname = target_dir / url.split("/")[-1]
            if not fname.exists() or force_download:
                urlretrieve_progress(url, filename=fname, desc=f"Downloading RIRS NOISES {fname.name}")
            
            if do_check_md5:
                if not check_md5(fname, md5):
                    logging.warning(f"MD5 check failed for {fname}.")
                else:
                    logging.info(f"MD5 check passed for {fname}.")

        logging.info(f"Unzipping RIRS NOISES...")
        parallel_unzip(zip_path, target_dir, jobs)

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

    download_rirs_noises(args.target_dir, args.force_download, args.check_md5, args.thread)

