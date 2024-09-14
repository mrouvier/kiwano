#!/usr/bin/env python3

import logging
import shutil
import sys
from kiwano.utils import Pathlike, urlretrieve_progress, check_md5, extract_tar
from pathlib import Path
from typing import Optional

import argparse

MUSAN_PARTS_URL = [
    ["https://www.openslr.org/resources/17/musan.tar.gz", "0c472d4fc0c5141eca47ad1ffeb2a7df"],
]

def download_musan(target_dir: Pathlike = ".", force_download: Optional[bool] = False, do_check_md5: Optional[bool] = False, jobs: int = 10):
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    zip_name = "musan.tar.gz"
    zip_path = target_dir / zip_name


    if zip_path.exists() and not force_download:
        logging.info(f"Skipping {zip_name} because file exists.")
    else:
        for url, md5 in MUSAN_PARTS_URL:
            fname=target_dir / url.split("/")[-1]
            if not fname.exists() and not force_download:
                urlretrieve_progress(url, filename=target_dir / url.split("/")[-1], desc=f"Downloading MUSAN {url.split('/')[-1]}")
            elif force_download :
                urlretrieve_progress(url, filename=target_dir / url.split("/")[-1], desc=f"Downloading MUSAN {url.split('/')[-1]}")


            if do_check_md5:
                file_path = target_dir / url.split("/")[-1]
                if not check_md5(file_path, md5):
                    logging.warning(f"MD5 check failed for {file_path}.")
                else:
                    logging.info(f"MD5 check passed for {file_path}.")
                    
        logging.info(f"Unzipping MUSAN...")
        extract_tar(zip_path, target_dir)


    

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

    download_musan(args.target_dir, args.force_download, args.check_md5, args.thread)

