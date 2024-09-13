#!/usr/bin/env python3

import logging
import zipfile
import shutil
import sys
import os
from tqdm import tqdm
from kiwano.utils import Pathlike, gdrive_download, check_md5, extract_tar, copy_files
from pathlib import Path
from typing import Optional

import argparse

VOXMOVIES_URL = [
    ["https://drive.google.com/uc?export=download&id=1K-CSOE3IJyhaq--wkj1RhZIMND4qdgri&authuser=1&confirm=t", "de90dc74430a1f65157d9b5505da7799"]
]

def delete_dot_underscore_files(directory: Path, verbose: bool = False):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.startswith("._"):
                file_path = Path(root) / file
                try:
                    file_path.unlink() 
                    if verbose:
                        logging.info(f"Deleted {file_path}")
                except Exception as e:
                    logging.error(f"Failed to delete {file_path}: {e}")

def download_voxmovies(target_dir: Pathlike = ".", force_download: Optional[bool] = False, do_check_md5: Optional[bool] = False, jobs: int = 10):
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    zip_name = "vox_movies.tar.gz"
    zip_path = target_dir / zip_name

    if zip_path.exists() and not force_download:
        logging.info(f"Skipping {zip_name} because file exists.")
    else:
        for url in VOXMOVIES_URL:
            gdrive_download(url[0], str(zip_path))

        logging.info("Unzipping VoxMovies...")
        extract_tar(tar_gz_path=zip_path, extract_path=target_dir)

        logging.info("Cleaning up metadata files...")
        delete_dot_underscore_files(target_dir)


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

    download_voxmovies(args.target_dir, args.force_download, args.check_md5, args.thread)