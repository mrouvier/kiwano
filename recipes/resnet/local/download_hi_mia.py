#!/usr/bin/env python3

import logging
import tarfile
import shutil
import sys
from kiwano.utils import Pathlike, urlretrieve_progress, check_md5, extract_tar
from pathlib import Path
from typing import Optional
import logging

import argparse


HI_MIA_PARTS_URL = [
    ["https://www.openslr.org/resources/85/train.tar.gz", "95aeffa182a7e9da668785babc64fc8b"],
    ["https://www.openslr.org/resources/85/dev.tar.gz", "f870cb7771412b173b93baac0cdf0f13"],
    ["https://www.openslr.org/resources/85/test_v2.tar.gz", "f49396aaea461cfd70bcaa906f53cc31"]
]



def download_hi_mia(target_dir: Pathlike = ".", force_download: Optional[bool] = False, max_retries: int = 3):
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    for retry in range(max_retries):
        missing_files = [file[0].split("/")[-1] for file in HI_MIA_PARTS_URL if not (target_dir / file[0].split("/")[-1]).exists()]
        invalid_files = []
        
        if not missing_files and not force_download:
            all_files_valid = True
            for url, md5 in HI_MIA_PARTS_URL:
                file_path = target_dir / url.split("/")[-1]
                if not check_md5(file_path, md5):
                    all_files_valid = False
                    invalid_files.append(file_path)
            
            if all_files_valid:
                logging.info("All files exist and have correct MD5 checksums. Skipping download.")
                return
            else:
                logging.info(f"Some files have incorrect MD5 checksums. Removing and re-downloading (Attempt {retry + 1}/{max_retries}).")
                for file in invalid_files:
                    file.unlink()
                missing_files.extend([f.name for f in invalid_files])
        
        for url, md5 in HI_MIA_PARTS_URL:
            fname = target_dir / url.split("/")[-1]
            if fname.name in missing_files or force_download:
                urlretrieve_progress(url, filename=fname, desc=f"Downloading Hi_Mia {fname.name}")
                if not check_md5(fname, md5):
                    logging.error(f"MD5 check failed for {fname}. Retrying...")
                    break
        else:
            break
    else:
        logging.error(f"Failed to download files with correct MD5 after {max_retries} attempts. Please try again later.")
        return

    logging.info(f"Unzipping Hi_Mia...")
    for url, _ in HI_MIA_PARTS_URL:
        extract_tar(target_dir / url.split("/")[-1], target_dir)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--force_download', action='store_true', default=False,
            help='Force the download, overwriting existing files (default: False)')
    parser.add_argument('--max_retries', type=int, default=3,
            help='Number of retries in case of failed download (default: 3)')
    parser.add_argument('target_dir', type=str, metavar='TARGET_DIR',
            help='Path to the target directory where the data will be stored')


    args = parser.parse_args()

    download_hi_mia(args.target_dir, args.force_download, args.max_retries)