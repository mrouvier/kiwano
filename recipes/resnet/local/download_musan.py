#!/usr/bin/env python3

import logging
import tarfile
import shutil
import sys
from kiwano.utils import Pathlike, urlretrieve_progress
from pathlib import Path
from typing import Optional


MUSAN_PARTS_URL = [
    ["https://www.openslr.org/resources/17/musan.tar.gz", "md5"],
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

        logging.info(f"Unzipping MUSAN...")
        with tarfile.open(zip_path) as zf:
            zf.extractall(target_dir)

if __name__ == '__main__':
    download_musan(sys.argv[1])
