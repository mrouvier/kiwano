#!/usr/bin/env python3

import logging
import zipfile
import shutil
import sys
from kiwano.utils import Pathlike, urlretrieve_progress
from pathlib import Path
from typing import Optional


RIRS_NOISES_PARTS_URL = [
    ["http://www.openslr.org/resources/28/rirs_noises.zip", "md5"],
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

        logging.info(f"Unzipping RIRS NOISES...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(target_dir)

if __name__ == '__main__':
    download_rirs_noises(sys.argv[1])
