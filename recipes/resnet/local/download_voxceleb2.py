#!/bin/python3

import sys
import logging
import zipfile
import shutil
from kiwano.utils import Pathlike, urlretrieve_progress
from pathlib import Path
from typing import Optional


VOXCELEB2_PARTS_URL = [
    ["https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aac_partaa", "md5"],
    ["https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aac_partab", "md5"],
    ["https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aac_partac", "md5"],
    ["https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aac_partad", "md5"],
    ["https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aac_partae", "md5"],
    ["https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aac_partaf", "md5"],
    ["https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aac_partag", "md5"],
    ["https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aac_partah", "md5"]
]


VOXCELEB2_META_URL = [
    ["https://www.openslr.org/resources/49/vox2_meta.csv", "md5"]
]


def download_voxceleb2(target_dir: Pathlike = ".", force_download: Optional[bool] = False):
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    zip_name = "vox2_aac.zip"
    zip_path = target_dir / zip_name

    if zip_path.exists() and not force_download:
        logging.info(f"Skipping {zip_name} because file exists.")
    else:
        for url in VOXCELEB2_PARTS_URL:
            urlretrieve_progress(url[0], filename=target_dir / url[0].split("/")[-1], desc=f"Downloading VoxCeleb2 {url[0].split('/')[-1]}")

        with open(zip_path, "wb") as outFile:
            for file in sorted(target_dir.glob("vox2_dev_aac_part*")):
                with open(file, "rb") as inFile:
                    shutil.copyfileobj(inFile, outFile)    

        logging.info(f"Unzipping dev...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(target_dir)

    for url in VOXCELEB2_META_URL:
        fname=target_dir / url[0].split("/")[-1]
        if not fname.exists() and not force_download:
            urlretrieve_progress(url[0], filename=target_dir / url[0].split("/")[-1], desc=f"Downloading VoxCeleb2 {url[0].split('/')[-1]}")


        #logging.info(f"Unzipping test...")
        #with zipfile.ZipFile(target_dir / "vox2_test_aac.zip") as zf:
        #    zf.extractall(target_dir)


if __name__ == '__main__':
    download_voxceleb2(sys.argv[1])

