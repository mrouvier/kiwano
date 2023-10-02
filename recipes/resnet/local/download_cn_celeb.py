#!/usr/bin/env python3

import logging
import tarfile
import shutil
import sys
from kiwano.utils import Pathlike, urlretrieve_progress, check_md5
from pathlib import Path
from typing import Optional

import argparse



CN_CELEB_PARTS_URL = [
    ["https://www.openslr.org/resources/82/cn-celeb_v2.tar.gz", "7ab1b214028a7439e26608b2d5a0336c"],
    ["https://www.openslr.org/resources/82/cn-celeb2_v2.tar.gzaa", "4cdf738cff565ce35ad34274848659c3"],
    ["https://www.openslr.org/resources/82/cn-celeb2_v2.tar.gzab", "4ff59a7009b79ef7043498ad3882b81b"],
    ["https://www.openslr.org/resources/82/cn-celeb2_v2.tar.gzac", "3aa8e6e7f7ec4382f9926b1b31498228"]
]



def download_cn_celeb(target_dir: Pathlike = ".", force_download: Optional[bool] = False):
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    tar_gz_name = "cn-celeb2.tar.gz"
    tar_gz_path = target_dir / tar_gz_name

    if tar_gz_path.exists() and not force_download:
        check_md5(target_dir, CN_CELEB_PARTS_URL)
        logging.info(f"Skipping {tar_gz_name} because file exists.")
    else:

        for url in CN_CELEB_PARTS_URL:
            fname=target_dir / url[0].split("/")[-1]
            if not fname.exists() and not force_download:
                urlretrieve_progress(url[0], filename=target_dir / url[0].split("/")[-1], desc=f"Downloading VoxCeleb1 {url[0].split('/')[-1]}")
            elif force_download :
                urlretrieve_progress(url[0], filename=target_dir / url[0].split("/")[-1], desc=f"Downloading VoxCeleb1 {url[0].split('/')[-1]}")

        check_md5(target_dir, CN_CELEB_PARTS_URL)

        with open(tar_gz_path, "wb") as outFile:
            for file in sorted(target_dir.glob("cn-celeb2_v2.tar.gza*")):
                with open(file, "rb") as inFile:
                    shutil.copyfileobj(inFile, outFile)

        logging.info(f"Unzipping train...")
        with tarfile.open(tar_gz_path) as zf:
            zf.extractall(target_dir)


    logging.info(f"Unzipping test...")
    with tarfile.open(target_dir / "cn-celeb_v2.tar.gz") as zf:
        zf.extractall(target_dir )







if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('target_dir', metavar='target_dir', type=str,
                        help='the path to the target directory where the data will be stored')
    parser.add_argument('--force_download', action="store_true", default=False,
                        help='force the download, overwrites files (default: False)')

    args = parser.parse_args()

    download_cn_celeb(args.target_dir, args.force_download)




