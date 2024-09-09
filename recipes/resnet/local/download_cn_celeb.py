#!/usr/bin/env python3

import logging
import tarfile
import shutil
import sys
import os
from kiwano.utils import Pathlike, urlretrieve_progress, check_md5, extract_tar
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

    cn_celeb2_tar_gz = "cn-celeb2.tar.gz"
    cn_celeb2_tar_gz_path = target_dir / cn_celeb2_tar_gz
    cn_celeb1_tar_gz_path = target_dir / "cn-celeb_v2.tar.gz"

    missing_files = [file.split("/")[-1] for file, _ in CN_CELEB_PARTS_URL if not (target_dir / file.split("/")[-1]).exists()]
    if not missing_files and not force_download:
        check_md5(target_dir, CN_CELEB_PARTS_URL)
        if missing_files:
            logging.info(f"Missing files: {', '.join(missing_files)}")
        else:
            logging.info("All files exist. Skipping download.")
    else:
        for url, md5 in CN_CELEB_PARTS_URL:
            fname = target_dir / url.split("/")[-1]
            if fname.name in missing_files or force_download:
                urlretrieve_progress(url, filename=fname, desc=f"Downloading CN_Celeb {fname.name}")

        check_md5(target_dir, CN_CELEB_PARTS_URL, "Cn_Celeb")

        if any(f"cn-celeb2_v2.tar.gz{part}" in missing_files for part in ['aa', 'ab', 'ac']) or force_download:
            logging.info("Building tar archive for CN_Celeb2...")
            with open(cn_celeb2_tar_gz_path, "wb") as outFile:
                for file in sorted(target_dir.glob("cn-celeb2_v2.tar.gza*")):
                    with open(file, "rb") as inFile:
                        shutil.copyfileobj(inFile, outFile)
            logging.info("Tar archive built, cleaning up...")

            for part in ["aa", "ab", "ac"]:
                os.remove(target_dir / f"cn-celeb2_v2.tar.gz{part}")

            logging.info(f"Unzipping CN_Celeb2 (train)...")
            extract_tar(cn_celeb2_tar_gz_path, target_dir)

        if "cn-celeb_v2.tar.gz" in missing_files or force_download:
            logging.info(f"Unzipping CN_Celeb1 (dev and test)...")
            extract_tar(cn_celeb1_tar_gz_path, target_dir)
            







if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('target_dir', metavar='target_dir', type=str,
                        help='the path to the target directory where the data will be stored')
    parser.add_argument('--force_download', action="store_true", default=False,
                        help='force the download, overwrites files (default: False)')

    args = parser.parse_args()

    download_cn_celeb(args.target_dir, args.force_download)








