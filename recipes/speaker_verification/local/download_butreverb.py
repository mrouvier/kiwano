#!/usr/bin/env python3

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import Optional

from kiwano.utils import Pathlike, check_md5, extract_tar, urlretrieve_progress

BUT_REVERB_PARTS_URL = [
    [
        "http://merlin.fit.vutbr.cz/ReverbDB/BUT_ReverbDB_rel_19_06_RIR-Only.tgz",
        "3ad91777f6cadec81fb4c07dfbef6f19",
    ],
]


def download_butreverb(
    target_dir: Pathlike = ".",
    force_download: Optional[bool] = False,
    check_md5: Optional[bool] = False,
    num_jobs: int = 10,
):
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    zip_name = "BUT_ReverbDB_rel_19_06_RIR-Only.tgz"
    zip_path = target_dir / zip_name

    if zip_path.exists() and not force_download:
        logging.info(f"Skipping {zip_name} because file exists.")
    else:
        for url in BUT_REVERB_PARTS_URL:
            fname = target_dir / url[0].split("/")[-1]
            if not fname.exists() and not force_download:
                urlretrieve_progress(
                    url[0],
                    filename=target_dir / url[0].split("/")[-1],
                    desc=f"Downloading BUT REVERB {url[0].split('/')[-1]}",
                )
            elif force_download:
                urlretrieve_progress(
                    url[0],
                    filename=target_dir / url[0].split("/")[-1],
                    desc=f"Downloading BUT REVERB {url[0].split('/')[-1]}",
                )

        logging.info(f"Unzipping BUT REVERB...")
        extract_tar(zip_path, target_dir)

    if check_md5:
        check_md5(target_dir, BUT_REVERB_PARTS_URL)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_jobs", type=int, default=30, help="Number of parallel jobs (default: 30)"
    )
    parser.add_argument(
        "--force_download",
        action="store_true",
        default=False,
        help="Force the download, overwriting existing files (default: False)",
    )
    parser.add_argument(
        "--check_md5",
        action="store_true",
        default=False,
        help="Verify MD5 checksums of the files (default: False)",
    )
    parser.add_argument(
        "target_dir",
        type=str,
        metavar="TARGET_DIR",
        help="Path to the target directory where the data will be stored",
    )

    args = parser.parse_args()

    download_butreverb(
        args.target_dir, args.force_download, args.check_md5, args.num_jobs
    )
