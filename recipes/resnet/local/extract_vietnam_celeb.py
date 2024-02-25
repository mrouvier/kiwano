import argparse
import shutil
import tarfile
import zipfile
from pathlib import Path

from tqdm import tqdm

from kiwano.utils import Pathlike


def extract_vietnam_celeb(target_dir: Pathlike = "."):
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    tar_gz_name = "vietnam-celeb.tar.gz"
    tar_gz_path = target_dir / tar_gz_name
    with open(tar_gz_path, "wb") as outFile:
        for file in tqdm(sorted(target_dir.glob("vietnam-celeb-part.z0*"))):
            with open(file, "rb") as inFile:
                shutil.copyfileobj(inFile, outFile)

    print(f"Unzipping train...", flush=True)
    with tarfile.open(tar_gz_path) as zf:
        zf.extractall(target_dir)

    print(f"Unzipping test...", flush=True)
    with tarfile.open(target_dir / "vietnam-celeb-part.zip") as zf:
        zf.extractall(target_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('target_dir', metavar='target_dir', type=str,
                        help='the path to the target directory where the data will be stored')
    args = parser.parse_args()

    extract_vietnam_celeb(args.target_dir)

