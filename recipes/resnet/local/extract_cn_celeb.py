import argparse
import shutil
import zipfile
from pathlib import Path

from kiwano.utils import Pathlike


def extract_cn_celeb(target_dir: Pathlike = "."):
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    zip_name = "cn_celeb.zip"
    zip_path = target_dir / zip_name
    with open(zip_path, "wb") as outFile:
        for file in sorted(target_dir.glob("cn-celeb2_v2.*")):
            with open(file, "rb") as inFile:
                shutil.copyfileobj(inFile, outFile)

    print(f"Unzipping files...", flush=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('target_dir', metavar='target_dir', type=str,
                        help='the path to the target directory where the data will be stored')
    args = parser.parse_args()

    extract_cn_celeb(args.target_dir)

