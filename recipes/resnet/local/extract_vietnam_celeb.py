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
    zip_name = "vietnam-celeb.zip"
    zip_path = target_dir / zip_name

    # zip_files = ['vietnam-celeb-part.zip',
    #              'vietnam-celeb-part.z01',
    #              'vietnam-celeb-part.z02',
    #              'vietnam-celeb-part.z03']
    with open(zip_name, "wb") as outFile:
        for file in tqdm(sorted(target_dir.glob("vietnam-celeb-part.zip"))):
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

    extract_vietnam_celeb(args.target_dir)

