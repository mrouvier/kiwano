#!/usr/bin/env python3

import json
import pytubefix
import logging
import requests
import argparse

from pytubefix import YouTube
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from kiwano.utils import Pathlike, check_md5, download_from_github
from pathlib import Path
from typing import Optional


ARAB_CELEB_URL = [
    ["https://github.com/CeLuigi/ArabCeleb/blob/main/utterance_info.json", "d22b74358ace281d866f00dad302d53f"]
]


def download_video(args):
    celeb_datadir, id_video, url = args
    try:
        yt = YouTube(url)
        s = yt.streams.filter(only_audio=True)[0]
        filename = str(id_video) + '.mp4'
        s.download(celeb_datadir, filename=filename)
        return None
    except (pytubefix.exceptions.VideoPrivate, pytubefix.exceptions.VideoUnavailable) as e:
        return url

def download_arab_celeb(target_dir: Pathlike = ".", jobs: int = 1, force_download: Optional[bool] = False):
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    utterance_info_file = target_dir / "utterance_info.json"
    
    if not utterance_info_file.exists() or force_download:
        utterance_info_url = ARAB_CELEB_URL[0][0]
        logging.info(f"Downloading utterance_info.json...")
        download_from_github(utterance_info_url, utterance_info_file)
        if not check_md5(utterance_info_file, ARAB_CELEB_URL[0][1]):
            logging.error(f"MD5 check failed for utterance_info.json. The file may be corrupted.")
            return
    else:
        logging.info("utterance_info.json already exists. Skipping download.")

    with open(utterance_info_file) as f:
        utterances = json.load(f)
    
    not_available = []

    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = []
        for key, value in utterances.items():
            celeb_datadir = target_dir / key
            celeb_datadir.mkdir(parents=True, exist_ok=True)
            
            for id_video, (url, _) in enumerate(value['utterances'].items()):
                futures.append(executor.submit(download_video, (celeb_datadir, id_video, url)))
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading Arab_Celeb YouTube base files...."):
            result = future.result()
            if result:
                not_available.append(result)
    
    logging.info('The following videos are no longer available:')
    for na in not_available: 
        logging.info(na)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('target_dir', metavar='target_dir', type=str,
                        help='the path to the target directory where the data will be stored')
    parser.add_argument('--jobs', type=int, default=1,
                        help='the number of parallel downloads (default: 1)')
    parser.add_argument('--force_download', action="store_true", default=False,
                        help='force the download, overwrites files (default: False)')

    args = parser.parse_args()

    download_arab_celeb(args.target_dir, args.jobs, args.force_download)

			
    