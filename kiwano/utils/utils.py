from typing import Union
from pathlib import Path
from tqdm.auto import tqdm

import hashlib
import os

Pathlike = Union[Path, str]

def tqdm_urlretrieve_hook(t):
    last_b = [0]

    def update_to(b=1, bsize=1, tsize=None):
        if tsize not in (None, -1):
            t.total = tsize
        displayed = t.update((b - last_b[0]) * bsize)
        last_b[0] = b
        return displayed

    return update_to

def urlretrieve_progress(url, filename=None, data=None, desc=None):
    from urllib.request import urlretrieve

    with tqdm(unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc=desc) as t:
        reporthook = tqdm_urlretrieve_hook(t)
        return urlretrieve(url=url, filename=filename, reporthook=reporthook, data=data)



def check_md5(dir, liste):
    """
    arg1 dir: the directory where the files in the liste are stocked
    arg2 liste: the name of the liste
    This function check if all the files in the list are downloaded correctly,
    if not, there are 3 attempts to re-download the file
    """

    for url in liste:
        fname = dir / url[0].split("/")[-1]

        for i in range(3):
            try:
                with open(fname, 'rb') as file:
                    hash = hashlib.md5()
                    while True:
                        chunk = file.read(8096)
                        if not chunk:
                            break
                        hash.update(chunk)
                    md5 = hash.hexdigest()

                if md5 != url[1]:
                    raise ValueError()
                else:
                    print("File ", fname, " correctly downloaded")
                    break
            except ValueError:
                print("error downloading file ", fname)
                urlretrieve_progress(url[0], filename=dir / url[0].split("/")[-1], desc=f"Downloading VoxCeleb1 {url[0].split('/')[-1]}")

        else:
            if hashlib.md5(fname.read_bytes()).hexdigest() != url[1]:
                print("Download failed for file ", fname)
                os.remove(fname)
            else:
                print("File ", fname," finally correctly downloaded")
