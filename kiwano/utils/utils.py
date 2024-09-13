from typing import Union
from pathlib import Path
from tqdm.auto import tqdm

import hashlib
import zipfile
import tarfile
import os
import concurrent.futures
import requests
import logging

Pathlike = Union[Path, str]

def extract_file_zip(zip_file, member, output_dir):
    if member.endswith('/'):
        return  
    member_path = os.path.join(output_dir, member)
    os.makedirs(os.path.dirname(member_path), exist_ok=True)
    with zip_file.open(member) as source, open(os.path.join(output_dir, member), 'wb') as target:
        target.write(source.read())



def parallel_unzip(zip_path, output_dir, jobs):
    with zipfile.ZipFile(zip_path, 'r') as zip_file:
        members = zip_file.namelist()

        os.makedirs(output_dir, exist_ok=True)

        with concurrent.futures.ThreadPoolExecutor(jobs) as ex:
            futures = []

            for member in members:
                futures.append(ex.submit(extract_file_zip, zip_file, member, output_dir))

            for future in tqdm(futures, desc="Unzipping files"):
                future.result()

def extract_tar(tar_gz_path, extract_path='.'):
    with tarfile.open(tar_gz_path, 'r:gz') as tar:
        total_members = len(tar.getmembers())
        for member in tqdm(tar.getmembers(), total=total_members, desc="Extracting"):
            tar.extract(member, path=extract_path)



def copy_files(zip_path, target_dir, part):
    files = sorted(target_dir.glob(part))

    total_size = sum(file.stat().st_size for file in files)

    with open(zip_path, "wb") as outFile:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Concatenating different file parts") as pbar:
            for file in files:
                with open(file, "rb") as inFile:
                    while True:
                        buf = inFile.read(1024 * 1024)  # Read in chunks of 1MB
                        if not buf:
                            break
                        outFile.write(buf)
                        pbar.update(len(buf))


def get_all_files(dirName, match_and=None, match_or=None, exclude_and=None, exclude_or=None):
    """
    https://github.com/speechbrain/speechbrain/blob/develop/speechbrain/utils/data_utils.py#L61C2-L170C1

    Returns a list of files found within a folder.

    Different options can be used to restrict the search to some specific
    patterns.

    Arguments
    ---------
    dirName : str
        The directory to search.
    match_and : list
        A list that contains patterns to match. The file is
        returned if it matches all the entries in `match_and`.
    match_or : list
        A list that contains patterns to match. The file is
        returned if it matches one or more of the entries in `match_or`.
    exclude_and : list
        A list that contains patterns to match. The file is
        returned if it matches none of the entries in `exclude_and`.
    exclude_or : list
        A list that contains pattern to match. The file is
        returned if it fails to match one of the entries in `exclude_or`.

    Returns
    -------
    allFiles : list
        The list of files matching the patterns.

    Example
    -------
    >>> get_all_files('tests/samples/RIRs', match_and=['3.wav'])
    ['tests/samples/RIRs/rir3.wav']
    """
    # Match/exclude variable initialization
    match_and_entry = True
    match_or_entry = True
    exclude_or_entry = False
    exclude_and_entry = False

    # Create a list of file and sub directories
    listOfFile = os.scandir(dirName)
    allFiles = list()

    # Iterate over all the entries
    for entry in listOfFile:

        # Create full path
        fullPath = os.path.join(dirName, entry.name)

        # If entry is a directory then get the list of files in this directory
        if entry.is_dir():
            allFiles = allFiles + get_all_files(
                fullPath,
                match_and=match_and,
                match_or=match_or,
                exclude_and=exclude_and,
                exclude_or=exclude_or,
            )
        else:

            # Check match_and case
            if match_and is not None:
                match_and_entry = False
                match_found = 0

                for ele in match_and:
                    if ele in fullPath:
                        match_found = match_found + 1
                if match_found == len(match_and):
                    match_and_entry = True

            # Check match_or case
            if match_or is not None:
                match_or_entry = False
                for ele in match_or:
                    if ele in fullPath:
                        match_or_entry = True
                        break

            # Check exclude_and case
            if exclude_and is not None:
                match_found = 0

                for ele in exclude_and:
                    if ele in fullPath:
                        match_found = match_found + 1
                if match_found == len(exclude_and):
                    exclude_and_entry = True

            # Check exclude_or case
            if exclude_or is not None:
                exclude_or_entry = False
                for ele in exclude_or:
                    if ele in fullPath:
                        exclude_or_entry = True
                        break

            # If needed, append the current file to the output list
            if (
                match_and_entry
                and match_or_entry
                and not (exclude_and_entry)
                and not (exclude_or_entry)
            ):
                allFiles.append(fullPath)

    listOfFile.close()

    return allFiles


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



def check_md5(file: Pathlike, hash: str) -> bool:
    """
    Check the md5 hash of a given file.

    Arguments
    ---------
    file : Pathlike
        The file to check.
    hash : str
        The expected md5 hash.

    Returns
    -------
    bool
        True if the downloaded file has the correct hash, False otherwise.
    """
    try:
        with open(file, 'rb') as file:
            md5 = hashlib.file_digest(file, 'md5').hexdigest()
        if md5 != hash:
            raise ValueError()
        else:
            logging.info("File ", file, " correctly downloaded")
            return True
    except ValueError:
        logging.info(f"Error. Hash of the downloaded file {file} not corresponding to the expected one.")
        return False

def download_from_github(url: str, save_path: str):
    """
    Download a file from a GitHub URL to a local path.

    Args:
        url: The URL of the file to download.
        save_path: The local path to save the file to.

    Raises:
        requests.HTTPError: If the request to the URL fails.

    """
    raw_url = url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
    response = requests.get(raw_url)
    response.raise_for_status() 
    
    with open(save_path, 'wb') as f:
        f.write(response.content)
    
    logging.info(f"Downloaded {url} to {save_path}")
