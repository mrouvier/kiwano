"""
YouTube audio downloader via pytubefix.

ArabCeleb pattern:
  - The JSON manifest is downloaded first (via github.py).
  - This plugin reads the JSON, resolves YouTube URLs, and downloads in parallel.
  - Private/unavailable videos are logged as warnings (optional resources).
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from .exceptions import DownloadError

log = logging.getLogger(__name__)

try:
    import pytubefix
    from pytubefix import YouTube

    _PYTUBE_AVAILABLE = True
except ImportError:
    _PYTUBE_AVAILABLE = False


def _download_one(celeb_dir: Path, video_id: str, url: str) -> Optional[str]:
    """
    Download a single YouTube audio stream.

    Returns the URL if unavailable (for logging), None on success.
    """
    if not _PYTUBE_AVAILABLE:
        raise ImportError(
            "pytubefix is required for ArabCeleb. "
            "Install it with: pip install pytubefix"
        )

    dest = celeb_dir / f"{video_id}.mp4"
    if dest.exists():
        log.debug("Already present: %s", dest)
        return None

    try:
        yt = YouTube(url)
        stream = yt.streams.filter(only_audio=True)[0]
        stream.download(str(celeb_dir), filename=dest.name)
        return None
    except Exception as exc:
        log.warning("Video unavailable (%s): %s", type(exc).__name__, url)
        return url


def download_from_manifest(
    manifest_path: Path,
    target_dir: Path,
    num_jobs: int = 1,
) -> None:
    """
    Read the ArabCeleb JSON manifest and download all videos in parallel.

    Expected JSON structure::
    {
        "<celeb_id>": {
            "utterances": {
                "<youtube_url>": [...],
                ...
            }
        },
        ...
    }
    """
    if not manifest_path.exists():
        raise DownloadError(f"Manifest not found: {manifest_path}")

    with manifest_path.open() as f:
        utterances = json.load(f)

    tasks = []
    for celeb_id, value in utterances.items():
        celeb_dir = target_dir / celeb_id
        celeb_dir.mkdir(parents=True, exist_ok=True)
        for video_idx, (url, _) in enumerate(value["utterances"].items()):
            tasks.append((celeb_dir, str(video_idx), url))

    not_available = []
    with ThreadPoolExecutor(max_workers=num_jobs) as pool:
        futures = {
            pool.submit(_download_one, celeb_dir, vid_id, url): url
            for celeb_dir, vid_id, url in tasks
        }
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Downloading ArabCeleb (YouTube)",
        ):
            result = future.result()
            if result:
                not_available.append(result)

    if not_available:
        log.warning("%d video(s) unavailable:", len(not_available))
        for url in not_available:
            log.warning(f"  {url}")
