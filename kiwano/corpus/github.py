"""
GitHub file downloader via kiwano.utils.download_from_github.

Blob URLs are automatically converted to raw URLs by kiwano.
"""

import logging
from pathlib import Path
from typing import Optional

from kiwano.utils import download_from_github as _download_from_github

from .exceptions import DownloadError

log = logging.getLogger(__name__)


def download_file(
    url: str,
    destination: Path,
    label: str = "",
    force: bool = False,
    max_retries: int = 3,
    expected_checksum: Optional[tuple] = None,
) -> None:
    """
    Download a file from GitHub (raw or blob URL).

    Skips if the file already exists and ``force`` is False.
    """
    if destination.exists() and not force:
        log.info("Already present, skipping: %s", destination.name)
        return

    try:
        _download_from_github(url, destination)
    except Exception as exc:
        raise DownloadError(f"GitHub download failed: {url}") from exc
