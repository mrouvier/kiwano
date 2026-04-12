"""
HTTP download with progress bar, retry, and inline MD5 verification.

Hi-MIA pattern: if the MD5 is wrong after download, the file is deleted
and re-downloaded.
"""

import logging
import time
from pathlib import Path
from typing import Optional

from kiwano.utils import urlretrieve_progress

from .exceptions import ChecksumError, DownloadError
from .checksum import _compute

log = logging.getLogger(__name__)

_RETRY_DELAY = 5  # seconds


def download_file(
    url: str,
    destination: Path,
    label: str = "",
    force: bool = False,
    max_retries: int = 3,
    expected_checksum: Optional[tuple] = None,  # (algo, value)
) -> None:
    """
    Download `url` to `destination`.

    Behavior:
      - Skipped if the file exists, force=False, AND checksum is valid.
      - If the checksum is wrong on an existing file → delete + re-download.
      - Retries up to `max_retries` times on network failure or bad checksum.
      - Raises DownloadError if all attempts fail.
    """
    desc = label or destination.name

    for attempt in range(1, max_retries + 1):
        # Already present?
        if destination.exists() and not force:
            if expected_checksum is None:
                log.info("Already present, skipping: %s", destination.name)
                return

            algo, value = expected_checksum
            computed = _compute(destination, algo)
            if computed == value:
                log.info("Already present with valid checksum: %s", destination.name)
                return
            else:
                log.warning(
                    "Checksum mismatch for %s (expected %s, got %s). "
                    "Deleting and re-downloading (attempt %d/%d).",
                    destination.name, expected_value, computed, attempt, max_retries,
                )
                destination.unlink()

        try:
            urlretrieve_progress(url, filename=destination, desc=desc)
        except Exception as exc:
            log.warning(
                "Attempt %d/%d failed (network) for %s: %s",
                attempt, max_retries, destination.name, exc,
            )
            if attempt < max_retries:
                time.sleep(_RETRY_DELAY)
            continue

        # Post-download verification
        if expected_checksum is not None:
            algo, value = expected_checksum
            computed = _compute(destination, algo)
            if computed != value:
                log.warning(
                    "Checksum mismatch after downloading %s (attempt %d/%d). Retrying...",
                    destination.name, attempt, max_retries,
                )
                destination.unlink()
                if attempt < max_retries:
                    time.sleep(_RETRY_DELAY)
                continue

        return  # Success

    raise DownloadError(
        f"Failed to download {url} after {max_retries} attempts "
        f"(network instability or persistent checksum mismatch)."
    )
