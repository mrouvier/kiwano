"""
Archive extraction for zip and tar/tgz formats.
"""


import logging
from pathlib import Path

from kiwano.utils import extract_tar, parallel_unzip

from .exceptions import ExtractionError
from .models import ResourceGroup

log = logging.getLogger(__name__)


def _extract_tar_path(archive: Path, target_dir: Path) -> None:
    """Helper direct pour extract_tar depuis un chemin (utilisé par postprocess_fn)."""
    try:
        extract_tar(archive, target_dir)
    except Exception as exc:
        raise ExtractionError(f"Échec d'extraction de {archive.name} : {exc}") from exc


def extract(
    group: ResourceGroup, target_dir: Path, num_jobs: int = 1, dataset_name: str = ""
) -> None:
    """
    Extract the archive described by ``group.extract``.

    Raises ExtractionError if the archive is missing or extraction fails.
    """
    espec = group.extract
    label = f"[{dataset_name}]" if dataset_name else ""

    if espec.method == "none":
        return

    # Resolve archive path
    if espec.archive_name:
        archive = target_dir / espec.archive_name
    else:
        archive = target_dir / group.resources[0].filename

    if not archive.exists():
        raise ExtractionError(f"Archive not found: {archive}")

    log.info("%sExtracting (%s): %s", label, espec.method, archive.name)

    try:
        if espec.method == "zip":
            parallel_unzip(archive, target_dir, num_jobs)
        elif espec.method in ("tar", "tgz"):
            extract_tar(archive, target_dir)
        else:
            raise ExtractionError(f"Unknown extraction method: {espec.method}")
    except ExtractionError:
        raise
    except Exception as exc:
        raise ExtractionError(f"Failed to extract {archive.name}: {exc}") from exc

    if espec.remove_after:
        archive.unlink()
        log.info("%sArchive removed after extraction: %s", label, archive.name)
