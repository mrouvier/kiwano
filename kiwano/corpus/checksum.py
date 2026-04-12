"""
Parallel MD5/SHA256 checksum verification.
"""


import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

from .exceptions import ChecksumError
from .models import ResourceSpec

log = logging.getLogger(__name__)

_CHUNK = 1 << 20  # 1 MiB


def _compute(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_resource(path: Path, resource: ResourceSpec) -> bool:
    """
    Verify the checksum of a single file.

    Policy:
      - Optional resource + missing file → warning, returns True.
      - No checksum in spec → warning, returns True.
      - Checksum mismatch → raises ChecksumError.
    """
    if not path.exists():
        if resource.optional:
            log.warning("Optional resource missing: %s", path.name)
            return True
        raise ChecksumError(f"File missing: {path}")

    if resource.checksum is None:
        log.warning("No checksum defined for %s — skipping verification.", path.name)
        return True

    algo = resource.checksum.algo
    expected = resource.checksum.value
    computed = _compute(path, algo)

    if computed != expected:
        raise ChecksumError(
            f"{algo.upper()} checksum mismatch for {path.name}:\n"
            f"  expected: {expected}\n"
            f"  computed: {computed}"
        )

    log.info("Checksum OK: %s", path.name)
    return True


def verify_resources_parallel(
    target_dir: Path,
    resources: List[ResourceSpec],
    num_jobs: int = 4,
) -> None:
    """Verify all checksums in parallel. Raises ChecksumError on first failure."""
    errors: List[str] = []

    with ThreadPoolExecutor(max_workers=num_jobs) as pool:
        futures = {
            pool.submit(verify_resource, target_dir / r.filename, r): r
            for r in resources
            if r.checksum is not None
        }
        for future in as_completed(futures):
            try:
                future.result()
            except ChecksumError as exc:
                errors.append(str(exc))

    if errors:
        raise ChecksumError("\n".join(errors))
