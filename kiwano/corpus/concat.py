"""
Multi-part archive assembly.

Supported methods:
  - copy_glob    : shell-like concat via kiwano's copy_files (VoxCeleb*)
  - concat_parts : pure-Python concatenation via shutil.copyfileobj
  - concat_clean : same as concat_parts + delete parts after assembly (CN-Celeb2)
"""

import logging
import shutil
from pathlib import Path

from kiwano.utils import copy_files

from .exceptions import AssemblyError
from .models import AssembleSpec, ResourceGroup

log = logging.getLogger(__name__)


def assemble(group: ResourceGroup, target_dir: Path, dataset_name: str = "") -> Path:
    """
    Reconstruct the final archive according to ``group.assemble``.

    Returns the path to the produced archive.
    """
    aspec: AssembleSpec = group.assemble
    label = f"[{dataset_name}] " if dataset_name else ""

    if aspec.method == "none" or aspec.output_name is None:
        return target_dir / group.resources[0].filename

    output = target_dir / aspec.output_name

    if output.exists():
        log.info("%sArchive already assembled: %s", label, output.name)
        return output

    if aspec.method == "copy_glob":
        if not aspec.pattern:
            raise AssemblyError("copy_glob requires a 'pattern' in AssembleSpec")
        log.info("%sAssembling (copy_glob) -> %s", label, output.name)
        copy_files(output, target_dir, aspec.pattern)

    elif aspec.method in ("concat_parts", "concat_clean"):
        parts = sorted(
            p for p in target_dir.glob(aspec.pattern or "*")
            if p.is_file() and p != output
        )
        if not parts:
            raise AssemblyError(
                f"No parts found matching pattern '{aspec.pattern}'"
            )
        log.info(
            "%sAssembling (%s, %d parts) -> %s",
            label, aspec.method, len(parts), output.name,
        )
        with output.open("wb") as out_f:
            for part in parts:
                with part.open("rb") as in_f:
                    shutil.copyfileobj(in_f, out_f)

        if aspec.method == "concat_clean" or aspec.remove_parts:
            log.info("%sCleaning up parts...", label)
            for part in parts:
                part.unlink()

    else:
        raise AssemblyError(f"Unknown assembly method: {aspec.method}")

    return output
