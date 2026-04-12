"""
Download state persistence for resumable pipelines.

Stored in ``<target_dir>/.download_state.json``.
"""

import json
import logging
from pathlib import Path
from typing import Dict

log = logging.getLogger(__name__)


class DownloadState:
    """
    Track progress of each resource and global pipeline stages.

    JSON structure::

        {
            "resources": {
                "vox2_dev_aac_partaa": {"downloaded": true, "verified": true},
                ...
            },
            "groups": {
                "vox2_aac.zip": {"assembled": true, "extracted": true},
                ...
            },
            "completed": false
        }
    """

    _FILENAME = ".download_state.json"

    def __init__(self, target_dir: Path, dataset_name: str):
        self._path = target_dir / self._FILENAME
        self._name = dataset_name
        self._state: Dict = self._load()

    # ── Persistence ───────────────────────────────────────────

    def _load(self) -> Dict:
        if self._path.exists():
            try:
                with self._path.open() as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                log.warning("Corrupted state file — resetting.")
        return {
            "resources": {},
            "assembled": False,
            "extracted": False,
            "completed": False,
        }

    def _save(self):
        with self._path.open("w") as f:
            json.dump(self._state, f, indent=2)

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------

    def is_completed(self) -> bool:
        return self._state.get("completed", False)

    def is_downloaded(self, filename: str) -> bool:
        return self._state["resources"].get(filename, {}).get("downloaded", False)

    def is_verified(self, filename: str) -> bool:
        return self._state["resources"].get(filename, {}).get("verified", False)

    def mark_downloaded(self, filename: str):
        self._state["resources"].setdefault(filename, {})["downloaded"] = True
        self._save()

    def mark_verified(self, filename: str):
        self._state["resources"].setdefault(filename, {})["verified"] = True
        self._save()

    def mark_assembled(self):
        self._state["assembled"] = True
        self._save()

    def mark_extracted(self):
        self._state["extracted"] = True
        self._save()

    def mark_completed(self):
        self._state["completed"] = True
        self._save()
        log.info("[%s] State: download complete.", self._name)

    def reset(self):
        self._state = {
            "resources": {},
            "assembled": False,
            "extracted": False,
            "completed": False,
        }
        self._save()
        log.info("[%s] State reset.", self._name)

    @property
    def is_assembled(self) -> bool:
        return self._state.get("assembled", False)

    @property
    def is_extracted(self) -> bool:
        return self._state.get("extracted", False)

    # Support multi-groupes (CN-Celeb, etc.)
    def is_assembled_group(self, group_key: str) -> bool:
        return self._state.get("groups", {}).get(group_key, {}).get("assembled", False)

    def is_extracted_group(self, group_key: str) -> bool:
        return self._state.get("groups", {}).get(group_key, {}).get("extracted", False)

    def mark_assembled_group(self, group_key: str):
        self._state.setdefault("groups", {}).setdefault(group_key, {})[
            "assembled"
        ] = True
        self._save()

    def mark_extracted_group(self, group_key: str):
        self._state.setdefault("groups", {}).setdefault(group_key, {})[
            "extracted"
        ] = True
        self._save()
