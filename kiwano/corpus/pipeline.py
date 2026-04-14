"""
Generic execution engine for download and preparation pipelines.

The pipeline knows NOTHING about specific datasets.
It receives a DatasetSpec and orchestrates the plugins.
"""

import logging
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

from .archive import extract
from .checksum import verify_resources_parallel
from .concat import assemble
from .exceptions import DatasetDownloaderError, DownloadError
from .models import DatasetSpec, ResourceGroup, ResourceSpec
from .prep_models import PrepareSpec
from .state import DownloadState

log = logging.getLogger(__name__)


def _worker_init(use_vad: bool = False) -> None:
    """
    Called once per worker process at startup.

    Pre-loads torch, soundfile, and silero_vad (if VAD is enabled)
    to reduce first-file latency.
    """
    try:
        import torch

        torch.set_num_threads(1)  # Avoid CPU contention
    except ImportError:
        pass

    try:
        import soundfile  # noqa: F401
    except ImportError:
        pass

    if use_vad:
        try:
            from silero_vad import load_silero_vad

            load_silero_vad()
        except ImportError:
            pass


# ══════════════════════════════════════════════════════════════════
# Preparation pipeline
# ══════════════════════════════════════════════════════════════════


class PreparePipeline:
    """
    Execute corpus preparation:
      1. Discover audio files
      2. Parallel processing (file_processor or multi_processor)
      3. Write output list files
      4. Convert trial files
      5. Optional cleanup
    """

    def run(
        self,
        spec: DatasetSpec,
        in_data: Path,
        out_data: Path,
        sampling_frequency: int = 16000,
        vad: bool = False,
        delete_archives: bool = False,
        num_jobs: int = 20,
    ) -> None:
        if spec.prepare_spec is None:
            raise ValueError(f"[{spec.name}] No PrepareSpec defined for this corpus.")

        pspec: PrepareSpec = spec.prepare_spec
        out_data.mkdir(parents=True, exist_ok=True)

        # 1. Discover files
        search_root = (in_data / pspec.search_root) if pspec.search_root else in_data
        wav_lst = self._discover_files(search_root, pspec)
        log.info("[%s] %d file(s) found in %s", spec.name, len(wav_lst), search_root)

        # 2. Parallel processing
        if pspec.file_processor is not None:
            self._run_single_list(
                spec, pspec, wav_lst, out_data, sampling_frequency, vad, num_jobs
            )
        else:
            self._run_multi_list(
                spec, pspec, wav_lst, out_data, sampling_frequency, vad, num_jobs
            )

        # 3. Convert trial files
        if pspec.trials is not None:
            log.info("[%s] Converting trial files...", spec.name)
            pspec.trials.converter_fn(in_data, out_data)

        # 4. Cleanup
        if pspec.cleanup_fn is not None:
            pspec.cleanup_fn(in_data, delete_archives)

        log.info("[%s] Preparation complete -> %s", spec.name, out_data)

    # ── File discovery ────────────────────────────────────────

    def _discover_files(self, root: Path, pspec: PrepareSpec) -> List[Path]:
        from kiwano.utils import get_all_files

        # files = get_all_files(root, match_and=pspec.file_extensions)
        files = [Path(f) for f in get_all_files(root, match_and=pspec.file_extensions)]

        if pspec.match_or:
            files = [
                f for f in files if any(token in str(f) for token in pspec.match_or)
            ]
        return files

    # ── Single-list mode (most corpora) ───────────────────────

    def _run_single_list(
        self,
        spec: DatasetSpec,
        pspec: PrepareSpec,
        wav_lst: List[Path],
        out_data: Path,
        sampling_frequency: int,
        vad: bool,
        num_jobs: int,
    ) -> None:
        list_name = pspec.output_lists[0]
        list_path = out_data / list_name

        with list_path.open("w") as liste:
            with ProcessPoolExecutor(
                max_workers=num_jobs, initializer=_worker_init, initargs=(vad,)
            ) as ex:
                iterator = ex.map(
                    pspec.file_processor,
                    wav_lst,
                    [out_data] * len(wav_lst),
                    [sampling_frequency] * len(wav_lst),
                    [vad] * len(wav_lst),
                    chunksize=128,
                )
                for name, spkid, duration, segment in tqdm(
                    iterator, total=len(wav_lst), desc=f"[{spec.name}] Preparing"
                ):
                    liste.write(f"{name} {spkid} {duration} {segment}\n")

                """
                futures = {
                    ex.submit(
                        pspec.file_processor, seg, out_data, sampling_frequency, vad
                    ): seg
                    for seg in wav_lst
                }
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"[{spec.name}] Preparing",
                ):
                    seg = futures[future]
                    try:
                        name, spkid, duration, segment = future.result()
                        liste.write(f"{name} {spkid} {duration} {segment}\n")
                    except Exception as exc:
                        log.warning("Error processing %s: %s", seg.name, exc)
                """

        log.info("[%s] List written: %s", spec.name, list_path)

    # ── Multi-list mode (CN-Celeb: train / dev / test) ────────

    def _run_multi_list(
        self,
        spec: DatasetSpec,
        pspec: PrepareSpec,
        wav_lst: List[Path],
        out_data: Path,
        sampling_frequency: int,
        vad: bool,
        num_jobs: int,
    ) -> None:
        # Ouvre tous les fichiers liste
        handles = {name: (out_data / name).open("w") for name in pspec.output_lists}

        try:
            with ProcessPoolExecutor(
                max_workers=num_jobs, initializer=_worker_init, initargs=(vad,)
            ) as ex:
                futures = {
                    ex.submit(
                        pspec.multi_processor, seg, out_data, sampling_frequency, vad
                    ): seg
                    for seg in wav_lst
                }
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"[{spec.name}] Preparing",
                ):
                    seg = futures[future]
                    try:
                        results: Dict = future.result()
                        # results = {"listeTrain": (name, spkid, dur, path), ...}
                        for list_name, row in results.items():
                            if row is not None and list_name in handles:
                                name, spkid, duration, segment = row
                                handles[list_name].write(
                                    f"{name} {spkid} {duration} {segment}\n"
                                )
                    except Exception as exc:
                        log.warning("Error processing %s: %s", seg.name, exc)
        finally:
            for h in handles.values():
                h.close()

        for name in pspec.output_lists:
            log.info("[%s] List written: %s", spec.name, out_data / name)


# ══════════════════════════════════════════════════════════════════
# Download pipeline
# ══════════════════════════════════════════════════════════════════


class DownloadPipeline:
    """
    Execute: download -> verify -> assemble -> extract -> post-process.

    Supports both simple DatasetSpecs (1 implicit group) and multi-group
    specs (CN-Celeb, etc.). Each step is idempotent via DownloadState.
    """

    def run(
        self,
        spec: DatasetSpec,
        target_dir: Path,
        force: bool = False,
        verify: bool = False,
        num_jobs: int = 4,
    ) -> None:
        target_dir.mkdir(parents=True, exist_ok=True)
        state = DownloadState(target_dir, spec.name)

        if state.is_completed() and not force:
            log.info("[%s] Already complete. Use --force to re-download.", spec.name)

            return

        if force:
            state.reset()

        log.info("[%s] Starting download.", spec.name)

        for group in spec.resolved_groups():
            self._process_group(group, spec, target_dir, state, force, verify, num_jobs)

        # Optional post-processing hook
        # Used by Hi-MIA (multiple extractions) and ArabCeleb (YouTube download).
        if spec.postprocess_fn:
            log.info("[%s] Running post-processing...", spec.name)
            spec.postprocess_fn(target_dir, **spec.postprocess_kwargs)

        state.mark_completed()
        log.info("[%s] Completed successfully.", spec.name)

    # ── Group processing ──────────────────────────────────────

    def _process_group(
        self,
        group: ResourceGroup,
        spec: DatasetSpec,
        target_dir: Path,
        state: DownloadState,
        force: bool,
        verify: bool,
        num_jobs: int,
    ) -> None:
        # 1. Download
        self._download_all(group, spec, target_dir, state, force)

        # 2. Verify checksums (optional, parallel)
        if verify:
            verify_resources_parallel(target_dir, group.resources, num_jobs=num_jobs)
            for r in group.resources:
                state.mark_verified(r.filename)

        # 3. Assemble
        group_key = group.assemble.output_name or (
            group.resources[0].filename if group.resources else "_"
        )
        if not state.is_assembled_group(group_key):
            assemble(group, target_dir, dataset_name=spec.name)
            state.mark_assembled_group(group_key)

        # 4. Extract
        if not state.is_extracted_group(group_key):
            extract(group, target_dir, num_jobs=num_jobs, dataset_name=spec.name)
            state.mark_extracted_group(group_key)

    # ── Download all resources ────────────────────────────────

    def _download_all(
        self,
        group: ResourceGroup,
        spec: DatasetSpec,
        target_dir: Path,
        state: DownloadState,
        force: bool,
    ) -> None:
        errors = []
        for resource in group.resources:
            if state.is_downloaded(resource.filename) and not force:
                log.info("[%s] Already downloaded: %s", spec.name, resource.filename)
                continue
            try:
                self._download_one(resource, target_dir, spec, force)
                state.mark_downloaded(resource.filename)
            except DownloadError as exc:
                if resource.optional:
                    log.warning(
                        "Optional resource skipped: %s (%s)",
                        resource.filename,
                        exc,
                    )
                else:
                    errors.append(str(exc))

        if errors:
            raise DatasetDownloaderError(
                f"[{spec.name}] {len(errors)} download(s) failed:\n" + "\n".join(errors)
            )

    def _download_one(
        self,
        resource: ResourceSpec,
        target_dir: Path,
        spec: DatasetSpec,
        force: bool,
    ) -> None:
        """Route to the appropriate downloader backend."""
        dest = target_dir / resource.filename
        checksum_tuple = (
            (resource.checksum.algo, resource.checksum.value)
            if resource.checksum
            else None
        )
        kwargs = dict(
            destination=dest,
            label=f"[{spec.name}] {resource.filename}",
            force=force,
            max_retries=spec.max_retries,
            expected_checksum=checksum_tuple,
        )

        if resource.backend == "http":
            from .http import download_file

            download_file(url=resource.url, **kwargs)
        elif resource.backend == "github":
            from .github import download_file

            download_file(url=resource.url, **kwargs)
        elif resource.backend == "youtube":
            pass  # Handled in bulk by postprocess_fn
        else:
            raise DatasetDownloaderError(f"Unknown backend: '{resource.backend}'")
