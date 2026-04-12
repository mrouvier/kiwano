"""
Per-corpus file processing logic.

Each ``process_*`` function implements the FileProcessorFn contract:
    (segment, out_data, sampling_frequency, vad)
    -> (name, spkid, duration, output_path)

The PrepareSpec instances are defined at the bottom of this module
and imported by the registry.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import soundfile as sf

from .audio_utils import (
    apply_vad,
    ffmpeg_to_tensor,
    ffmpeg_to_wav,
    get_duration,
    load_audio,
)
from .prep_models import PrepareSpec, TrialsSpec

log = logging.getLogger(__name__)

# ── BUT ReverbDB ──────────────────────────────────────────────────


def _process_butreverb(
    segment: Path, out_data: Path, sampling_frequency: int, vad: bool
) -> Tuple[str, str, str, Path]:
    parts = segment.parts
    name = "_".join(
        [
            parts[-6],
            parts[-5],
            parts[-4],
            parts[-3],
            parts[-2],
            segment.stem,
            segment.suffix.lstrip("."),
        ]
    )
    duration = str(round(get_duration(segment), 2))
    return name, "butreverb", duration, segment


prepare_butreverb = PrepareSpec(
    file_processor=_process_butreverb,
    file_extensions=[".wav"],
)


# ── MUSAN ─────────────────────────────────────────────────────────


def _process_musan(
    segment: Path, out_data: Path, sampling_frequency: int, vad: bool
) -> Tuple[str, str, str, Path]:
    name = segment.stem
    spkid = segment.parts[-3]

    if sampling_frequency != 16000:
        out_dir = out_data / segment.parts[-4] / segment.parts[-3] / segment.parts[-2]
        out_dir.mkdir(parents=True, exist_ok=True)
        output = out_dir / segment.name
        ffmpeg_to_wav(segment, output, sampling_frequency)
        new_segment = output
    else:
        new_segment = segment

    duration = str(round(get_duration(new_segment), 2))
    return name, spkid, duration, new_segment


prepare_musan = PrepareSpec(
    file_processor=_process_musan,
    file_extensions=[".wav"],
)

# ── RIRS Noises ───────────────────────────────────────────────────


def _process_rirs(
    segment: Path, out_data: Path, sampling_frequency: int, vad: bool
) -> Tuple[str, str, str, Path]:
    spkid = segment.parts[-3]
    name = f"{spkid}_{segment.stem}"

    if sampling_frequency != 16000:
        out_dir = (
            out_data
            / segment.parts[-5]
            / segment.parts[-4]
            / segment.parts[-3]
            / segment.parts[-2]
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        output = out_dir / segment.name
        ffmpeg_to_wav(segment, output, sampling_frequency)
        new_segment = output
    else:
        new_segment = segment

    duration = str(round(get_duration(new_segment), 2))
    return name, spkid, duration, new_segment


prepare_rirs_noises = PrepareSpec(
    file_processor=_process_rirs,
    file_extensions=[".wav"],
    match_or=["mediumroom", "smallroom"],
)


# ── VoxCeleb 1 ────────────────────────────────────────────────────


def _process_voxceleb1(
    segment: Path, out_data: Path, sampling_frequency: int, vad: bool
) -> Tuple[str, str, str, Path]:
    name = "_".join(segment.parts[-3:]).rsplit(".", 1)[0]
    spkid = segment.parts[-3]
    emission = segment.parts[-2]
    n = segment.stem

    if sampling_frequency != 16000 or vad:
        out_dir = out_data / "wav" / spkid / emission
        out_dir.mkdir(parents=True, exist_ok=True)
        output = out_dir / (n + ".wav")
        if not output.exists():
            wav = load_audio(segment, sampling_frequency, use_vad=vad)
            sf.write(str(output), wav.numpy(), sampling_frequency)
        new_segment = output
        duration = str(round(get_duration(new_segment), 2))
    else:
        new_segment = segment
        duration = str(round(get_duration(segment), 2))

    return name, spkid, duration, new_segment


def _convert_voxceleb1_trials(in_data: Path, out_data: Path) -> None:
    mapping = {
        "veri_test2.txt": "voxceleb1-o-cleaned.trials",
        "list_test_hard2.txt": "voxceleb1-h-cleaned.trials",
        "list_test_all2.txt": "voxceleb1-e-cleaned.trials",
    }
    for src_name, dst_name in mapping.items():
        src = in_data / src_name
        dst = out_data / dst_name
        if not src.exists():
            log.warning("Trials file missing: %s", src)
            continue
        with src.open() as r, dst.open("w") as w:
            for line in r:
                cols = line.strip().split()
                label = "target" if cols[0] != "0" else "nontarget"
                utt1 = cols[1].replace("/", "_").rsplit(".", 1)[0]
                utt2 = cols[2].replace("/", "_").rsplit(".", 1)[0]
                w.write(f"{utt1} {utt2} {label}\n")


def _cleanup_voxceleb1(in_data: Path, delete: bool) -> None:
    if not delete:
        return
    for f in sorted(in_data.glob("vox1_dev_wav_part*")):
        f.unlink()
    for name in ["vox1_test_wav.zip", "vox1_dev_wav.zip"]:
        p = in_data / name
        if p.exists():
            p.unlink()


prepare_voxceleb1 = PrepareSpec(
    file_processor=_process_voxceleb1,
    file_extensions=[".wav"],
    search_root="wav",
    trials=TrialsSpec(converter_fn=_convert_voxceleb1_trials),
    cleanup_fn=_cleanup_voxceleb1,
)

# ── VoxCeleb 2 ────────────────────────────────────────────────────


def _process_voxceleb2(
    segment: Path, out_data: Path, sampling_frequency: int, vad: bool
) -> Tuple[str, str, str, Path]:
    name = "_".join(segment.parts[-3:]).rsplit(".", 1)[0]
    spkid = segment.parts[-3]
    emission = segment.parts[-2]
    n = segment.stem

    out_dir = out_data / "wav" / spkid / emission
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / (n + ".wav")

    if not output.exists():
        ffmpeg_to_wav(segment, output, sampling_frequency)

    duration = str(round(get_duration(output), 2))
    return name, spkid, duration, output


def _cleanup_voxceleb2(in_data: Path, delete: bool) -> None:
    if not delete:
        return
    for f in sorted(in_data.glob("vox2_dev_aac_part*")):
        f.unlink()
    p = in_data / "vox2_aac.zip"
    if p.exists():
        p.unlink()


prepare_voxceleb2 = PrepareSpec(
    file_processor=_process_voxceleb2,
    file_extensions=[".m4a"],
    search_root="dev/aac",
    cleanup_fn=_cleanup_voxceleb2,
)


# ── CN-Celeb (multi-list: train / dev / test) ────────────────────


def _process_cnceleb_multi(
    segment: Path, out_data: Path, sampling_frequency: int, vad: bool
) -> Dict[str, Optional[Tuple]]:
    """
    Route the segment to the correct output list based on its path.

    Returns a dict {list_name: row_tuple | None}.
    """
    result: Dict[str, Optional[Tuple]] = {
        "listeTrain": None,
        "listeDev": None,
        "listeTest": None,
    }

    str_seg = str(segment)

    # CN-Celeb2 (train) : CN-Celeb2_flac/data/...
    if "CN-Celeb2_flac" in str_seg:
        spkid = segment.parts[-2]
        n = segment.stem
        out_dir = out_data / "CN-Celeb2_flac" / "wav" / spkid
        out_dir.mkdir(parents=True, exist_ok=True)
        output = out_dir / (n + ".wav")
        if not output.exists():
            ffmpeg_to_wav(segment, output, sampling_frequency)
        duration = str(round(get_duration(output), 2))
        name = "_".join(segment.parts[-3:]).rsplit(".", 1)[0]
        toolkit_path = Path("db") / "cnceleb2" / "wav" / spkid / (n + ".wav")
        result["listeTrain"] = (name, spkid, duration, toolkit_path)

    # CN-Celeb1 dev : CN-Celeb_flac/data/... spkid < 800
    elif "CN-Celeb_flac" in str_seg and "data" in str_seg:
        try:
            spkid_str = segment.parts[segment.parts.index("data") + 1]
            spk_num = int(spkid_str[2:])
        except (ValueError, IndexError):
            return result
        if spk_num < 800:
            spkid = spkid_str
            n = segment.stem
            out_dir = out_data / "CN-Celeb_flac" / "dev" / "wav" / spkid
            out_dir.mkdir(parents=True, exist_ok=True)
            output = out_dir / (n + ".wav")
            if not output.exists():
                ffmpeg_to_wav(segment, output, sampling_frequency)
            duration = str(round(get_duration(output), 2))
            name = "_".join(segment.parts[-3:]).rsplit(".", 1)[0]
            toolkit_path = Path("db") / "cnceleb1_dev" / "wav" / spkid / (n + ".wav")
            result["listeDev"] = (name, spkid, duration, toolkit_path)

    # CN-Celeb1 eval
    elif "CN-Celeb_flac" in str_seg and "eval" in str_seg:
        n = segment.stem
        out_dir = out_data / "CN-Celeb_flac" / "eval" / "wav"
        out_dir.mkdir(parents=True, exist_ok=True)
        output = out_dir / (n + ".wav")
        if not output.exists():
            ffmpeg_to_wav(segment, output, sampling_frequency)
        duration = str(round(get_duration(output), 2))
        spkid = segment.name.split("-")[0]
        name = "_".join(segment.parts[-3:]).rsplit(".", 1)[0]
        toolkit_path = Path("db") / "cnceleb1" / "wav" / (n + ".wav")
        result["listeTest"] = (name, spkid, duration, toolkit_path)

    return result


def _convert_cnceleb_trials(in_data: Path, out_data: Path) -> None:
    enroll_path = in_data / "CN-Celeb_flac" / "eval" / "lists" / "enroll.lst"
    trials_path = in_data / "CN-Celeb_flac" / "eval" / "lists" / "trials.lst"
    out_path = in_data / "CN-Celeb_flac" / "eval" / "lists" / "new_trials.txt"

    if not enroll_path.exists() or not trials_path.exists():
        log.warning(
            "CN-Celeb enroll/trials files not found — skipping trial conversion."
        )
        return

    enroll = {}
    with enroll_path.open() as f:
        for line in f:
            col1, col2 = line.strip().split()
            enroll[col1] = col2

    with trials_path.open() as f_in, out_path.open("w") as f_out:
        for line in f_in:
            col1, col2, col3 = line.strip().split()
            label = "target" if col3 != "0" else "nontarget"
            utt1 = "eval_enroll_" + enroll[col1][7:-4]
            utt2 = "eval_test_" + col2[5:-4]
            f_out.write(f"{utt1} {utt2} {label}\n")


def _cleanup_cnceleb(in_data: Path, delete: bool) -> None:
    if not delete:
        return
    for f in sorted(in_data.glob("cn-celeb2_v2.tar*")):
        f.unlink()
    for name in ["cn-celeb2.tar.gz", "cn-celeb_v2.tar.gz"]:
        p = in_data / name
        if p.exists():
            p.unlink()


prepare_cn_celeb = PrepareSpec(
    multi_processor=_process_cnceleb_multi,
    file_extensions=[".flac"],
    output_lists=["listeTrain", "listeDev", "listeTest"],
    trials=TrialsSpec(converter_fn=_convert_cnceleb_trials),
    cleanup_fn=_cleanup_cnceleb,
)


# ── ESC-50 ────────────────────────────────────────────────────────


def _process_esc50(
    segment: Path, out_data: Path, sampling_frequency: int, vad: bool
) -> Tuple[str, str, str, Path]:
    name = segment.stem
    spkid = "audio"

    out_dir = out_data / "wav"
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / f"{name}.wav"

    if not output.exists():
        ffmpeg_to_wav(segment, output, sampling_frequency)

    duration = str(round(get_duration(output), 2))
    return name, spkid, duration, output


prepare_esc50 = PrepareSpec(
    file_processor=_process_esc50,
    file_extensions=[".wav"],
    search_root="ESC-50-master/audio",
)
