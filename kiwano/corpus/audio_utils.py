"""
Audio loading, resampling, and VAD utilities for corpus preparation.
"""

import logging
from pathlib import Path
from subprocess import PIPE, run
from typing import Optional

import numpy as np
import torch
import soundfile as sf

log = logging.getLogger(__name__)

# ── Duration ──────────────────────────────────────────────────────

def get_duration(path: Path) -> float:
    """Return audio duration in seconds."""
    info = sf.info(str(path))
    return info.frames / info.samplerate


# ── FFmpeg conversion ─────────────────────────────────────────────


_FFMPEG_BASE = "ffmpeg -y -threads 1 -i {src} -acodec pcm_s16le -ac 1 -ar {sr} -ab 48 -threads 1"

def ffmpeg_to_wav(src: Path, dst: Path, sampling_rate: int = 16000) -> None:
    """Convert/resample `src` to a WAV file at `dst`."""
    cmd = _FFMPEG_BASE.format(src=src, sr=sampling_rate) + f" {dst}"
    proc = run(cmd, shell=True, stdout=PIPE, stderr=PIPE)
    if proc.returncode != 0:
        log.warning("FFmpeg returned %d for %s", proc.returncode, src.name)


def ffmpeg_to_tensor(src: Path, sampling_rate: int = 16000) -> torch.Tensor:
    """Convert `src` to a float32 tensor [-1, 1] via FFmpeg pipe (no temp file)."""
    cmd = _FFMPEG_BASE.format(src=src, sr=sampling_rate) + " -f wav -"
    proc = run(cmd, shell=True, stdout=PIPE, stderr=PIPE)
    audio_array = np.frombuffer(proc.stdout, dtype=np.int16)
    return torch.tensor(audio_array.astype(np.float32) / 32768.0)


# ── Silero VAD ────────────────────────────────────────────────────

# Loaded once per worker process to avoid reloading for each file.
_vad_model: Optional[object] = None


def get_vad_model():
    """Return the cached Silero VAD model (lazy-loaded)."""
    global _vad_model
    if _vad_model is None:
        from silero_vad import load_silero_vad
        _vad_model = load_silero_vad()
    return _vad_model


def apply_vad(wav: torch.Tensor, sampling_rate: int = 16000, threshold: float = 0.6) -> torch.Tensor:
    """
    Apply Silero VAD and return only speech segments.

    If no speech is detected, returns the original tensor unchanged.
    """
    from silero_vad import collect_chunks, get_speech_timestamps
    model = get_vad_model()
    timestamps = get_speech_timestamps(wav, model, sampling_rate=sampling_rate, threshold=threshold)
    if timestamps:
        return collect_chunks(timestamps, wav)
    log.warning("VAD: no speech detected — keeping original audio.")
    return wav


# ── Combined load: resample + optional VAD → tensor ──────────────

def load_audio(
    src: Path,
    sampling_rate: int = 16000,
    use_vad: bool = False,
) -> torch.Tensor:
    """
    Load an audio file as a tensor, optionally resampling and applying VAD.

    Uses FFmpeg for resampling when ``sampling_rate != 16000`` or when
    the source file's native rate differs from the target. For 16 kHz
    targets, reads directly with soundfile for efficiency.
    """
    if sampling_rate != 16000:
        wav = ffmpeg_to_tensor(src, sampling_rate)
    else:
        data, _ = sf.read(str(src), dtype="float32", always_2d=False)
        wav = torch.tensor(data if data.ndim == 1 else data[:, 0])

    if use_vad:
        wav = apply_vad(wav, sampling_rate)

    return wav

