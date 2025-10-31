# Lightweight Augmentation & Pipelines for Speaker Verification

## Quick Start

```python
import torch
from typing import List

# Pipelines
train_wav_aug = Compose([
    OneOf([AddGaussianNoise(0.001, 0.01), Codec(), MediaMP3()]),
    SomeOf(num_transforms=(1, 2), transforms=[
        SpeedPerturb(1.1), SignFlip(0.5), ClippingDistortion(10, 30)
    ]),
    Reverb(segs=rir_segment_set),         # requires your SegmentSet
    Noise(segs=noise_segment_set, snr_range=[5, 20]),
])

# Waveform example (16 kHz mono)
wav = torch.randn(16000)
wav_aug, sr = train_wav_aug(wav, 16000)

# Feature-space pipeline (e.g., log-mel: [time, freq])
feat_aug = Compose([CMVN(norm_means=True, norm_vars=True),
                    SpecAugment(num_t_mask=2, num_f_mask=2, max_t=30, max_f=8, prob=1.0),
                    Crop(duration=200, random=True)])
feat = torch.randn(300, 80)
feat = feat_aug(feat)
```


## Overview

### Pipelines: `Compose`, `OneOf`, `SomeOf`

- **`Compose(transforms: List[Augmentation])`**
  - Applies all transforms sequentially.
  - Accepts either `(tensor)` **or** `(tensor, sample_rate)`; every transform in the chain must accept the same signature and return the same.

- **`OneOf(transforms: List[Augmentation])`**
  - Uniformly picks **one** transform and applies it.

- **`SomeOf(num_transforms: int | (min, max), transforms: List[Augmentation])`**
  - Samples **k** transforms **without replacement** and applies them in ascending index order.
  - `num_transforms=n` ⇒ exactly *n*.  
    `num_transforms=(a,b)` ⇒ random *k ∈ [a,b]*; `(a,)` ⇒ *k ∈ [a, len(transforms)]*.

> All pipelines pass through either a **single tensor** or a **(tensor, sr)** pair. Stick to one signature consistently inside a pipeline.

---

### Transforms (waveform)

All waveform transforms return `(tensor, sample_rate)`.

- **`AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015)`**  
  i.i.d. Gaussian noise with σ sampled from range.

- **`ClippingDistortion(min_percentile_threshold=0, max_percentile_threshold=40)`**  
  Percentile-based symmetric clipping (defaults mimic light distortion). Uses an internal percentile approximation.

- **`Aliasing(original_sr: int, target_sr: int)`**  
  Downsample (linear interpolation, no anti-alias) to random SR in mel-space between `target_sr` and `original_sr`, then upsample back. Simulates aliasing artifacts.

- **`SpeedPerturb(factor: float)`**  
  Uses `torchaudio.sox_effects` to apply `speed` then `rate` back to the original SR. Factor > 1 speeds up; < 1 slows down.

- **`Normal()`**  
  Identity (no-op). Useful placeholder / branch element.

- **`B12Attack(mu=0.08, sigma2=2.5e-5, amplitude_threshold=0.05)`**  
  Segment, filter by mean amplitude, randomly permute segments (see Duroselle et al., ASVspoof5).

- **`Noise(segs: SegmentSet, snr_range: List[int])`**  
  Mix a random noise segment at target SNR. Repeats or crops noise to match length. Expects your `SegmentSet` to expose `.get_random().load_audio()`.

- **`Reverb(segs: SegmentSet)`**  
  Convolve with a random RIR; power-normalize and clamp to [-1, 1]. Expects RIR `SegmentSet`.

- **`Filtering()`**  
  Random band-pass or band-reject via `sox` effects.

- **`Codec()`**  
  Stochastic codec stress (μ-law, A-law, MP3, Vorbis) via `torchaudio.functional.apply_codec`.

- **`Codec16kHzWithEffector(...)`**  
  16 kHz preset using `apply_codec` with randomized params (incl. optional commented `AudioEffector` flow). Useful for aggressive codec robustness.

- **`MediaG722` / `MediaVorbis` / `MediaOpus` / `MediaMP3`**  
  Encode/decode with `torchaudio.io.AudioEffector` (when available) to create realistic media artifacts. Parameters randomized per call.

- **`SignFlip(flip_prob=0.5)`**  
  Random sign inversion.

- **`AirAbsorption` (heuristic)**  
  Frequency-dependent attenuation vs distance/temperature/humidity using a simple model in STFT domain.

- **`AirAbsorption` (tabulated)**  
  Frequency-band attenuation from lookup tables (10/20 °C; humidity bins), applied via STFT with interpolation to FFT bins.

---

### Transforms (feature-space)

Operate on `[time, freq]` style tensors and return a **tensor only**.

- **`SpecAugment(num_t_mask=1, num_f_mask=1, max_t=10, max_f=8, prob=0.6)`**  
  Time/frequency masking in-place style.

- **`CMVN(norm_means=True, norm_vars=False)`**  
  Mean subtraction and optional variance normalization along **time**.

- **`Crop(duration: int, random: bool=True)`**  
  Temporal crop (repeat-pad if needed when `random=True`; left-crop otherwise).

---


