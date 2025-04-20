# Kiwano Notebook: Managing Segments, Features & Augmentation

## Notebook Goals

This notebook will help you understand and use Kiwano’s segment management system for speaker verification. By the end, you’ll know how to:

- Load and organize a speaker-labeled audio dataset using SegmentSet
- Apply audio and feature-level data augmentations
- Extract filterbank features with Fbank
- Build a Dataset ready for training deep learning models


## Segments and the SegmentSet

### What is a Segment?

A Segment is Kiwano’s fundamental unit of data. It represents a single audio file segment associated with a speaker.

Each Segment stores:

- A unique ID (segmentid)
- A speaker label (spkid)
- Duration (in seconds)
- The file path to the audio
- An optional data augmentation

When audio is loaded with load_audio(), the segment:

- Reads the waveform using torchaudio
- Applies SpeedPerturb if defined
- Returns the waveform and sample rate

```python
segment = segments["id001"]
audio, sr = segment.load_audio()
```

### What is a SegmentSet?

SegmentSet is a dictionary-like container of Segment objects. It includes many utilities for managing your dataset.

Key Features:

- Load metadata from a text file (from_dict)
- Access segments by index or ID
- Generate speaker labels (get_labels)
- Apply preprocessing like truncation, speaker selection, or augmentation

Example usage:

```python
segments = SegmentSet()
segments.from_dict(Path("data/voxceleb1"))

segment = segments["id001"]
print(segment.spkid, segment.duration)
```

## Data Augmentation and Composition classes in Kiwano

Kiwano includes a flexible system for applying audio-level and feature-level augmentations to improve model robustness.


### Audio-level Augmentation (before feature extraction)

- Noise: Adds background noise (music/speech/other)
- Reverb: Adds reverberation (room impulse response)
- Normal:	Identity transform (no change)
- SpeedPerturb: Changes the playback speed of audio
- AddGaussianNoise:Adds white noise to the waveform
- Aliasing: Applies aliasing distortion

### Feature-level Augmentation (after feature extraction)

- SpecAugment: Time and frequency masking


### Composition Utilities for Augmentation


Kiwano supports composable and randomized transformation pipelines via three core components:


#### Compose

Applies a sequence of transforms one after another.

Usage example:
```python
Compose([
    CMVN(),
    Crop(350),
    SpecAugment(),
])
```

#### OneOf

Randomly pickup **one** transform from a list each time it's called.

Usage example:
```python
OneOf([
    Noise(musan_music, snr_range=[5, 15]),
    Reverb(reverb),
    Normal(),  # No-op
])
```

#### SomeOf

Randomly picks several of the given transforms each time it's called:

- Pick exactly N:

```python
SomeOf(2, [transform1, transform2, transform3])
```

- Pick between min and max:

```python
SomeOf((1, 3), [transform1, transform2, transform3])
```

- Pick at least N, up to all:

```python
SomeOf((2, None), [transform1, transform2, transform3, transform4])
```


## Feature Extraction

Kiwano provides its own feature extractor class, such as Fbank, which computes filterbank energies from raw audio.

```python
feature_extractor = Fbank()
features = feature_extractor.extract(audio, sampling_rate=sr)
```

## Putting It All Together


_SpeakerTrainingSegmentSet_ class combines segments, transforms, and feature extraction into a dataset suitable for model training:
- Loads segments from a manifest file
- Applies audio_transforms (optional)
- Extracts features using feature_extractor
- Applies feature_transforms (optional)
- Returns (features, speaker_label) for each call

```python
training_data = SpeakerTrainingSegmentSet(
    audio_transforms=OneOf([
        Noise(musan_music, snr_range=[5, 15]),
        Reverb(reverb),
        Normal()
    ]),
    feature_extractor=Fbank(),
    feature_transforms=Compose([
        CMVN(),
        Crop(350),
        SpecAugment(),
    ])
)

training_data.from_dict(Path("data/voxceleb1"))
features, label = training_data[0]
```



## Dataset Description and Stats

You can inspect the dataset with .describe():


```python
training_data.describe()
```

This will output:
- Total number of segments
- Number of unique speakers
- Cumulative duration of all audio
- Duration statistics (min, max, mean, std, percentiles)
