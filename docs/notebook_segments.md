# Kiwano Notebook: Managing Segments, Features & Augmentation

## Notebook Goals

This notebook will help you understand and use Kiwano’s segment management system for speaker verification. By the end, you’ll know how to:

- Load and organize a speaker-labeled audio dataset using SegmentSet
- Apply audio and feature-level data augmentations
- Extract filterbank features with Fbank
- Build a Dataset ready for training deep learning models


## Dataset in Kiwano: Segment and SegmentSet


In Kiwano, datasets are described using manifest files in CSV format. These manifests are backed by a small set of Python classes that make it easy to manipulate, inspect, and augment the data programmatically. Kiwano's data representation is built around two core components:
- **Segment**: Represents a single audio sample linked to a specific speaker. It includes metadata such as the speaker ID, duration, file path, and optional augmentation configuration.
- **SegmentSet**: A collection of Segment objects. It acts as a high-level manifest that provides utilities for loading data from disk, accessing and filtering segments, and preparing data for training.

Think of **SegmentSet** as the manifest for your entire dataset, and each **Segment** as a single entry in that manifest.


### Segment: A Single Data Sample

A _Segment_ is the fundamental unit of data in Kiwano. It represents one audio segment tied to a specific speaker and contains all the metadata required for processing and training.


Each _Segment_ stores:
- _segmentid_: A unique identifier for the segment
- _spkid_: The speaker label (used as the training target)
- _duration_: Length of the audio segment in seconds
- _audio_path_: File path to the audio file
- _augmentation_ (optional): Information about any audio transform applied to the segment


Kiwano allows lazy loading of audio:
- Reads the waveform using torchaudio
- Applies SpeedPerturb if defined
- Returns the waveform and sample rate


Usage exemple:
```python
from kiwano import Segment

segment = Segment.from_file("meeting.wav")
audio, sr = segment.load_audio()
```


### SegmentSet: A Manifest of Segments

A _SegmentSet_ is a container that holds multiple Segment objects. It represents an entire dataset or corpus and provides high-level tools to manage it.

Think of it as:
- A manifest of your data
- A dictionary mapping segment IDs to Segment objects
- A toolkit for filtering, transforming, splitting, and inspecting your dataset

Main Features:
- Load data from a manifest file using from_dict()
- Access segments by ID or index (segments["id123"])
- Extract all speaker labels (segments.get_labels())
- Filter by duration, speaker, or any custom logic

Example usage:

```python
segments = SegmentSet()
segments.from_dict(Path("data/voxceleb1"))

segment = segments["id001"]
print(segment.spkid, segment.duration)
```

Together, Segment and SegmentSet give you a clean and efficient way to manage speaker verification datasets — from raw audio to model-ready features. These classes are inspired by the design principles of the Lhotse toolkit.


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
