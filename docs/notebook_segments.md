# Kiwano Notebook: Dataset, Data Augmentation & Features Extractor

## 🎯 Objective of the Notebook

This notebook will help you understand and use Kiwano’s segment management system for speaker verification. By the end, you’ll know how to:

- Load and organize a speaker-labeled audio dataset using SegmentSet
- Apply audio and feature-level data augmentations
- Extract filterbank features with Fbank
- Build a Dataset ready for training deep learning models


## 📁 Dataset in Kiwano: Segment and SegmentSet


In Kiwano, datasets are described using manifest files in CSV format. These manifests are backed by a concise set of Python classes designed to facilitate the programmatic manipulatation, inspection, and augmentation of data. The data representation in Kiwano is structured around two fundamental components:
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


Usage exemple:
```python
from kiwano.dataset import Segment

segment = Segment.from_file("meeting.wav")
audio, sr = segment.load_audio()
```


### SegmentSet: A Manifest of Segments

A _SegmentSet_ is a structured container that holds multiple _Segment_ instances. It represents an entire dataset or corpus and provides high-level tools to manage it. Think of it as:
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
from kiwano.dataset import Segment, SegmentSet

segments = SegmentSet()
segments.from_dict(Path("data/voxceleb1"))

segment = segments["id001"]
print(segment.spkid, segment.duration)
```

Together, Segment and SegmentSet give you a clean and efficient way to manage speaker verification datasets — from raw audio to model-ready features. These classes are inspired by the design principles of the Lhotse toolkit.


## 🎧 Data Augmentation and Composition classes in Kiwano

Kiwano includes a modular data augmentation that supports both audio-level and feature-level transformations to increase the robustness and generalizability of speaker recognition models.



### Audio-level Augmentation (before feature extraction)

These transformations operate directly on the raw audio waveform and are primarily intended to simulate real-world acoustic variability or introduce diversity into the training data:

- **Noise**: Simulates various acoustic environments by adding background noise to the waveform, targeting a specific signal-to-noise ratio (SNR).
- **Reverb**: Applies reverberation effects using convolution with Room Impulse Responses (RIRs) to emulate different room acoustics.
- **Normal**: An identity transformation that returns the waveform unchanged. Often used as a placeholder in augmentation pipelines.
- **SpeedPerturb**: Alters the playback speed of the audio to simulate variations in speech rate, serving as a simple yet effective data augmentation technique.
- **AddGaussianNoise**: Injects Gaussian noise into the audio signal, with the noise intensity controlled by a random standard deviation sampled from a defined range (min_amplitude to max_amplitude).
- **Aliasing**: Introduces aliasing distortion to simulate lossy signal artifacts.
- **AirAbsorption**: Models the attenuation of sound in air due to distance, temperature, and humidity, mimicking propagation effects in real-world environments.
- **ClippingDistortion**: Applies audio distortion by threshold-based clipping, using percentile-based bounds to introduce non-linear amplitude artifacts.
- **B12Attack**: Implements a data augmentation technique that randomly segments and rearranges portions of the audio signal to increase temporal variability.
- **SignFlip**: Randomly inverts the sign of the waveform with a given probability, serving as a simple signal-level transformation.


### Feature-level Augmentation (after feature extraction)

These transformations are applied to time-frequency representations of the audio signal (e.g., Mel filterbanks or spectrograms), introducing variability directly at the feature level:


- **SpecAugment**: An augmentation method that applies time and frequency masking to the spectrogram, encouraging robustness to missing information.
- **CMVN (Cepstral Mean and Variance Normalization)**: Normalizes the features by removing mean and scaling variance across time, enhancing model stability and reducing speaker/session variability.
- **Crop**: Performs temporal cropping of the feature matrix, typically used to enforce uniform input length or simulate segment-level variability.


### Composition Utilities for Augmentation


Kiwano supports composable and randomized transformation pipelines via three core components:


#### Compose

Applies a sequence of transforms one after another.

Usage example:
```python
from kiwano.augmentation import Compose

Compose([transform1(), transform2(), transform3()])
```

#### OneOf

Randomly pickup **one** transform from a list each time it's called.

Usage example:
```python
from kiwano.augmentation import OneOf

OneOf([transform1(), transform2(), transform3()])
```

#### SomeOf

Randomly picks several of the given transforms each time it's called:

- Pick exactly N:

```python
from kiwano.augmentation import SomeOf

SomeOf(2, [transform1(), transform2(), transform3()])
```

- Pick between min and max:

```python
from kiwano.augmentation import SomeOf

SomeOf((1, 3), [transform1(), transform2(), transform3()])
```

- Pick at least N, up to all:

```python
from kiwano.augmentation import SomeOf

SomeOf((2, None), [transform1(), transform2(), transform3(), transform4()])
```


## 🎛️ Feature Extraction

Kiwano provides its own feature extractor class, such as Fbank, which computes filterbank energies from raw audio.

```python
from kiwano.features import Fbank

feature_extractor = Fbank()
features = feature_extractor.extract(audio, sampling_rate=sr)
```

## 🧠 Putting It All Together


_SpeakerTrainingSegmentSet_ class combines segments, transforms, and feature extraction into a dataset suitable for model training:
- Loads segments from a manifest file
- Applies audio_transforms (optional)
- Extracts features using feature_extractor
- Applies feature_transforms (optional)
- Returns (features, speaker_label) for each call

```python
from kiwano.augmentation import (
    CMVN,
    Augmentation,
    Compose,
    Crop,
    Normal,
    OneOf,
    Reverb,
    SpecAugment,
    AirAbsorption,
    AddGaussianNoise,
    ClippingDistortion
)
from kiwano.dataset import Segment, SegmentSet
from kiwano.features import Fbank
from kiwano.utils import Pathlike


class SpeakerTrainingSegmentSet(Dataset, SegmentSet):
    def __init__(
        self,
        audio_transforms: List[Augmentation] = None,
        feature_extractor=None,
        feature_transforms: List[Augmentation] = None,
    ):
        super().__init__()
        self.audio_transforms = audio_transforms
        self.feature_transforms = feature_transforms
        self.feature_extractor = feature_extractor
    def __getitem__(self, segment_id_or_index: Union[int, str]) -> Segment:
        segment = None
        if isinstance(segment_id_or_index, str):
            segment = self.segments[segment_id_or_index]
        else:
            segment = next(
                val
                for idx, val in enumerate(self.segments.values())
                if idx == segment_id_or_index
            )
        audio, sample_rate = segment.load_audio()
        if self.audio_transforms != None:
            audio, sample_rate = self.audio_transforms(audio, sample_rate)
        if self.feature_extractor != None:
            feature = self.feature_extractor.extract(audio, sampling_rate=sample_rate)
        if self.feature_transforms != None:
            feature = self.feature_transforms(feature)
        return feature, self.labels[segment.spkid]


training_data = SpeakerTrainingSegmentSet(
    audio_transforms=OneOf(
        [
            AirAbsorption(),
            AddGaussianNoise(),
            ClippingDistortion(),
            Normal(),
        ]
    ),
    feature_extractor=Fbank(),
    feature_transforms=Compose(
        [
            CMVN(),
            Crop(350),
            SpecAugment(),
        ]
    ),
)

training_data.from_dict(Path("data/voxceleb2/"))
features, label = training_data[0]
```



## 📊 Dataset Description and Stats

You can inspect the dataset with .describe():


```python
training_data.describe()
```

This will output:
- Total number of segments
- Number of unique speakers
- Cumulative duration of all audio
- Duration statistics (min, max, mean, std, percentiles)
