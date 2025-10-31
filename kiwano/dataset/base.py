import copy

import random
from typing import List, TypeVar, Union

import numpy as np
import torchaudio

from kiwano.utils import Pathlike


class Segment:
    """
    Atomic audio unit used by Kiwano recipes.

    A `Segment` carries minimal metadata (segment id, speaker id, duration in
    seconds, file path) and optionally the loaded waveform + sample rate.
    Speed perturbation can be *logically* attached to the segment and will be
    applied on-the-fly during `load_audio()`.

    Examples
    --------
    Basic usage

    >>> seg = Segment("id0001_00001", "spk0001", 3.20, "/path/to/utt.wav")
    >>> x, sr = seg.load_audio()   # lazy load; applies augmentation if configured
    >>> x.shape[0] / sr            # ~ actual duration in seconds

    Attaching speed perturbation (1.1x faster)

    >>> seg = Segment("id0002_00001", "spk0002", 2.75, "/p/utt.wav")
    >>> seg.perturb_speed(1.1)     # mutates ids + scales duration
    >>> x, sr = seg.load_audio()

    Caching waveform in memory

    >>> seg = Segment("id0003_00001", "spk0003", 4.0, "/p/utt.wav")
    >>> _ = seg.load_audio(keep_memory=True)
    >>> seg.audio_data is not None
    True
    """

    segmentid: str
    duration: float
    spkid: str
    file_path: str

    sample_rate: int
    audio_data: list

    augmentation: None

    def __init__(self, segmentid: str, spkid: str, duration: float, file_path: str):
        self.segmentid = segmentid
        self.spkid = spkid
        self.duration = duration
        self.file_path = file_path

        self.sample_rate = None
        self.audio_data = None

        self.augmentation = None

    def perturb_speed(self, factor: float):

        self.augmentation = factor
        self.duration /= factor
        self.spkid = "speed" + str(factor) + "_" + self.spkid
        self.segmentid = "speed" + str(factor) + "_" + self.segmentid

    def load_audio(self, keep_memory: bool = False):
        from kiwano.augmentation import SpeedPerturb

        audio_data, self.sample_rate = torchaudio.load(self.file_path)
        audio_data = audio_data[0]

        if self.augmentation != None:
            s = SpeedPerturb(self.augmentation)
            audio_data, self.sample_rate = s(audio_data, self.sample_rate)

        if keep_memory == True:
            self.audio_data = audio_data
            self.sample_rate = self.sample_rate

        return audio_data, self.sample_rate


class SegmentSet:
    """
    In-memory container for `Segment` objects with utilities for
    selection, augmentation, label mapping, and dataset statistics.

    Attributes
    ----------
    segments : Dict[str, Segment]
        Mapping from `segmentid` to `Segment`.
    labels : Dict[str, int]
        Mapping from `spkid` to contiguous class index.

    Examples
    --------
    Build from a manifest

    >>> # 'liste' file with lines: "<segmentid> <spkid> <duration_sec> <audio_path>"
    >>> sset = SegmentSet()
    >>> sset.from_dict("data/voxceleb2/")   # expects data/voxceleb2/liste
    >>> len(sset), list(sset.labels)[:3]

    Sampling and loading waveforms

    >>> seg = sset.get_random()
    >>> wav, sr = seg.load_audio()

    Speed perturb an entire set (1.1x)

    >>> sp110 = sset.perturb_speed(1.1)     # new SegmentSet
    >>> assert len(sp110) == len(sset)

    Filter by duration and recompute labels

    >>> sset.truncate(min_duration=2.0, max_duration=12.0)
    >>> min(s.duration for s in sset.segments.values()) >= 2.0
    True
    """
    def __init__(self):
        self.segments = {}
        self.labels = {}

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, segment_id_or_index: Union[int, str]) -> Segment:
        if isinstance(segment_id_or_index, str):
            return self.segments[segment_id_or_index]
        return next(
            val
            for idx, val in enumerate(self.segments.values())
            if idx == segment_id_or_index
        )

    def load_audio(self):
        for key in self.segments:
            self.segments[key].load_audio(keep_memory=True)

    def get_labels(self):
        spkid_dict = {}
        self.labels = {}
        for key in self.segments:
            spkid_dict[self.segments[key].spkid] = 0

        for index, token in enumerate(spkid_dict.keys()):
            self.labels[token] = index

    def from_dict(self, target_dir: Pathlike):
        with open(target_dir / "liste") as f:
            for line in f:
                segmentid, spkid, duration, audio = line.strip().split(" ")
                self.segments[segmentid] = Segment(
                    segmentid, spkid, (float)(duration), audio
                )
        self.get_labels()

    def get_random(self):
        name, segment = random.choice(list(self.segments.items()))
        return segment

    def append(self, segment: Segment):
        self.segments[segment.segmentid] = segment

    def truncate(self, min_duration: float, max_duration: float):
        for key in list(self.segments):
            d = False
            if self.segments[key].duration > max_duration:
                d = True

            if self.segments[key].duration < min_duration:
                d = True

            if d == True:
                del self.segments[key]

        self.get_labels()

    def get_speaker(self, spkid: str):
        s = SegmentSet()

        for key in self.segments:
            if self.segments[key].spkid == spkid:
                s.append(self.segments[key])

        self.get_labels()

        return s

    def perturb_speed(self, factor: float):
        c = SegmentSet()

        for key in self.segments:
            sset = copy.copy(self.segments[key])
            sset.perturb_speed(factor)
            c.append(sset)
        c.get_labels()

        return c

    def display(self):
        print(self.segments)

    def copy(self):
        return copy.deepcopy(self)

    def __iter__(self):
        for key in self.segments:
            yield key

    def combine(self, l: List):
        for x in l:
            for s in x:
                self.segments[s] = x[s]
        self.get_labels()

    def describe(self):
        listDuration = []
        differentSpeakers = {}
        totalDuration = 0

        for key in self.segments:
            duration = self.segments[key].duration
            totalDuration = totalDuration + duration

            listDuration.append(duration)
            differentSpeakers[self.segments[key].spkid] = True

        mean = np.mean(listDuration)
        max = np.max(listDuration)
        min = np.min(listDuration)
        std = np.std(listDuration)
        quartiles = np.quantile(listDuration, [0.25, 0.5, 0.75])

        print("Speaker count: ", len(differentSpeakers))
        print("Number of segments : ", len(self.segments))
        print("Total duration (hours): ", totalDuration / 3600)
        print("***")
        print("Duration statistics (seconds):")
        print("mean   ", mean)
        print("std    ", std)
        print("min    ", min)
        print("max    ", max)
        print("25%    ", quartiles[0])
        print("50%    ", quartiles[1])
        print("75%    ", quartiles[2])
