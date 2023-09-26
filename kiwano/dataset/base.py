import numpy as np
from typing import Union
from kiwano.utils import Pathlike
import random
import torchaudio



class Segment():
    segmentid: str
    duration: float
    spkid: str
    file_path: str

    sample_rate: int
    audio_data: list

    def __init__(self, segmentid: str, spkid: str, duration: float, file_path: str):
        self.segmentid = segmentid
        self.spkid = spkid
        self.duration = duration
        self.file_path = file_path

        self.sample_rate = None
        self.audio_data = None

    def load_audio(self, keep_memory: bool = False):
        audio_data, self.sample_rate = torchaudio.load(self.file_path)

        if keep_memory == True:
            self.audio_data = audio_data[0]
            self.sample_rate = self.sample_rate

        return audio_data[0], self.sample_rate



class SegmentSet():
    def __init__(self):
        self.segments = {}
        self.labels = {}

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, segment_id_or_index: Union[int, str]) -> Segment:
        if isinstance(segment_id_or_index, str):
            return self.segments[segment_id_or_index]
        return next(val for idx, val in enumerate(self.segments.values()) if idx == segment_id_or_index)

    def load_audio(self):
        for key in self.segments:
            self.segments[key].load_audio(keep_memory=True)

    def get_labels(self):
        spkid_dict = {}
        self.labels = {}
        for key in self.segments:
            spkid_dict[ self.segments[ key ].spkid ] = 0

        for index, token in enumerate(spkid_dict.keys()):
            self.labels[token] = index

    def from_dict(self, target_dir: Pathlike):
        with open(target_dir / "liste") as f:
            for line in f:
                segmentid, spkid, duration, audio = line.strip().split(" ")
                self.segments[segmentid] = Segment(segmentid, spkid, (float)(duration), audio)
        self.get_labels()


    def get_random(self):
        name, segment = random.choice(list(self.segments.items()))
        return segment

    def append(self, segment: Segment):
        self.segments[ segment.segmentid ] = segment

    def get_speaker(self, spkid: str):
        s = SegmentSet()

        for key in self.segments:
            if self.segments[key].spkid == spkid:
                s.append( self.segments[key] )

        self.get_labels()

        return s

    def display(self):
        print(self.segments)

    def describe(self):
    #This function calculate and display several information about the segments
    # (number of different speakers, the total duration, min, max, mean...)


        listDuration = []
        differentSpeakers = []
        totalDuration = 0

        for key in self.segments:
            duration = self.segments[key].duration
            totalDuration = totalDuration + duration

            listDuration.append(duration)
            spkid = self.segments[key].spkid

            if spkid not in differentSpeakers:
                differentSpeakers.append(spkid)

        mean = np.mean(listDuration)
        max = np.max(listDuration)
        min = np.min(listDuration)
        std = np.std(listDuration)
        quartiles = np.quantile(listDuration, [0.25, 0.5, 0.75])

        print("Speaker count: ", len(differentSpeakers))
        print("Total duration (hours): ", totalDuration/3600)
        print("***")
        print("Duration statistics (seconds):")
        print("mean   ", mean)
        print("std    ", std)
        print("min    ", min)
        print("max    ", max)
        print("25%    ", quartiles[0])
        print("50%    ", quartiles[1])
        print("75%    ", quartiles[2])



