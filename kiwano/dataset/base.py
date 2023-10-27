from typing import Union, TypeVar, List

from kiwano.utils import Pathlike
#from kiwano.augmentation import Augmentation
import random
import torchaudio
import copy
import numpy as np


class Segment():
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
        from kiwano.augmentation import SpeedPerturb

        self.augmentation = SpeedPerturb(factor)
        self.duration /= factor
        self.spkid = "speed"+str(factor)+"_"+self.spkid
        self.segmentid = "spped"+str(factor)+"_"+self.segmentid

    def load_audio(self, keep_memory: bool = False):
        audio_data, self.sample_rate = torchaudio.load(self.file_path)
        audio_data = audio_data[0]

        if self.augmentation != None:
            audio_data, self.sample_rate = self.augmentation(audio_data, self.sample_rate)

        if keep_memory == True:
            self.audio_data = audio_data
            self.sample_rate = self.sample_rate

        return audio_data, self.sample_rate


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
            spkid_dict[self.segments[key].spkid] = 0

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
        self.segments[segment.segmentid] = segment


    def truncate(self, duration: float):
        for key in list(self.segments):
            if self.segments[key].duration < duration:
                del self.segments[key]
        self.get_labels()

    def get_speaker(self, spkid: str):
        s = SegmentSet()

        for key in self.segments:
            if self.segments[key].spkid == spkid:
                s.append( self.segments[key] )

        self.get_labels()

        return s

    def perturb_speed(self, factor: float):
        c = SegmentSet()

        for key in self.segments:
            sset = copy.copy( self.segments[ key ] )
            sset.perturb_speed(factor)
            c.append( sset )
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
            #counter = 0
            for s in x:
                #print(str(counter)+" "+str(len(x)))
                #counter += 1
                self.segments[ s ] = x[ s ]
                #self.append( s )
        self.get_labels()


    def describe(self):
    #This function calculate and display several information about the segments
    # (number of different speakers, the total duration, min, max, mean...)


        listDuration = []
        differentSpeakers = {}
        totalDuration = 0

        for key in self.segments:
            duration = self.segments[key].duration
            totalDuration = totalDuration + duration

            listDuration.append(duration)
            differentSpeakers[ self.segments[key].spkid ] = True


        mean = np.mean(listDuration)
        max = np.max(listDuration)
        min = np.min(listDuration)
        std = np.std(listDuration)
        quartiles = np.quantile(listDuration, [0.25, 0.5, 0.75])

        print("Speaker count: ", len(differentSpeakers))
        print("Number of segments : ", len(self.segments))
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



