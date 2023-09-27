#!/usr/bin/env python3

import sys
from pathlib import Path
from typing import Optional, Union, List

import numpy as np
import torch
import time
from torch import nn

from kiwano.utils import Pathlike
from kiwano.features import Fbank
from kiwano.augmentation import Augmentation, Noise, Codec, Filtering, Normal, Sometimes, Linear, CMVN, Crop
from kiwano.dataset import Segment, SegmentSet
from kiwano.model import ResNet



#trials

def read_xvector(file_path: str):
    h = {}
    f = open(file_path, "r")
    for line in f:
        line = line.strip().split(" ")
        arr = line[1:-1]
        my_list = [float(i) for i in arr]
        h[ line[0] ] = torch.Tensor(my_list)
    f.close
    return h



if __name__ == '__main__':
    enrollment = read_xvector(sys.argv[2])
    test = read_xvector(sys.argv[3])


    print(enrollment)



