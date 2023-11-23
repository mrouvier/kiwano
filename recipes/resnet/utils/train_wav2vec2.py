import argparse
import os
import pdb
from pathlib import Path

import hostlist
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import Wav2Vec2Tokenizer

from kiwano.dataset import SegmentSet
from kiwano.model import ECAPAModel
from kiwano.model.wav2vec2 import CustomWav2Vec2Model
from recipes.resnet.utils.train_resnet import SpeakerTrainingSegmentSet
import pdb
from kiwano.augmentation import Augmentation, Noise, Codec, Filtering, Normal, Sometimes, Linear, CMVN, Crop, \
    SpecAugment, Reverb, CropWaveForm
import torch
import sys


class Wav2Vec2Dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def custom_collate_fn(batch):
    inputs_batch, labels_batch = zip(*batch)

    # Transform 2d to 1d
    inputs_batch = [item.squeeze(dim=0) for item in inputs_batch]
    # Pad sequences to the length of the longest sequence in the batch
    inputs_batch = torch.nn.utils.rnn.pad_sequence(inputs_batch, batch_first=True)

    # Create attention mask
    attention_mask = (inputs_batch != tokenizer.pad_token_id).float()

    return {
        'input_values': inputs_batch,
        'attention_mask': attention_mask,
        'labels': torch.tensor(labels_batch)
    }


if __name__ == '__main__':
    hostnames = hostlist.expand_hostlist(os.environ['SLURM_JOB_NODELIST'])
    os.environ["MASTER_ADDR"] = hostnames[0]
    os.environ["MASTER_PORT"] = "29500"
    rank = int(os.environ["SLURM_NODEID"])
    world = int(os.environ["SLURM_JOB_NUM_NODES"])
    master_addr = hostnames[0]
    port = int(os.environ["MASTER_PORT"])

    device = torch.device("cuda")

    torch.distributed.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=world)

    print("START Loading data")
    sys.stdout.flush()
    musan = SegmentSet()
    musan.from_dict(Path("data/musan/"))

    musan_music = musan.get_speaker("music")
    musan_speech = musan.get_speaker("speech")
    musan_noise = musan.get_speaker("noise")

    model_name = "facebook/wav2vec2-base-960h"
    model_wav2vec2 = CustomWav2Vec2Model(model_name)
    model_wav2vec2 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_wav2vec2)
    model_wav2vec2.to(device)
    model_wav2vec2 = torch.nn.parallel.DistributedDataParallel(model_wav2vec2)

    tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_name)
    training_data = SpeakerTrainingSegmentSet(
        audio_transforms=Sometimes([
            Noise(musan_music, snr_range=[5, 15]),
            Noise(musan_speech, snr_range=[13, 20]),
            Noise(musan_noise, snr_range=[0, 15]),
            Codec(),
            Filtering(),
            Normal()
        ]),
        feature_extractor=model_wav2vec2,
        feature_transforms=Linear([
            CropWaveForm(3)
        ]),
    )
    training_data.from_dict(Path("data/voxceleb1/"))
    train_sampler = DistributedSampler(training_data, num_replicas=dist.get_world_size(), rank=dist.get_rank(),
                                       shuffle=True)

    train_dataloader = DataLoader(training_data, batch_size=32, drop_last=True, shuffle=False, num_workers=15,
                                  sampler=train_sampler, pin_memory=True)
    iterator = iter(train_dataloader)
    print("END Loading data")
    sys.stdout.flush()

    wav2vec2_outputs = []

    # train_dataloader = DataLoader(training_data, batch_size=48, drop_last=True, shuffle=True, num_workers=10,
    #                              collate_fn=custom_collate_fn)

    # The wav2vec2 output
    print(f"START Wav2vec2 ")
    sys.stdout.flush()
    for i, (feats, iden) in enumerate(train_dataloader, start=1):
        print(f"Batch: {i}")
        sys.stdout.flush()
        feats = feats.float().to(device)
        iden = iden.to(device)
        with torch.cuda.amp.autocast(enabled=True):
            preds = model_wav2vec2(feats, iden)
            wav2vec2_outputs.extend(preds)

    print(f"END Wav2vec2")
    sys.stdout.flush()
    wav2vec2_dataset = Wav2Vec2Dataset(wav2vec2_outputs)
    train_dataloader = DataLoader(wav2vec2_dataset, batch_size=128, drop_last=True, shuffle=True, num_workers=10)

    num_iterations = 5

    ecapa_tdnn_model = ECAPAModel(
        lr=0.001,
        lr_decay=0.97,
        channel_in=768,
        channel_size=1024,
        n_class=6000,
        loss_margin=0.2,
        loss_scale=30,
        test_step=1
    )
    print(f"START ECAPA-TDNN {num_iterations} iterations")
    sys.stdout.flush()
    for epoch in range(1, num_iterations + 1):
        print(f"\t [{epoch} / {num_iterations}]")
        sys.stdout.flush()
        loss, lr, acc = ecapa_tdnn_model.train_network(epoch=epoch, loader=train_dataloader)

    print(f"END ECAPA-TDNN")
    sys.stdout.flush()
