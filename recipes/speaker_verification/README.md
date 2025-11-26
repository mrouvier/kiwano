# 🔊  Speaker Verification

This folder contains all the scripts and configuration files required to run speaker verification experiments using the VoxCeleb datasets. Additional scripts are provided for other datasets as well.


## 📁 Data Preparation (VoxCeleb2)

Run the prepare\_data.sh script, This script will automatically:

- Download and prepare the VoxCeleb1 and VoxCeleb2 datasets (used for training and evaluation)

- Download and prepare the MUSAN dataset (used for data augmentation with noise, music, and speech)

- Download and prepare the RIRS\_NOISES dataset (used to simulate reverberation conditions)

```
sh prepare_data.sh
```


## 🧠 Training Speaker Embeddings

You can train a speaker embedding extractor (ResNet-based) in different ways depending on your environment. All training logs and checkpoints will be saved under the specified experiment directory (e.g., exp/resnet/).

### Local Training


Run the following command to train ResNet locally:

```
torchrun --standalone --nproc_per_node=4 utils/train_resnet_ddp.py data/voxceleb2/ exp/resnet/
```

### Training with Hugging Face Accelerate


Run the following command to train ResNet using multi-GPU or distributed training, use the Accelerate toolkit:

```
accelerate launch utils/train_resnet_accelerate.py data/voxceleb2/ exp/resnet/
```

### Training with SLURM

Run the following command to train ResNet using SLURM:


```
sbatch train_resnet.sh
```


## 🎙️ Extracting Speaker Embeddings

Run the following command to extract speaker embedding:

```
python utils/extract_resnet.py data/voxceleb1/ exp/resnet/model51.ckpt pkl:exp/resnet/voxceleb1/xvector.0.pkl
```

Run the following command to extract speaker embedding using SLURM:
```
sbatch extract_xvector.sh exp/resnet/ 51
```

These scripts will generate .pkl files containing 256-dimensional embeddings for each utterance.

## 🧩 Evaluating Speaker Verification Performance

Run the following command to score speaker embedding on VoxCeleb-1 O/E/H :

```
sh compute_score.sh exp/resnet/ 51
```

Run the following command to score embedding on specific corpus :

```
python utils/compute_cosine.py  data/voxceleb1/voxceleb1-o-cleaned.trials  "pkl:cat exp/resnet/voxceleb1/xvector.*.pkl |"   "pkl:cat exp/resnet/voxceleb1/xvector.*.pkl |" > exp/resnet/voxceleb1/scores.txt
python utils/compute_eer.py data/voxceleb1/voxceleb1-o-cleaned.trials exp/resnet/voxceleb1/scores.txt
```



## 📊 Experiments Results

All models listed below are trained on VoxCeleb2-dev (5,994 speakers, ~1M utterances). The VoxCeleb1-O/E/H test sets serve as in-domain evaluations, assessing performance under conditions similar to the training data. To measure generalization and cross-domain robustness, we additionally evaluate on out-of-domain datasets: CN-Celeb, DiPCo, and CommonBench.

The front-ends include SE-ResNet-200, WavLM-MHFA, ECAPA2, and Xi-Vector. All embeddings are 256-dimensional, trained for 51 epochs with a batch size of 512, and evaluated using cosine back-ends. The AM-Softmax loss (margin = 0.2, scale = 30) is used for all models except ECAPA2, which employs AAM-Softmax. For WavLM-MHFA, we use the WavLM-Large SSL encoder with frozen parameters during training.

We report performance using the Equal Error Rate (EER) metric.


| Model | vox1-O-clean | vox1-E-clean | vox1-H-clean | CN-Celeb | DiPCo | CommonBench |
|:------|:------:|:------:|:--:|:-------:|:---:|:------------:|
| ResNet-200 | 0.50 | 0.64 | 1.14 | 10.54 | 3.87 | 3.21 |
| WavLM-MHFA | 1.29 | 1.42 | 2.71 | 19.88 | 25.93 | 8.18 |
| ECAPA2 | 0.53 | 0.69 | 1.33 | 13.04 | 6.07 | 4.09 |
| Xi-Vector | 0.64 | 0.66 | 1.17 | 12.32 | 3.69 | 2.97 |
