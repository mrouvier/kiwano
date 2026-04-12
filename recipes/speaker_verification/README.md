# 🔊  Speaker Verification

This folder contains all the scripts and configuration files required to run speaker verification experiments using the VoxCeleb datasets. Additional scripts are provided for other datasets as well.

## 📦 Dataset Management (CLI)

Kiwano provides a unified command-line interface to list, download, and prepare all supported corpora. No corpus-specific scripts are needed — the same three commands work for every dataset.

### Listing Available Datasets

To see which corpora are available and what operations are supported for each:

```
python -m kiwano.corpus.cli list
```

This prints a table showing each dataset, whether it supports automatic download, and whether a preparation pipeline is defined:

```
Available corpora:
dataset      download  prepare  description
---------    --------  -------  -----------
voxceleb1    ✓         ✓        Large-scale celebrity speech corpus for speaker verification...
voxceleb2    ✓         ✓        Expanded celebrity speech corpus with more speakers...
musan        ✓         ✓        Music, speech, and noise corpus for data augmentation.
rirs_noises  ✓         ✓        Simulated and real room impulse responses...
cn_celeb     ✓         ✓        Chinese celebrity speech corpus...
esc50        ✓         ✓        Environmental sound dataset with 50 balanced classes...
but_reverb   ✓         ✓        Real room impulse responses...
hi_mia       ✓         —        Mandarin wake-word and speaker corpus...
arab_celeb   ✓         —        Arabic celebrity speech corpus collected from YouTube...
```

### Downloading a Dataset

To download a corpus, specify its name and a target directory:

```
python -m kiwano.corpus.cli download voxceleb2 db/voxceleb2/
```

The pipeline automatically handles multi-part archives (e.g. VoxCeleb split files), checksum verification, reassembly, and extraction. It is **resumable**: if interrupted, re-running the same command will skip already-downloaded files.

Available options:

| Option | Description |
|:-------|:------------|
| `--num-jobs N` | Number of parallel threads for extraction (default: 4) |
| `--verify` | Run checksum verification after download |
| `--force` | Re-download everything, ignoring cached state |

Examples:

```bash
# Download VoxCeleb1 with checksum verification and 8 threads
python -m kiwano.corpus.cli download voxceleb1 db/voxceleb1/ --verify --num-jobs 8

# Download MUSAN noise corpus
python -m kiwano.corpus.cli download musan db/musan/

# Download RIRS_NOISES impulse responses
python -m kiwano.corpus.cli download rirs_noises db/rirs_noises/

# Download CN-Celeb (two independent groups, handled automatically)
python -m kiwano.corpus.cli download cn_celeb db/cn_celeb/

# Force re-download of VoxCeleb2
python -m kiwano.corpus.cli download voxceleb2 db/voxceleb2/ --force
```

### Preparing a Downloaded Dataset

Once a corpus is downloaded, the `prepare` command converts raw files to the format expected by the training scripts: it generates a `liste` manifest file and optionally resamples audio, applies VAD, and converts trial files.

```
python -m kiwano.corpus.cli prepare voxceleb2 db/voxceleb2/ data/voxceleb2/
```

This reads raw audio from `db/voxceleb2/` (the download directory) and writes processed files and manifests to `data/voxceleb2/`.

Available options:

| Option | Description |
|:-------|:------------|
| `--num-jobs N` | Number of parallel worker processes (default: 20) |
| `--downsampling RATE` | Target sampling rate in Hz (default: 16000) |
| `--vad` | Apply Silero VAD to strip silence (VoxCeleb1/2 only) |
| `--delete-archives` | Remove archive files after preparation to save disk space |

Examples:

```bash
# Prepare VoxCeleb1 with VAD and 16 workers
python -m kiwano.corpus.cli prepare voxceleb1 db/voxceleb1/ data/voxceleb1/ --vad --num-jobs 16

# Prepare VoxCeleb2 (converts .m4a to .wav via FFmpeg)
python -m kiwano.corpus.cli prepare voxceleb2 db/voxceleb2/ data/voxceleb2/ --num-jobs 20

# Prepare MUSAN at 8kHz for narrowband experiments
python -m kiwano.corpus.cli prepare musan db/musan/ data/musan_8k/ --downsampling 8000

# Prepare RIRS_NOISES (filters smallroom and mediumroom only)
python -m kiwano.corpus.cli prepare rirs_noises db/rirs_noises/ data/rirs_noises/

# Prepare CN-Celeb (generates three lists: listeTrain, listeDev, listeTest)
python -m kiwano.corpus.cli prepare cn_celeb db/cn_celeb/ data/cn_celeb/ --num-jobs 20

# Prepare and clean up archives
python -m kiwano.corpus.cli prepare voxceleb2 db/voxceleb2/ data/voxceleb2/ --delete-archives
```

### Programmatic API

The same functionality is available from Python:

```python
from kiwano.corpus.cli import download_dataset, prepare_dataset

# Download
download_dataset("voxceleb2", "data/voxceleb2/", verify=True, num_jobs=16)

# Prepare
prepare_dataset("voxceleb1", "data/voxceleb1/", "data/voxceleb1/", vad=True, num_jobs=20)
```

### Full Workflow Example

A typical setup for a VoxCeleb experiment downloads four corpora and prepares them:

```bash
# 1. Download everything
python -m kiwano.corpus.cli download voxceleb1  db/voxceleb1/  --verify --num-jobs 8
python -m kiwano.corpus.cli download voxceleb2  db/voxceleb2/  --verify --num-jobs 8
python -m kiwano.corpus.cli download musan      db/musan/      --verify
python -m kiwano.corpus.cli download rirs_noises db/rirs_noises/ --verify

# 2. Prepare all datasets
python -m kiwano.corpus.cli prepare voxceleb1   db/voxceleb1/   data/voxceleb1/   --vad --num-jobs 20
python -m kiwano.corpus.cli prepare voxceleb2   db/voxceleb2/   data/voxceleb2/   --num-jobs 20
python -m kiwano.corpus.cli prepare musan       db/musan/       data/musan/
python -m kiwano.corpus.cli prepare rirs_noises db/rirs_noises/ data/rirs_noises/
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
