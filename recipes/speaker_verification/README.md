# 📊 Speaker Verification Results

All models listed below are trained on VoxCeleb2-dev (5,994 speakers, ~1M utterances). The VoxCeleb1-O/E/H test sets serve as in-domain evaluations, assessing performance under conditions similar to the training data. To measure generalization and cross-domain robustness, we additionally evaluate on out-of-domain datasets: CN-Celeb, DiPCo, and CommonBench.

The front-ends include SE-ResNet-200, WavLM-MHFA, ECAPA2, and Xi-Vector. All embeddings are 256-dimensional, trained for 51 epochs with a batch size of 512, and evaluated using cosine back-ends. The AM-Softmax loss (margin = 0.2, scale = 30) is used for all models except ECAPA2, which employs AAM-Softmax. For WavLM-MHFA, we use the WavLM-Large SSL encoder with frozen parameters during training.

We report performance using the Equal Error Rate (EER) metric.


| Model | vox1-O-clean | vox1-E-clean | vox1-H-clean | CN-Celeb | DiPCo | CommonBench |
|:------|:------:|:------:|:--:|:-------:|:---:|:------------:|
| ResNet-200 | 0.50 | 0.64 | 1.14 | 10.54 | 3.87 | 3.21 |
| WavLM-MHFA | 1.29 | 1.42 | 2.71 | 19.88 | 25.93 | 8.18 |
| ECAPA2 | 0.53 | 0.69 | 1.33 | 13.04 | 6.07 | 4.09 |
| Xi-Vector | 0.64 | 0.66 | 1.17 | 12.32 | 3.69 | 2.97 |



