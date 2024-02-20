# Kiwano

Kiwano is an advanced open-source toolkit for speaker verification based on PyTorch.


# Entraînement et l'Évaluation

## Les Paramètres

Lors de l'utilisation des scripts pour l'entraînement et l'évaluation avec ECAPA-TDNN, plusieurs paramètres peuvent être configurés. Voici une liste de ces paramètres avec leurs descriptions :

- `--eval_list`: Chemin vers la liste des données d'évaluation. Par défaut: `db/voxceleb1/veri_test2.txt` (vox1_e)
- `--eval_path`: Chemin vers le répertoire contenant les données d'évaluation. Par défaut: `db/voxceleb1/wav/`
- `--n_cpu`: Nombre de CPU à utiliser pour l'entraînement. Par défaut: 20
- `--batch_size`: Taille des échantillons. Par défaut: 400
- `--max_epoch`: Nombre maximal d'itérations. Par défaut: 80
- `--test_step`: Nombre d'itérations après lesquelles il faut enregistrer le modèle. Par défaut: 1 (après chaque itération)
- `--save_path`: Répertoire où enregistrer le dernier modèle, le meilleur modèle et les scores. Par défaut: `exps/exp1`
  - `exps/exp1/model`: Contiendra le meilleur et le dernier modèle
  - `exps/exp1/score.txt`: Contiendra les scores d'évaluation
- `--initial_model`: Modèle spécifique à partir duquel continuer l'entraînement. Par défaut: Aucune valeur (l'entraînement commence à partir du dernier modèle enregistré)
- `--musan_list_path`: Chemin vers la liste de musan. Par défaut: `data/musan/`
- `--rirs_noise_list_path`: Chemin vers la liste de rirs_noises. Par défaut: `data/rirs_noises/`
- `--training_list_path`: Chemin vers la liste des fichiers d'entraînement. Par défaut: `data/voxceleb2/`
- `--feat_type`: Type de l'extracteur des caractéristiques acoustiques à utiliser (fbank, wav2vec2, wavlm, hubert). Par défaut: `fbank`
- `--feat_dim`: Nombre de filtres à utiliser (à spécifier si feat_type = fbank). Par défaut: 80
- `--model_name`: Nom du modèle utilisé (à spécifier si vous utilisez un SSL comme extracteur de caractéristique). Par défaut: `facebook/wav2vec2-base-960h`
- `--eval`: Effectuer une évaluation. Si spécifié, `--initial_model` doit également être spécifié.

## Entraînement

### Avec un seul GPU

#### À partir du dernier modèle

- **FBank (origine) + ECAPA-TDNN**
```bash
python3 utils/train_ecapa_tdnn.py --save_path exps/exp1 --feat_type fbank --feat_dim 80 --n_cpu 10 --batch_size 128
```

Lance l'entraînement de l’ECAPA-TDNN avec l'extracteur de caractéristiques de type filter bank du projet d'origine utilisant 80 filtres. L'entraînement utilise 10 CPUs avec une taille d'échantillonnage de 128 et enregistre les résultats dans `exps/exp1`.

Vous pouvez specifier une autre valeur `feat_dim` si vous le souhaitez
- **FBank (kiwano) + ECAPA-TDNN**
```bash
python3 utils/train_ecapa_tdnn_2.py --save_path exps/exp2 --feat_type fbank --feat_dim 81 --n_cpu 10 --batch_size 128
```
Lance l'entraînement de l’ECAPA-TDNN avec  l'extracteur de caractéristiques de type filter bank de kiwano utilisant 81 filtres. L'entraînement utilise 10 CPUs avec une taille d'échantillonnage de 128 et enregistre les résultats dans `exps/exp2`.

Il faut noter ici que `feat_dim` pour kiwano ne marche qu'avec `81` filtres

- **Wav2Vec2 + ECAPA-TDNN**
```bash
python3 utils/train_ecapa_tdnn.py --save_path exps/exp3 --feat_type wav2vec2 --n_cpu 10 --batch_size 128 --model_name facebook/wav2vec2-base-960h
```
Lance l'entraînement de l’ECAPA-TDNN avec l'extracteur de caractéristiques de type wav2vec2 utilisant le modèle `facebook/wav2vec2-base-960h`. L'entraînement utilise 10 CPUs avec une taille d'échantillonnage de 128 et enregistre les résultats dans `exps/exp3`.

Vous pouvez effectuer l'entraînement avec d'autre modèles wav2vec2:

  - `facebook/wav2vec2-large-960h`
  - `facebook/wav2vec2-large-robust-ft-libri-960h`
  - `facebook/wav2vec2-large-960h-lv60-self`



- **WavLM + ECAPA-TDNN**
```bash
python3 utils/train_ecapa_tdnn.py --save_path exps/exp4 --feat_type wavlm --n_cpu 10 --batch_size 128 --model_name microsoft/wavlm-base
```
Lance l'entraînement de l’ECAPA-TDNN avec l'extracteur de caractéristiques de type wavlm utilisant le modèle `microsoft/wavlm-base`. L'entraînement utilise 10 CPUs avec une taille d'échantillonnage de 128 et enregistre les résultats dans `exps/exp4`.

Vous pouvez effectuer l'entraînement avec d'autre modèles wavlm: `microsoft/wavlm-base-plus`,`microsoft/wavlm-base-plus-sv`, `microsoft/wavlm-base-sv`, `microsoft/wavlm-large`


- **HuBERT + ECAPA-TDNN**
```bash
python3 utils/train_ecapa_tdnn.py --save_path exps/exp5 --feat_type hubert --n_cpu 10 --batch_size 128 --model_name microsoft/wavlm-base
```
Lance l'entraînement de l’ECAPA-TDNN avec l'extracteur de caractéristiques de type hubert utilisant le modèle `microsoft/wavlm-base`. L'entraînement utilise 10 CPUs avec une taille d'échantillonnage de 128 et enregistre les résultats dans `exps/exp4`.

Vous pouvez effectuer l'entraînement avec d'autre modèles hubert: `facebook/hubert-base-ls960`, `facebook/hubert-xlarge-ls960-ft`, `facebook/hubert-large-ls960-ft`, `facebook/hubert-xlarge-ll60k`, `facebook/hubert-large-ll60k`

#### À partir d'un modèle spécifique

```bash
python3 utils/train_ecapa_tdnn.py --save_path exps/exp1 --feat_type fbank --feat_dim 80 --n_cpu 10 --batch_size 128 --initial_model exps/exp1/model/model_0003.model
```

L'entraînement commence à partir du modèle `model_0003.model`.

### Avec plusieurs GPU

#### À partir du dernier modèle

```bash
python3 utils/train_ecapa_tdnn_ddp.py --save_path exps/exp1_ddp --feat_type fbank --feat_dim 80 --n_cpu 10 --batch_size 128
```

L'entraînement se fait sur tous les GPUs disponibles et utilise le filter bank d'origine fourni avec ECAPA-TDNN.

#### À partir d'un modèle spécifique

```bash
python3 utils/train_ecapa_tdnn_ddp.py --save_path exps/exp1_ddp --feat_dim 81 --n_cpu 10 --batch_size 128 --initial_model exps/exp1_1_ddp/model/model_0003.model
```

L'entraînement commence à partir du modèle `model_0003.model`.

## Évaluation

```bash
python3 utils/train_ecapa_tdnn.py --eval --initial_model exps/exp1_ddp/model/model_0001.model --feat_dim 81 --n_cpu 10
```

Cette commande évalue le modèle `model_0001.model`.

## Avec SLURM

Des exemples de sript slurm se trouvent dans le répertoire `resnet`

- `train_ecapa.sh`: contient le script pour entraîner l'ecapa-tdnn avec un filter bank sur sur un seul gpu
- `train_ecapa_ddp.sh`: permet l'entraînement sur plusieurs GPU



# License

Kiwano is released under the Apache License, version 2.0. The Apache license is a popular BSD-like license. Kiwanocan be redistributed for free, even for commercial purposes, although you can not take off the license headers (and under some circumstances, you may have to distribute a license document). Apache is not a viral license like the GPL, which forces you to release your modifications to the source code. Note that this project has no connection to the Apache Foundation, other than that we use the same license terms.


# Citing Kiwano

Please, cite Kiwano if you use it :









