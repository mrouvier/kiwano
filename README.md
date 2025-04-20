# Kiwano

Kiwano is an advanced open-source toolkit for speaker verification, built on PyTorch and designed for both research and real-world applications. It provides a modular and extensible framework that includes:

- High-performance speaker embedding architectures such as ResNet, ECAPA-TDNN, ECAPA2, ReDimNet.
- A suite of speaker embedding preprocessing tools for normalization (AS-Norm, AD-Norm...), enhancement, and more.
- Audio data augmentation methods to improve model robustness in diverse environments
- Ready-to-use recipes for popular datasets like VoxCeleb, CN-Celeb...

Kiwano makes it easy to build, train, and evaluate speaker verification systems. Whether you're working on academic research, industrial applications, or competition pipelines, Kiwano helps you develop state-of-the-art models with ease. Its integration with PyTorch ensures flexibility and scalability, while its clean toolkit supports reproducible experiments and efficient workflows.


# Installation


## Installing Kiwano


- First, clone the repo :

```bash
git clone https://github.com/mrouvier/kiwano
```

- Second, install the requirements :

```bash
cd kiwano
pip install -r requirements.txt
```

- Third, install kiwano in the environment :

```bash
cd kiwano
pip install -e .
```

## Recipes

There are recipes for several architectures in the `./recipes` directory.


# License

Kiwano is released under the Apache License, version 2.0. The Apache license is a popular BSD-like license. Kiwano can be redistributed for free, even for commercial purposes, although you can not take off the license headers (and under some circumstances, you may have to distribute a license document). Apache is not a viral license like the GPL, which forces you to release your modifications to the source code. Note that this project has no connection to the Apache Foundation, other than that we use the same license terms.
