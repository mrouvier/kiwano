# Getting Started with Kiwano

New to _Kiwano_? You’re in the right place! This guide will walk you through installation and getting ready to run speaker verification recipes.


## 📘 About

**Kiwano** is an advanced open-source toolkit for **speaker verification**, built on **PyTorch** and designed to support both **academic research** and **real-world applications**.


### Main goals


- Provide a **modular**, **extensible**, and **reproducible** framework for speaker verification research.
- Support **cutting-edge speaker embedding models**, including:
  - ResNet
  - ECAPA-TDNN
  - ECAPA2
  - ReDimNet
  - WavLM + MHFA
- Include robust preprocessing tools for:
  - Normalization (AS-Norm, AD-Norm, etc.)
  - Enhancement and augmentation of audio data
- Enable quick prototyping and experimentation with **ready-to-use recipes**.




### Tutorials

_Coming soon_ — A series of hands-on guides that will help you:

- Set up and preprocess speaker datasets
- Run and modify training recipes
- Customize model architectures and training parameters
- Extend the framework with your own scoring/back-end logic


### Examples of use

Kiwano is designed for a variety of speaker verification use-cases, including:

- Academic experiments and benchmarks
- Industrial speaker recognition systems
- Robust evaluation pipelines for challenges or competitions
- Research on normalization and augmentation techniques

Whether you're building a baseline model or experimenting with novel architectures, Kiwano provides the tools to do it effectively.



## 🛠 Installation

Kiwano requires **Python 3.8+** and **PyTorch** (GPU support is highly recommended for training).

To install the latest Kiwano release:


### 1. Clone the repository

```bash
git clone https://github.com/mrouvier/kiwano
```

### 2. Install required dependencies

```bash
cd kiwano
pip install -r requirements.txt
```

### 3. Install Kiwano in editable mode


```bash
pip install -e .
```

This is an editable installation (-e option), meaning that your changes to the source code are automatically reflected when importing kiwano (no re-install needed).


## 📄 License

Kiwano is released under the **Apache License, version 2.0**. The Apache license is a popular BSD-like license. Kiwano can be redistributed for free, even for commercial purposes, although you can not take off the license headers (and under some circumstances, you may have to distribute a license document). Apache is not a viral license like the GPL, which forces you to release your modifications to the source code. Note that this project has no connection to the Apache Foundation, other than that we use the same license terms.
