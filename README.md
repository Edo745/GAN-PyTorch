# GAN-PyTorch

## Overview
This project implements a Generative Adversarial Network (GAN) in PyTorch, following the paper "Generative Adversarial Nets" by Ian J. Goodfellow et al. CIFAR-10 dataset is used for training.

## Dependencies
- torch
- torchvision
- matplotlib

## Usage
1. Make sure you have the required dependencies installed.
2. Run `training.py` to train the GAN on the CIFAR-10 dataset.
3. Run `generate.py --checkpoint=path/to/checkpoint.pth` to generate images from the trained GAN.

## Colab Notebook
For a detailed walkthrough and experimentation, refer to the [Colab Notebook](https://colab.research.google.com/drive/1CB0APHpRRcdYZMjsEqe3fC6VywH7n7w7?usp=sharing).

## Results
Generated images and training progress will be saved in the `results` directory.

## References
- Goodfellow, I. J., et al. "Generative Adversarial Nets." [arXiv preprint arXiv:1406.2661 (2014)](https://arxiv.org/abs/1406.2661).

