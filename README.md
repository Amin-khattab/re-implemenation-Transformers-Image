# Re-implementation: Transformers for Image Classification

A comparative study of **Vision Transformers (ViT)** and **CNN-based architectures** on image classification, featuring scratch-built implementations and benchmarks against pre-trained models.

## Overview

This repository explores the question: *how do Vision Transformers stack up against well-established CNN architectures on standard image classification benchmarks?* It contains from-scratch implementations of both paradigms, along with fine-tuning pipelines for pre-trained variants, evaluated on CIFAR-10.

## Repository Structure

├── CNN_Models/
│   └── ResNet50.py - CNN.py             # ResNet-50 implementation / training
├── VIT_Models/
│   └── pre-trained-transformer.py   # Pre-trained ViT fine-tuning
└── pipe_line.py - LNA.py               # Shared data loading & training pipeline



## Models

### Vision Transformers (ViT)
Transformer-based architecture that treats images as sequences of patches. Each image is split into fixed-size patches, linearly embedded, augmented with positional encodings, and passed through standard transformer encoder blocks.

### ResNet-50 (CNN baseline)
Deep residual network with skip connections — a strong, well-understood CNN baseline for comparison against the attention-based ViT approach.

## Dataset

Experiments are run on **CIFAR-10** (60,000 32×32 color images across 10 classes).

## Getting Started

### Requirements
```bash
pip install torch torchvision transformers numpy matplotlib
```

### Running the pipeline
```bash
# Train/evaluate ResNet-50
python CNN_Models/ResNet50.py

# Fine-tune pre-trained ViT
python VIT_Models/pre-trained-transformer.py

# Unified pipeline
python pipe_line.py
```

## Goals

- Implement ViT and CNN architectures from the ground up
- Benchmark both paradigms on the same dataset under comparable conditions
- Analyze trade-offs in accuracy, training time, and parameter efficiency
- Understand where attention-based models outperform convolutions — and where they don't

## Topics

`transformers` · `cnn` · `vit` · `cifar10` · `resnet-50` · `vision-transformer`

## License

Add a license of your choice (MIT, Apache 2.0, etc.).

## Author

[@Amin-khattab](https://github.com/Amin-khattab)
