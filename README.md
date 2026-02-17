# Interpretable Counterfactual Explanations Guided by Prototypes

This repository provides a PyTorch implementation of the paper "Interpretable Counterfactual Explanations Guided by Prototypes" (Van Looveren and Klaise, 2020).

Paper: [https://arxiv.org/pdf/1907.02584](https://arxiv.org/pdf/1907.02584)

## Overview

This implementation generates interpretable counterfactual explanations for image classification models using prototype-based guidance. The method combines autoencoders with prototype learning to produce plausible and class-representative counterfactual examples.

## Features

- Prototype-based counterfactual generation using k-means clustering
- Autoencoder-based plausibility constraints
- Support for MNIST dataset
- Interpretability metrics (IM1 and IM2)
- Visualization tools for latent space exploration
- TensorBoard integration for training monitoring

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Dependencies include:

- PyTorch
- torchvision
- matplotlib
- tqdm
- scikit-learn

## Project Structure

```
.
├── algorithm.py                    # Core counterfactual generation algorithm
├── CFG_main.py                     # Main execution script
├── train_class_aes_mnist.py        # Train class-specific autoencoders
├── train_global_ae_mnist.py        # Train global autoencoder
├── visualize.py                    # Visualization utilities
├── models/
│   ├── Autoencoder.py              # Autoencoder architecture
│   └── Model_MNIST.py              # MNIST classifier
├── metrics/
│   ├── IM1.py                      # Interpretability metric 1
│   └── IM2.py                      # Interpretability metric 2
├── utils/
│   └── ae_io.py                    # Autoencoder I/O utilities
├── weights/                        # Model checkpoints
├── logs/                           # Training logs
└── outputs/                        # Generated visualizations
```

## Usage

### Training

1. Train the global autoencoder:

```bash
python train_global_ae_mnist.py
```

2. Train class-specific autoencoders:

```bash
python train_class_aes_mnist.py
```

### Generating Counterfactuals

Run the main script to generate counterfactual explanations:

```bash
python CFG_main.py
```

Configuration options in `CFG_main.py`:

- `PROTO_METHOD`: Set to `"kmeans"` for prototype-based guidance or `None` for nearest neighbor
- `K_CLUSTERS`: Number of prototypes per class (default: 3)
- `AE_CHECKPOINT_DIR`: Directory containing model checkpoints
- `AE_GLOBAL_CHECKPOINT`: Path to global autoencoder checkpoint

### Visualization

Visualize counterfactual results and latent space using TSNE:

```bash
python visualize.py
```

## Algorithm

The counterfactual generation process optimizes the following objective:

- **Prediction Loss**: Ensures the counterfactual is classified as the target class
- **L1/L2 Loss**: Minimizes the perturbation from the original input
- **Prototype Loss**: Guides the counterfactual towards class prototypes in latent space
- **Autoencoder Loss**: Maintains plausibility through reconstruction

The main objective function to optimize is:

$$\min_{\delta} \; c \cdot L_{\text{pred}} + \beta \|\delta\|_1 + \|\delta\|_2^2 + L_{AE} + L_{\text{proto}}$$

## Metrics

- **IM1**: Measures consistency with class-specific autoencoder
- **IM2**: Measures plausibility using global autoencoder reconstruction

## Results

Generated counterfactuals and visualizations are saved in the `outputs/` directory. Training logs and metrics are available in the `logs/` directory and can be viewed using TensorBoard.

## References

```bibtex
@article{van2019interpretable,
  title={Interpretable counterfactual explanations guided by prototypes},
  author={Van Looveren, Arnaud and Klaise, Janis},
  journal={arXiv preprint arXiv:1907.02584},
  year={2019}
}
```

## Author

BAIM M. Jalal
