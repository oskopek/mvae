# Mixed-curvature Variational Auto-Encoders

![Python Package using Conda](https://github.com/oskopek/mvae/workflows/Python%20Package%20using%20Conda/badge.svg)

### PyTorch implementation

## Overview

This repository contains a PyTorch implementation of the Mixed-curvature Variational Autoencoder, or M-VAE,
as presented in [[1]](#citation). For the arXiv paper, please see: https://arxiv.org/abs/1911.08411.

## Installation

Install Python 3.7+.
To install all dependencies, make sure you have installed [conda](https://docs.conda.io/en/latest/miniconda.html), and run

```bash
make conda
conda activate pt
make download_data
```


## Structure

* `chkpt/` - Checkpoints for trained models.
* `data/` - Data folder. Contains a script necessary for downloading the datasets, and the downloaded data.
* `mathematica/` - Mathematica scripts (various formula derivations, etc).
* `mt/` - Source folder (stands for Master Thesis).
  * `data/` - Data loading, preprocessing, batching, and pre-trained embeddings.
  * `examples/` - Contains the main executable file. Reads flags and runs the corresponding training and/or evaluation.
  * `mvae/` - Model directory. Note that models heavily use inheritance!
  * `test_data/` - Data used for testing.
  * `visualization/` - Utilities for visualization of latent spaces or training statistics.
* `plots/` - Folder to store generated plots.
* `scripts/` - Contains scripts to run experiments and plot the results.
* `tests/` - (A few) unit tests.
* `Makefile` - Defines "aliases" for various tasks.
* `README.md` - This manual.
* `LICENSE` - Apache Standard License 2.0.
* `environment.yml` - Required Python packages.
* `THIRD_PARTY.md` - List of third party software used in this thesis.

## Usage

To run training and inference, activate the created conda environment and run the examples:

```bash
conda activate pt

# MNIST:
python -m mt.examples.run --dataset="mnist" --model="h2,s2,e2" --fixed_curvature=False

# CIFAR:
python -m mt.examples.run --dataset="cifar" --model="h2,s2,e2" --fixed_curvature=False --h_dim=8192 --architecture="conv"
```

Take a look at `mt/examples/run.py` for a list of command line arguments.

For an evaluation run, see `mt/examples/eval.py`.

Please cite [[1](#citation)] in your work when using this repository in your experiments.

### Other make commands

```bash
make clean   # format source code
make check   # check for formatting and code errors
make test    # run tests
```

## Feedback

For questions and comments, feel free to contact [Ondrej Skopek](mailto:oskopek@oskopek.com).

## License

ASL 2.0

## Citation

Ondrej Skopek, Octavian-Eugen Ganea, Gary Bécigneul. Mixed-curvature Variational Autoencoders. International Conference on Learning Representations (ICLR) 2020. URL https://openreview.net/forum?id=S1g6xeSKDS

BibTeX format:
```bibtex
@inproceedings{skopek2020mixedcurvature,
  title={Mixed-curvature Variational Autoencoders},
  author={Ondrej Skopek and Octavian-Eugen Ganea and Gary B{\'e}cigneul,
  booktitle={International Conference on Learning Representations},
  year={2020},
  url={https://openreview.net/forum?id=S1g6xeSKDS}
}
```
