# Mixed-curvature Variational Auto-Encoders

### PyTorch implementation

## Overview

This repository contains a PyTorch implementation of the mixed-curvature variational autoencoder, or M-VAE,
as presented in [[1]](#citation). Also check out our blogpost (anonymized).
Code base is been based on [[2]](#citation).

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
* `mt/` - Source folder.
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
* `THIRD_PARTY.md` - List of third party software used in this project.

## Usage

To run training and inference, activate the created conda environment and run the examples:
```bash
conda activate pt
python -m mt.examples.run --dataset="mnist" --model="h2,s2,e2" --fixed_curvature=False 
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

For questions and comments, feel free to contact (anonymized).

## License

ASL 2.0

## Citation

```
[1] (anonymized)
[2] Davidson, T. R., Falorsi, L., De Cao, N., Kipf, T.,
and Tomczak, J. M. (2018). Hyperspherical Variational
Auto-Encoders. 34th Conference on Uncertainty in Artificial Intelligence (UAI-18).
```

BibTeX format:
```bibtex
(anonymized)
```
