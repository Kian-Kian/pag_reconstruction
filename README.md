# Prior-austenite grain reconstruction

Implementation of *Prior-austenite grain reconstruction through physics-informed residual learning and graph partitioning*.

The pipeline constructs a grain adjacency graph from an alpha-phase EBSD map, estimates boundary probabilities with physics-informed residual MLP and GNN models, partitions the graph using greedy additive edge contraction, and estimates a parent orientation for each region. Inference uses an alpha CTF file and the supplied model assets.

## Contents

- `pag_recon/`: graph construction, crystallography, training and reconstruction.
- `data/ctf_train/`: 100 paired development maps.
- `data/ctf_test/`: 10 paired independent test maps.
- `models/`: five MLP and five GNN2 checkpoints, normalization statistics and orientation relationships.
- `configs/`: reconstruction parameters and development split memberships.
- `docs/`: method parameters and input conventions.
- `tests/`: automated checks.

## Installation

Use Python 3.10 or newer. Run commands from this directory in a virtual environment:

```bash
python -m pip install -e ".[test]"
```

The local CPU verification environment uses Python 3.12.0 and the versions in `requirements-tested.txt`.

## Reconstruct a map

```bash
python -m pag_recon reconstruct --input data/ctf_train/alpha/001.ctf --models models --output runs/reconstruction --device cpu
```

Choose a new output directory for each run. The alpha phase identifier defaults to 1. The output contains:

- `reconstructed.ctf`: reconstructed parent orientations.
- `reconstruction.npz`: grain and region identifiers, edge probabilities and orientation diagnostics.
- `run.json`: execution settings, input hashes and reconstruction counts.

The output CTF defines Austenite as phase 1. Its default lattice parameter is 3.66 angstrom and can be set using `--parent-lattice-angstrom`. Region identifiers are stored in the NPZ file.

## Train models

```bash
python -m pag_recon prepare-training --data data --output runs/prepared
python -m pag_recon.train_folds --prepared runs/prepared --output runs/trained --device cpu
```

Preparation builds paired grain graphs and fold-specific orientation relationships. Training fits feature normalization on the training maps and selects checkpoints using inner-validation average precision. The resulting `runs/trained/models` directory can be supplied to the reconstruction command. Use `--device cuda` with a compatible CUDA-enabled PyTorch installation for GPU execution.

The MLP uses hidden width 96, 40 epochs and initial learning rate 0.0003. GNN2 uses hidden width 128, two message-passing layers, 120 epochs and initial learning rate 0.001. Boundary probabilities are combined with MLP weight 0.4 and GNN2 weight 0.6. Further settings are documented in [method configuration](docs/method.md).

## Data

Only original CTF files are included in `data/`. Each dataset contains `alpha/` and `gamma/` directories; matching filenames identify registered pairs. Gamma files contain AZtecCrystal-derived reference reconstructions used for supervision and evaluation. The independent test maps are reserved for final evaluation and excluded from training and parameter selection.

See [data description](data/README.md) and [input format](docs/input_format.md) for conventions.

## Tests

```bash
python -m pytest tests -q
```

Tests cover CTF parsing and export, grain graphs, cubic symmetry, orientation relationships, residual learning, cross-validation splits, graph partitioning and parent-orientation estimation.
