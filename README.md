# Hier_PPO_LSTM
# Hier_PPO_LSTM

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15250811.svg)](https://doi.org/10.5281/zenodo.15250811)

This repository contains the code for **Hierarchical PPO+LSTM** applied to multi‑UAV mission planning in discrete grid environments. The project implements a two‑tier reinforcement‑learning framework where a high‑level manager selects coarse waypoints and a shared LSTM‑backboned PPO worker executes collision‑aware motion primitives under partial observability.

## Overview

- Manager: selects grid waypoints every K steps, optimising long‑horizon objectives.
- Worker: a PPO agent with LSTM memory that executes primitive actions to reach waypoints while avoiding obstacles and respecting battery constraints.
- Curriculum optimisation: pre‑train the worker on small grids, freeze it, then train the manager on larger maps.

## Repository Structure

```
Hier_PPO_LSTM/
├── Environment.py             # MultiUAVEnv environment implementation
├── Grid.py                    # Grid and obstacle utilities
├── manager_env.py             # Wrapper for macro-action interface
├── PPO_LSTM_Train.py          # Worker pre-training script
├── train_hierarchical.py      # Manager training script
├── plot_training_results.py   # Scripts to plot convergence curves
├── policies/                  # Saved policy checkpoints
├── Train folder/              # CSV logs from training and evaluation
└── requirements.txt           # Python dependencies
```

## Requirements

- Python 3.10
- numpy
- gymnasium
- ray[rllib]==2.7
- torch

Install via:
```bash
pip install -r requirements.txt
```

## Usage

### Worker Pre‑Training
```bash
python PPO_LSTM_Train.py --config scenario=small --output ./models/worker_small
```

### Hierarchical Training
```bash
python train_hierarchical.py --worker-model ./models/worker_small \
    --scenario medium --output ./models/manager_medium
```

### Evaluation and Plotting
```bash
python plot_training_results.py --logs "Train folder/*_results*.csv"
```

## Configuration

Training parameters and environment settings are defined in `config_params` within the training scripts. You can adjust:

- `scenario`: small, medium, large
- `K`: macro horizon (default 20)
- PPO hyper‑parameters (learning rate, batch size, discount, etc.)



## License

This project is licensed under the MIT License. See `LICENSE` for details.

## Contact

For questions or access to private data/models, please contact:

Alparslan Güzey
Email: alparslanguzey@gmail.com

