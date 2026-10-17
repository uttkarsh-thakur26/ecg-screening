#!/usr/bin/env bash
# Train the 1D ResNet: the overfit sanity check first, then the full run on folds 1-8 with early stopping on fold 9.
# Extra arguments go to the full run, e.g. --name no-amp-scale --set augmentation.amplitude_scale.enabled=false
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pytest tests/test_cuda.py -q
python -m src.training.train --overfit
python -m src.training.train "$@"
