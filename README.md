# Koopman Dynamics for Active Glassy Fluids

This repository contains pipelines to extract the underlying linear dynamics of active glassy fluid relaxation using Koopman Operator Theory.

## Physical Significance
Near the critical active force ($f_c \approx 1.65$), the kinetic energy of an active glassy system decays non-linearly, exhibiting two-step relaxation (cage effects) and microscopic avalanches before jamming. By applying Takens' Embedding Theorem and Dynamic Mode Decomposition (DMD), we map this 1D time-series into a higher-dimensional space where the dynamics advance linearly. The eigenvalues extracted represent the inverse relaxation times ($1/\tau$).

## Repository Structure
* `data/raw/`: Place raw time-series kinetic energy logs here.
* `src/hdmd_core.py`: Exact mathematical formulation of Hankel DMD.
* `src/deep_koopman.py`: Neural network approach for highly non-linear threshold dynamics.

## Usage
Both scripts can be executed directly for a sanity check against hardware availability:
```bash
python src/hdmd_core.py
python src/deep_koopman.py
```
