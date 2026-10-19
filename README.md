# Koopman Dynamics for Active Glassy Fluids

Data-driven analysis of the kinetic energy K(t) of a 2D Kob–Andersen active glass (N = 1000 particles)
as a function of the active force f: scaling and master curves, Koopman-based prediction of unseen
trajectories, HAVOK, and discovery of the governing differential equation.

**Main results: [docs/RESULTS.md](docs/RESULTS.md)**

## Physical Significance
Near the critical active force f_c (≈ 1.8–1.95), the kinetic energy decays slowly and almost
exponentially, then collapses abruptly when the system arrests. All trajectories with 1.0 ≤ f ≤ 1.8
collapse onto one master curve log K = a(f) + G(t/τ(f)) with a diverging time scale τ(f).

## Repository Structure
* `data/raw/`: kinetic-energy time series `avg_ke_1000_{f}` (N = 1000, f = active force).
* `src/data_loader.py`: loading and preprocessing.
* `src/hdmd_core.py`, `src/hdmd_analysis.py`: Hankel DMD.
* `src/deep_koopman.py`, `src/train.py`: force-conditioned deep Koopman autoencoder (one-step and rollout training).
* `src/scaling_analysis.py`: relaxation times, critical fits, master-curve collapse.
* `src/havok.py`: HAVOK (Hankel Alternative View Of Koopman).
* `src/equation_discovery.py`: weak-form SINDy, universal ODE, Langevin model.
* `src/koopman_forecast.py`: forecasting benchmark for unseen forces and from prefixes.

## Usage
Run from the project root, in this order:
```bash
python src/scaling_analysis.py
python src/havok.py
python src/equation_discovery.py
python src/koopman_forecast.py      # ~1 h on GPU; add --quick for a 4-force test
```
Dependencies: numpy, scipy, matplotlib, torch, scikit-learn, pydmd (cross-check), pysindy (cross-check).
