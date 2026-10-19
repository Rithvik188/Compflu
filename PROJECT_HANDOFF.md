# Project Handoff: Decoding Dense Active Fluids Through Machine Learning

---

## 1. Project Overview

This project studies a **2D active glass** system — a dense assembly of self-propelled particles that are driven persistently out of thermodynamic equilibrium by an active force $f$. The central physics question is: *what happens to the kinetic energy of this system as the active force approaches a critical threshold?*

The system undergoes a **jamming transition** at a critical active force $f_c$. Below this value, the system's kinetic energy decays to zero (the system "arrests" into a jammed state). Above it, the system flows indefinitely as an active fluid.

> [!IMPORTANT]
> **Updated (Oct 2026):** the original estimate $f_c \approx 1.65$ was a finite-simulation-time artifact. f = 1.65 arrests (t ≈ 23 600) and f = 1.8 is still in the slow decay at the end of the run. Current estimate: **$f_c \approx 1.8$–$1.95$, $\gamma \approx 1.4$–$2.1$** (degenerate). Full results are in `docs/RESULTS.md`. The project uses data-driven techniques — Hankel Dynamic Mode Decomposition (HDMD) and a Force-Conditioned Deep Koopman Autoencoder — to characterize this transition and predict kinetic energy trajectories.

---

## 2. Physical System: The Kob-Andersen Active Glass

### Model Details
- **System:** 2D binary mixture of particles (Type A and Type B)
- **Composition:** 65:35 ratio (A:B) — this is the canonical **Kob-Andersen (KA) glass-forming mixture**
- **Interactions:** Lennard-Jones (LJ) potential between all pairs (AA, AB, BB) with different $\epsilon$ and $\sigma$ parameters
- **Activity:** Each particle is subjected to a persistent **active force** of tunable magnitude $f$. This is the single control parameter of the study.
- **Primary Observable:** **Mean Kinetic Energy** $K(t)$ as a function of simulation time $t$.

### Raw Data Files
- **Location:** `data/raw/`
- **Naming Convention:** `avg_ke_1000_{force}` where `{force}` is the active force value
- **Format:** Plain text, 3 columns: `[time, kinetic_energy, (third column discarded)]`
- **Available Active Forces:** `[0.0, 0.25, 0.5, 0.75, 1.0, 1.05, 1.15, 1.25, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.8, 2.0]` — **17 trajectories total**

### Physical Regimes
| Force Range | Behaviour |
|---|---|
| $f \le 0.75$ | **Quench-like:** fast decay with oscillations; outside the master curve. |
| $1.0 \le f \le 1.65$ | **Jammed / Glassy:** slow exponential decay of K, then an abrupt collapse at a force-independent rate (arrest). Arrest time grows from ~1 800 to ~23 600. |
| $f = 1.8$ | **Near-critical:** still in the slow decay at t = 29 505 (τ_K ≈ 19 000); expected to arrest at t ~ 10⁵. |
| $f = 2.0$ | **Active Fluid (nearly stationary):** KE ≈ 0.020, Gaussian fluctuations. |

### Key Physical Discovery: Critical Slowing Down
As $f \to f_c$, the characteristic relaxation time $\tau$ diverges following a **power law**:
$$\tau \propto (f_c - f)^{-\gamma}$$
Our regression on the HDMD-extracted relaxation times yields:
$$\boxed{\gamma \approx 1.64 \pm 0.31, \quad R^2 \approx 0.70}$$
This is notably lower than the standard Mode-Coupling Theory value of $\gamma \approx 2.3$–$2.6$ for passive 3D glasses, suggesting that active cage-breaking changes the universality class. **This is an exploratory observation, not a confirmed claim** — it motivates the next step of equation extraction.

> [!WARNING]
> **Superseded.** This fit assumed $f_c = 1.65$ and used HDMD on linear K. With $f_c$ fitted (above every force used) and several τ definitions, the robust result is $f_c \approx 1.8$–$1.95$ and $\gamma \approx 1.4$–$2.1$, with Arrhenius rejected and power law ≈ VFT. See `src/scaling_analysis.py` and `docs/RESULTS.md` §1.

---

## 3. Project Directory Structure

```
C:\Users\rithv\OneDrive\Desktop\ML MODEL FOR ACTIVE GLASSY FLUIDS\
│
├── README.md
│
├── data/
│   ├── raw/                  ← 17 raw simulation files (avg_ke_1000_{f})
│   └── processed/
│       └── koopman_model.pth ← Saved trained PyTorch model weights
│
├── docs/                     ← All output figures (publication-ready PNGs)
│   ├── pub_critical_slowing_down.png
│   ├── pub_master_trajectories.png
│   ├── relaxation_time_scaling.png
│   └── scaling_exploration.png
│
├── notebooks/                ← (empty, kept for future Jupyter exploration)
│
└── src/                      ← All Python source code
    ├── data_loader.py        (+ preprocess / load_clean)
    ├── hdmd_core.py
    ├── hdmd_analysis.py
    ├── deep_koopman.py       (+ rollout())
    ├── train.py              (+ train_rollout(): multi-step loss)
    ├── publication_plots.py
    ├── demo.py
    ├── scaling_analysis.py   ← Q1: τ definitions, critical fits, master-curve collapse
    ├── havok.py              ← HAVOK (own implementation, Lorenz + PyDMD validated)
    ├── equation_discovery.py ← weak SINDy, master-curve ODE, Langevin, PySINDy cross-check
    └── koopman_forecast.py   ← Q2: leave-one-force-out + prefix forecasting benchmark
```

Run the new analysis in order (from the project root): `scaling_analysis.py` → `havok.py` → `equation_discovery.py` → `koopman_forecast.py` (the last takes ~1 h on the RTX 3060; `--quick` runs 4 forces). Results: `docs/RESULTS.md`.

---

## 4. Codebase: File-by-File Breakdown

### `src/data_loader.py`
**Purpose:** Parses all raw simulation files into Python.

**Key functions:**
- `load_and_preprocess_data(raw_dir)` — Globs `avg_ke_1000_*`, parses the force from the filename, loads the `(time, ke)` columns, and returns a `dict` mapping `float(force) → (time_array, ke_array)`.
- `create_train_test_split(dataset, test_forces)` — Splits the dict into train and test dicts by specifying which forces to hold out.

**Usage:**
```python
from data_loader import load_and_preprocess_data, create_train_test_split
dataset = load_and_preprocess_data("data/raw")
train, test = create_train_test_split(dataset, test_forces=[0.5, 1.25, 1.6, 1.8])
```

---

### `src/hdmd_core.py`
**Purpose:** Core implementation of Hankel Dynamic Mode Decomposition.

**Class:** `KoopmanHDMD(delay_window, rank, use_gpu=True)`

**How it works:**
1. Takes a 1D kinetic energy time series $K(t)$.
2. Builds a **Hankel matrix** by stacking time-delay shifted copies of the signal. This embeds the 1D signal into a high-dimensional space where the dynamics are approximately linear (the Koopman embedding).
3. Performs **SVD** on the Hankel matrix to find the dominant dynamical modes.
4. Extracts the **discrete-time Koopman eigenvalues** $\lambda_i$ from the reduced operator $\tilde{A}$.
5. The relaxation time is extracted from the dominant eigenvalue: $\tau = -1 / \ln(|\lambda_{\text{dominant}}|)$

**Parameters used in practice:**
- `delay_window=1000` (captures the long-time dynamics)
- `rank=10` (keeps the 10 most significant dynamical modes)

---

### `src/hdmd_analysis.py`
**Purpose:** Loops over all forces in the jamming regime ($f < 1.65$) and extracts $\tau$ for each using `KoopmanHDMD`.

**Key function:** `extract_relaxation_times()` — returns two numpy arrays: `forces` and `taus`.

**How to run:**
```bash
cd "C:\Users\rithv\OneDrive\Desktop\ML MODEL FOR ACTIVE GLASSY FLUIDS"
python src/hdmd_analysis.py
```
Outputs: `docs/relaxation_time_scaling.png`

> [!IMPORTANT]
> This file uses relative imports (`from data_loader import ...`). Always run scripts from the project root directory, or add `src/` to `sys.path` when running inline (`sys.path.append('src')`).

---

### `src/deep_koopman.py`
**Purpose:** Defines the neural network architecture — a **Force-Conditioned Deep Koopman Autoencoder**.

**Class:** `ForceConditionedKoopmanAE(input_dim=100, latent_dim=10, hidden_dim=64)`

**Architecture:**
- **Encoder:** 3-layer MLP (`input_dim → 64 → 64 → 10`). Maps a sliding window of KE values to a 10-dimensional latent Koopman space.
- **Koopman Generator:** Small MLP (`1 → 32 → 100`). Takes the scalar active force $f$ as input and generates a $10 \times 10$ Koopman operator matrix $K(f)$.
- **Decoder:** 3-layer MLP (`10 → 64 → 64 → input_dim`). Maps from latent space back to KE window space.

**Forward pass:**
```
z       = Encoder(x)                   # Encode current window
K       = KoopmanGenerator(f)          # Generate force-specific operator
z_next  = K @ z                        # Advance in latent space (linear evolution!)
x_pred      = Decoder(z)               # Reconstruct current window
x_next_pred = Decoder(z_next)          # Predict next window
```

**Forward pass returns:** `(x_pred, x_next_pred, z, z_next)`

---

### `src/train.py`
**Purpose:** Trains the Koopman autoencoder on all 17 force trajectories.

**Critical Design Decision — Log-Space Training:**

> [!IMPORTANT]
> The kinetic energy decays from $\sim 10^{-1}$ to $\sim 10^{-24}$. If you train in **linear space**, the MSE loss treats a miss at $10^{-5}$ the same as a miss at $10^{-20}$ (both effectively zero). This causes the network to **drop to zero thousands of timesteps early**.
>
> The fix: transform KE to log space before training:
> ```python
> ke_log = np.log10(ke + 1e-12)
> ```
> The model is trained entirely in log space. Predictions are transformed back: `ke_pred = 10**pred_log - 1e-12`.

**Training setup:**
- **Input:** Sliding windows of size 100 timesteps in log-space
- **Target:** The next shifted window (1-step-ahead prediction)
- **Loss:** `MSE(x_pred, x_curr) + MSE(x_next_pred, x_next)` — reconstruction + one-step prediction
- **Optimizer:** Adam, `lr=1e-3`
- **Epochs:** 50
- **Batch size:** 256

**Train/Test Split:**
- **Test forces (held out — never seen during training):** `[0.5, 1.25, 1.6, 1.8]`
- **Training forces:** All remaining 13 forces

**Training results (on RTX 3060):**
- Training samples: 765,856 sliding windows
- Test samples: 235,648 windows
- Final Training Loss (Log-Space MSE): `~6.25e-03`
- Zero-Shot Test Loss (Log-Space MSE): `~1.68e-03`

**Model saved to:** `data/processed/koopman_model.pth`

**How to retrain from scratch:**
```bash
python src/train.py
```

---

### `src/publication_plots.py`
**Purpose:** Generates publication-quality figures.

**Two functions:**

1. `plot_critical_slowing_down()` — Runs HDMD analysis and plots $\tau$ vs $f$ on a log scale with the critical force marked. Saved to `docs/pub_critical_slowing_down.png`.

2. `plot_global_master_trajectory()` — Loads the trained model, generates autoregressive predictions for a subset of forces (`{0.0, 0.5, 1.0, 1.4, 1.6, 1.8, 2.0}`), and overlays them on the ground truth. Uses a `plasma` colormap gradient for force values. Solid lines = ground truth; dashed lines = predictions. Saved to `docs/pub_master_trajectories.png`.

**How to run:**
```bash
python src/publication_plots.py
```

---

### `src/demo.py`
**Purpose:** End-to-end demo for showing the project to a supervisor.

**Runs 4 sequential steps:**
1. Data loading summary (stats per force)
2. HDMD analysis with ASCII bar chart of $\tau$ values in the terminal
3. Zero-shot evaluation on test forces with MSE printed per force
4. Generates both publication figures

**How to run (pre-trained model, ~5–8 min):**
```bash
python src/demo.py
```

**How to run (retrain from scratch, ~30 min on RTX 3060):**
```bash
python src/demo.py --retrain
```

---

## 5. Output Figures

| File | What It Shows |
|---|---|
| `docs/pub_critical_slowing_down.png` | $\tau$ vs $f$ on log scale. Clear divergence as $f \to 1.65$. This is the physics result. |
| `docs/pub_master_trajectories.png` | All KE trajectories (ground truth + ML prediction). Plasma colormap by force. Log scale on Y-axis. |
| `docs/scaling_exploration.png` | Log-log plot of $\tau$ vs $(f_c - f)$ with power-law fit line showing $\gamma \approx 1.64$. |
| `docs/relaxation_time_scaling.png` | Same as `pub_critical_slowing_down.png` but with plain blue dots, generated by `hdmd_analysis.py` directly. |

---

## 6. Conference Abstract (CompFlu 2026)

**Title:** *Decoding Dense Active Fluids Through Machine Learning*

The abstract (~195 words) along with keywords and references is saved at:
`C:\Users\rithv\.gemini\antigravity\brain\b081edfe-5d94-4dfd-807b-4eda5dea7b8f\compflu_abstract_draft.md`

**Keywords:** Active matter, Glass transition, Jamming, Dense active fluids, Machine learning, Kob-Andersen mixture, Lennard-Jones potential, Non-equilibrium dynamics, Kinetic arrest, Data-driven methods

---

## 7. Current Status & Known Issues

### What Works
- [x] Full data pipeline (loader, HDMD, training, plotting)
- [x] HDMD correctly extracts $\tau$ for all jamming-regime forces
- [x] Log-space training resolves the early-drop artifact (previously the model snapped to zero too early)
- [x] Zero-shot predictions for forces the model never saw during training
- [x] Master trajectory plot covering both jammed and fluid regimes
- [x] Scaling exponent $\gamma$ extracted and quantified

### Known Limitations / Open Issues
1. **Non-monotonic $\tau$:** The HDMD $\tau$ values are not perfectly monotonic with force (e.g., $\tau$ at $f=1.45$ dips below $f=1.4$). This is likely due to HDMD sensitivity to trajectory length and noise at intermediate forces. More simulation statistics per force would help.
2. **$R^2 = 0.70$:** The power-law fit has moderate goodness-of-fit. The low-force regime ($f = 0.0, 0.25, 0.5$) deviates from the power law because true critical scaling only applies asymptotically near $f_c$. One approach: restrict the fit to $f > 1.0$.
3. **HDMD import path:** Scripts in `src/` use bare imports (`from data_loader import ...`). Always run from the project root or prepend `sys.path.append('src')`.

---

### Update (Oct 2026): what the new analysis found
- **Teacher forcing:** `publication_plots.plot_global_master_trajectory()` feeds the true previous 100 values at every step, so `pub_master_trajectories.png` is not a forecast. Run freely, the original model is off by 0.6–8.7 decades (`docs/forecast_original_model.png`).
- **Master curve exists:** log₁₀K = a(f) + G(t/τ(f)) for 1.0 ≤ f ≤ 1.8 (spread 1.81 → 0.030), with a force-independent final collapse rate of 0.40 decades/time.
- **Governing equation:** τ(f)·dy/dt = g(y − a(f)) with g a cubic (weak SINDy), validated leave-one-force-out.
- **HAVOK:** intermittent forcing in every arresting force, Gaussian for f ≥ 1.8; it switches on in the last ~20% of the slow stage.
- Items A–C of §8 below are done (see `docs/RESULTS.md`).

## 8. Next Steps (originally planned; A–C now done)

### A. Equation Discovery (SINDy-style)
The idea is to extract the governing differential equation $\frac{dK}{dt} = \mathcal{F}(K, f)$ directly from data using sparse regression (Lasso) and SymPy for symbolic output. The hypothesis is that an activity-induced cage-breaking term ($+B \cdot f \cdot K^n$) will appear alongside the standard dissipation term ($-A \cdot K$), which would physically justify the anomalous $\gamma$.

**Plan:**
1. Numerically compute $\dot{K}(t)$ via `np.gradient(ke, time)` on raw trajectories
2. Build a candidate feature library: $[K, K^2, K^3, f, f \cdot K, f \cdot K^2, \ldots]$
3. Run `sklearn` Lasso regression to find sparse coefficients
4. Use `sympy` to assemble the symbolic differential equation

### B. Restrict Scaling Fit to Near-Critical Region
Re-run the $\gamma$ regression using only forces $f > 1.0$ (where the power law is physically expected to hold) and compare the improved $R^2$ and updated $\gamma$.

### C. Vary $f_c$
The value $f_c = 1.65$ was chosen by visual inspection. A more rigorous approach would be to treat $f_c$ as a free parameter in the regression and optimize over it.

---

## 9. Environment & Dependencies

- **Python:** 3.x
- **GPU:** NVIDIA RTX 3060 (all scripts auto-detect and fall back to CPU)
- **Key packages:** `torch`, `numpy`, `scipy`, `matplotlib`, `sklearn`
- **Running scripts:** Always run from the project root directory:
  ```bash
  cd "C:\Users\rithv\OneDrive\Desktop\ML MODEL FOR ACTIVE GLASSY FLUIDS"
  python src/<script>.py
  ```
