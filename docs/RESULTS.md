# Results: scaling, Koopman prediction, HAVOK and governing equations

System: 2D Kob–Andersen active glass, N = 1000 particles, **one simulation per active force**
f ∈ {0, 0.25, …, 2.0}, all apparently started from the same initial configuration (identical K(0)).
Observable: total kinetic energy K(t), analysed as y = log₁₀ K (columns 1–2 of
`data/raw/avg_ke_1000_{f}`; column 3 is ignored).

All numbers below are reproduced by the scripts named in each section (run from the project root).

**Scope of the claims.** Everything here is a *data-driven, phenomenological* description of one
scalar observable in one model system (one density, one system size, one run per force).
"Force-independent" means "the same for all forces in this dataset", not universal in the
statistical-physics sense. None of the equations below are derived from the particle dynamics. See §6.

---

## 0. Corrections to the earlier picture

| Earlier claim (handoff) | What the data show |
|---|---|
| f_c ≈ 1.65 | **f = 1.65 still arrests** (at t ≈ 23 600). f = 1.8 has not arrested by the end of the run (t = 29 505) but is still in the same slow exponential decay (τ_K ≈ 19 000; extrapolated arrest at t ~ 1.5·10⁵). Only f = 2.0 is (nearly) stationary. ⇒ **f_c > 1.8** is the most consistent reading, best estimates 1.8–1.95. |
| γ ≈ 1.64 ± 0.31 (HDMD) | γ depends on how τ is defined and is strongly correlated with f_c. HDMD on *linear* K is dominated by the first few hundred time units. Range from the more robust definitions: **γ ≈ 1.4–2.1**, see §1. |
| Zero-shot Koopman predictions match the data | The old master-trajectory plot is **teacher-forced**: every prediction step is given the true previous 100 values. Run freely, the same model is off by 3.5–11.7 decades (§4). |

Data notes: dt = 0.5 except the last row (dt = 0.05, dropped). After arrest K sits on a numerical
floor ~10⁻²⁶; everything below 10⁻²⁰ is cut.

---

## 1. Q1: Is there scaling? Does a master curve exist?

**Short answer: yes, within this dataset, for 1.0 ≤ f ≤ 1.8.**

Script: `src/scaling_analysis.py` → `docs/master_curve.png`, `docs/tau_scaling_robust.png`, `docs/fc_profile.png`

**Anatomy of a trajectory (f ≥ 1.0).** (i) A short transient (t ≲ 20); (ii) a long *slow stage*
in which log K falls almost linearly, i.e. K decays almost exponentially, by ~3.5 decades;
(iii) an abrupt *collapse* to the floor once K ≈ 10⁻⁵·⁵–10⁻⁶. Forces f ≤ 0.75 behave like a
quench (fast decay with oscillations) and are a different regime, not covered here.

**Master curve.** The 12 trajectories with f = 1.0 … 1.8 collapse onto

  **log₁₀ K(t, f) = a(f) + G(t / τ(f))**

| rescaling | spread of log₁₀ K across forces (mean variance) |
|---|---|
| none | 1.81 |
| time only, t/τ(f) | 0.11 |
| time + amplitude, a(f) + G(t/τ) | **0.030** |

*Caveat:* each curve gets two free numbers (τ, a), i.e. 22 fitted parameters for 12 curves, and the
curves are close to straight lines in log K. Such a family collapses easily under a stretch and a
shift. The non-trivial content is therefore narrower than "the curves collapse":
(1) the arrest (cliff) happens at the **same rescaled time and level** for every force;
(2) f = 1.8, which has not arrested, still falls on the same G;
(3) the fitted τ(f) **diverges** (below);
(4) **the final collapse rate is the same for all forces**: log K falls at 0.40 ± 0.09 decades per
time unit after the cliff (K ∝ e^(−0.91 t)).

**Critical scaling of τ.** τ grows faster than exponentially: an Arrhenius law log τ = a + b f is
rejected by ΔAIC = 11–26 for every robust τ definition. Power-law fits τ = A (f_c − f)^(−γ), with
f_c constrained above every force used in the fit:

| τ definition | best f_c | best γ | acceptable range (ΔAIC < 2) |
|---|---|---|---|
| arrest time t_arr (f ≤ 1.65) | 1.81 | 1.44 | f_c 1.74–1.94, γ 1.15–1.97 |
| time to reach K = 10⁻⁵ | 1.86 | 1.99 | f_c 1.78–2.00, γ 1.60–2.69 |
| slow-decay time τ_K (incl. f = 1.8) | 1.89 | 1.89 | f_c 1.86–1.93, γ 1.66–2.18 |
| master-curve τ (incl. f = 1.8) | 1.94 | 2.10 | f_c 1.89–2.00, γ 1.81–2.49 |

Times to reach a *fixed* level (10⁻³, 10⁻⁴) and HDMD on linear K give γ = 3–11 with f_c pinned at
the edge. They are biased because the plateau height a(f) itself rises with f. A power law and a
VFT form (log τ ∝ 1/(f_c − f)) fit equally well, so the data **cannot distinguish a true power-law
divergence from a VFT-like one**, and f_c and γ are strongly degenerate (`docs/fc_profile.png`).

**Answer to Q1:** scaling and a master curve exist for 1.0 ≤ f ≤ 1.8 in this dataset, with a
diverging time scale and a force-independent final collapse. Current estimate:
**f_c ≈ 1.8–1.95, γ ≈ 1.4–2.1**, not separately determined. With one run per force these are
single-realisation estimates (run-to-run scatter of t_arr is plausibly ±25 %, §3).

---

## 2. HAVOK: intermittent bursts before arrest

Script: `src/havok.py` → `docs/havok_summary.png`, `docs/havok_f*.png`, `docs/havok_lorenz_check.png`

**What HAVOK is.** HAVOK (Hankel Alternative View Of Koopman; Brunton et al., *Nat. Commun.* 8, 19,
2017) takes one measured signal, stacks time-delayed copies of it (a Hankel matrix), and uses an
SVD to find a few coordinates v₁…v_r. It fits a *linear* model to v₁…v_{r−1} and keeps the last
coordinate v_r as an external **forcing**. When the forcing is quiet, the dynamics are simple
(linear). When it bursts, something the linear model cannot explain is happening. Our own
implementation was validated (i) on the Lorenz system: the forcing precedes lobe switches at 2.6×
chance, with 3.9 % reconstruction error; and (ii) against PyDMD's HAVOK, which gives the same
least-damped eigenvalue at f = 1.4 (0.0001 + 0.189i vs 0.00004 + 0.188i).

Applied to the fluctuations of log K around the slow trend:

* **Intermittency separates the regimes.** The forcing is strongly non-Gaussian (excess kurtosis
  3.5–32) for every arresting force (1.0–1.65) and **Gaussian for f = 1.8 and 2.0**
  (kurtosis 0.6, −0.1).
* **It switches on late.** The forcing is essentially never active in the first 60 % of the slow
  stage and appears in the last ~20 % (t/τ ≳ 0.8, K ~ 10⁻⁵), for every arresting force.
* **The forcing precedes sudden drops of K** ("dips" that recover): 17–68 % of dips are preceded
  within 20 t.u. by active forcing, versus 8–27 % at random times (2–4× chance, measured causally).
* **Before the final collapse:** forcing is active 14–68 % of the time in the window 120–20 t.u.
  before arrest, versus 3–5 % on average. Part of this is simply that fluctuations grow as K
  falls. Against the *preceding* 500 t.u., a specific pre-arrest burst is clear only for f ≤ 1.45
  (2–7×) and fades towards f_c (1.1–1.5× at 1.5–1.65).
* The linear part of the HAVOK model reconstructs the fluctuations only over short horizons and is
  marginally stable for most forces. HAVOK is used here as a *diagnostic*, not as a forecaster.

---

## 3. Governing differential equations

Script: `src/equation_discovery.py` → `docs/equations.md`, `docs/ode_validation.png`, `docs/ode_laws.png`

Method: our own **weak-form SINDy** (multiply by smooth test functions and integrate by parts, so
noisy data are never differentiated), sparse regression with BIC. It passes a synthetic test: it
recovers a known law from data with the same correlated noise as the simulations. PySINDy is used
as a cross-check.

**Result: one reduced equation fits all forces in the dataset (Model B)**

  **τ(f) · dy/dt = g(y − a(f)),  y = log₁₀ K**

  g(u) = 7.10 u + 2.86 u² + 0.307 u³,  τ(f) = 1433 (1.935 − f)^(−2.10),  a(f) = 1.36 f − 1.47

followed by dy/dt = −0.40 (decades/time) once u = y − a(f) < −6.9.

**What this equation is and is not.** Mathematically it is the *differential form of the master
curve*: any family that collapses as y = a(f) + G(t/τ(f)) obeys an equation of this type, with g
determined by G. Its value is that it is compact (≈ 8 constants), interpretable, and predicts forces
left out of the fit. It is an empirical reduced model for one global observable. It is **not** a
first-principles equation of motion, and it says nothing about spatial structure.

* **Validation, leave-one-force-out** (re-derived without the force it predicts): log K within
  0.2–0.6 decades and arrest time within 5–30 % for f = 1.05–1.65 (f = 1.0: 1.0 decade, +52 %).
* |g| has a minimum (≈ 2) near u ≈ −4.6, a bottleneck where the decay is slowest. It sits at the
  same stage where HAVOK finds intermittency switching on. This is suggestive, not established.
* **Physical reading (interpretation):** the slow-stage decay rate 1/τ_K(f) = 1/215 … 1/19 000 is
  200–20 000× smaller than the post-arrest decay rate 0.91. In an energy balance
  dK/dt = −ΓK + Π_active, active power injection would cancel all but a fraction 1/(Γτ_K) of
  dissipation in the slow stage and vanish at arrest.
* **Generic library fails:** a polynomial library in (y, f) (Model A, and PySINDy) gives dense,
  mutually inconsistent equations that blow up for f ≥ 1.55, because a polynomial in f cannot
  represent a diverging rate. The scaling structure from Q1 is what makes a stable equation
  identifiable.
* **Hypothesis dK/dt = −AK + B f Kⁿ (handoff): not supported.** n is not identifiable (residual
  changes < 15 % over n = 0–3), the best fit has B < 0 (unphysical), and arrest times are off by
  50–200 %.
* **Stochastic version (Langevin):** du = g(u) ds + √(2D₂(u)) dW, with s = t/τ. The diffusion D₂
  grows ~1000× as K approaches arrest. The model reproduces the median arrest time (s* = 1.21 vs
  1.17 observed) but overestimates its spread (16–84 %: 0.66–2.11 vs 0.92–1.47).

---

## 4. Q2: Can Koopman methods predict unknown KE time series?

Script: `src/koopman_forecast.py` → `docs/forecast_lofo.png`, `docs/forecast_prefix.png`,
`docs/forecast_original_model.png`, `docs/forecast_table.md`

All predictions are **free-running**: nothing from the test trajectory is fed in after the start.
13 forces (1.0–2.0), each left out in turn; every model and every fitted quantity (master curve,
τ(f), a(f), ODE, neural network) is rebuilt without that force.

* **(a) Unseen force:** predict the whole trajectory from f alone. The first 100 t.u. are
  interpolated from neighbouring training forces, never taken from the test run.
* **(b) Prefix:** given the first 10 / 25 / 50 % (of the arrest time) of the unseen trajectory,
  forecast the rest.

Models: **HDMD-param** (Hankel DMD, linear Koopman operator interpolated in f) / **HDMD-prefix**
(fitted on the prefix only; equivalent to HAVOK with unknown future forcing set to zero);
**DeepKoopman** (force-conditioned Koopman autoencoder retrained with a 40-step rollout loss);
**MasterCurve** (a(f) + G(t/τ(f)): a slow Koopman mode whose eigenvalue −1/τ(f) follows the critical
law, switched to the collapse mode at arrest); **ODE-B** (§3).

A non-Koopman **neighbour baseline** is included: copy the nearest training trajectory and stretch its
time axis to an interpolated arrest time.

**(a) Unseen force** (medians over the 11 arresting forces, 95 % bootstrap CI; slow-stage errors in
decades of log₁₀K; a missed arrest counts as 100 %)

| model | arrest-time error | within 25 % | missed arrests | false arrests (1.8, 2.0) | RMSE | MAE | R² |
|---|---|---|---|---|---|---|---|
| Neighbour baseline | 8 % [3, 24] | 10/11 | 0 | 1/2 | 0.36 | 0.16 | 0.89 |
| HDMD-param | 67 % [59, 73] | 0/11 | 1 | 0/2 | 4.53 | 4.20 | < 0 |
| DeepKoopman | 100 % | 0/11 | 11 | 0/2 | 3.86 | 3.58 | < 0 |
| **MasterCurve** | **14 % [10, 18]** | 10/11 | 0 | 0/2 | **0.31** | 0.17 | 0.87 |
| ODE-B | 17 % [9, 31] | 7/11 | 0 | 0/2 | 0.45 | 0.26 | 0.80 |

**(b) Forecast from a prefix** (median arrest-time error, arresting forces; missed arrests in brackets)

| model | 10 % given | 25 % given | 50 % given |
|---|---|---|---|
| HDMD-prefix (≈ HAVOK) | 100 % (6) | 100 % (6) | 100 % (3) |
| DeepKoopman | 100 % (11) | 100 % (10) | 41 % (4) |
| **MasterCurve** | **20 % (0)** | **15 % (0)** | **13 % (0)** |
| ODE-B | 60 % (1) | 53 % (1) | 31 % (1) |

**One-step-ahead versus free-running evaluation** of the one-step-trained autoencoder
(`koopman_model.pth`, first 10 000 t.u., RMSE of log₁₀(K + 10⁻¹²)):

| f | one-step-ahead | free-running |
|---|---|---|
| 0.5 | 0.14 | 8.77 |
| 1.25 | 0.08 | 3.50 |
| 1.6 | 0.02 | 5.74 |
| 1.8 | 0.02 | 11.75 |

**Answer to Q2.**
* *Plain* data-driven Koopman models (linear HDMD/HAVOK operators, a deep Koopman autoencoder)
  **cannot** predict an unknown KE trajectory over the needed horizon (thousands of steps). A
  linear operator cannot produce the abrupt arrest, and the neural model drifts in free rollout.
  Accuracy in one-step-ahead evaluation does not carry over to free-running prediction.
* A **Koopman model built on the scaling structure** (one slow mode with eigenvalue −1/τ(f), τ(f)
  from the critical law, plus the collapse mode) **does** predict unseen forces: slow-stage error
  ≈ 0.3 decades, arrest time within 14 % (median), no missed or false arrests. From a 10 % prefix the
  arrest-time error is ~20 %.
* The time-stretched neighbour baseline is **at least as accurate for interior forces** (8 %). It is
  itself a one-parameter collapse, so this supports the common-shape result. The critical law helps
  only under extrapolation: f = 1.0 (10 % vs 47 %) and f = 1.8 (the baseline predicts a false arrest).
* These are single-run comparisons, so ~±25 % run-to-run scatter of the true arrest time is not
  accounted for.

Full report with validation of every claim: `reports/Technical_Report.pdf`.

---

## 5. Limitations

* **One simulation per force.** τ, t_arr, f_c and γ carry run-to-run scatter that we cannot
  measure; the Langevin model suggests ~±25 % in t_arr.
* **One system:** one density, one N (= 1000), one persistence time, apparently one initial
  configuration. Nothing here tests universality across systems.
* f_c and γ are degenerate; power law vs VFT is undecided; f = 1.8 is assumed to be arresting
  because it is still decaying.
* All validation is within the same dataset (leave-one-force-out). The real test is predicting new
  simulations in advance.
* Only f = 2.0 is in the flowing state; nothing can be said about the fluid side.
* The regime f ≤ 0.75 is not covered.

## 6. Relation to existing theory, and what would strengthen this

Active matter and glasses do have theoretical frameworks: hydrodynamic theories (Toner–Tu for
polar active fluids, Active Model B/B+ for motility-induced phase separation, active nematics);
mode-coupling theory (MCT) and random first-order transition (RFOT) theory for glasses; and their
extensions to active glasses (e.g. active MCT by Szamel, Berthier and co-workers; RFOT for active
glasses, Nandi et al., PNAS 2018). There is no generally accepted first-principles theory of dense
active glassy dynamics. The reduced equation found here does not fill that gap: it is an empirical
model for one global observable. Its use is as a quantitative target that such theories should
reproduce (a diverging τ(f), a force-independent master curve, intermittency before arrest).

Most useful next steps:
1. **Several runs (seeds/initial configurations) per force**, so every number gets an error bar.
2. **Blind prediction:** use the fitted law to predict t_arr for f = 1.7, 1.75, 1.85, 1.9 *before*
   running them, then run them longer than 10⁵.
3. **A second system size** (e.g. N = 4000) and a second density/persistence time, to test whether
   G and the collapse rate are really force- and system-independent.
4. Compare γ and the shape of G with **active MCT/RFOT predictions**.
