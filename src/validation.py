"""
Validation and robustness tests for every claim in docs/RESULTS.md.

  V1  Is f = 1.8 (and 2.0) still decaying?  OLS slope of log10 K over t >= 5000 with a
      moving-block bootstrap (blocks >> correlation time) -> CI and one-sided p-value.
  V2  Is arrest at a fixed rescaled time?  tau_K is measured from the slow stage only
      (never using the cliff); test t_arr = c * tau_K^beta (beta = 1 expected), CV of
      t_arr / tau_K and of the number of decades decayed before arrest.
  V3  Is the HAVOK intermittency real?  Compare forcing kurtosis of the data with 19
      phase-randomised surrogates (same power spectrum, Gaussian, no bursts) -> rank p-value;
      block-bootstrap CI of the data's forcing kurtosis.
  V4  HAVOK sensitivity to the delay window q (20, 40, 80 samples).
  V5  Equation (Model B): bootstrap over forces -> CIs of the g(u) coefficients;
      polynomial degree 2/3/4 -> leave-one-force-out arrest-time error.
  V6  Weak-SINDy recovery error versus noise amplitude (0.5x .. 4x the data's noise).
  V7  Critical fit (tau_K) robustness to the slow-stage window.

Run from the project root:  python src/validation.py
"""
import os
import json
import numpy as np
from scipy.stats import kurtosis
from data_loader import load_clean
import scaling_analysis as SA
import havok as HV
import equation_discovery as EQ

ROOT = os.path.join(os.path.dirname(__file__), "..")
PROC = os.path.join(ROOT, "data", "processed")
RNG = np.random.default_rng(42)


def block_bootstrap_slope(t, y, block=500, n_boot=2000):
    """OLS slope with a moving-block bootstrap of residuals (keeps autocorrelation)."""
    c = np.polyfit(t, y, 1)
    res = y - np.polyval(c, t)
    n = len(y)
    nb = int(np.ceil(n / block))
    slopes = []
    for _ in range(n_boot):
        starts = RNG.integers(0, n - block, nb)
        r = np.concatenate([res[s:s + block] for s in starts])[:n]
        slopes.append(np.polyfit(t, np.polyval(c, t) + r, 1)[0])
    slopes = np.array(slopes)
    # p-value for H0: slope >= 0 (no decay), using the bootstrap distribution centred at 0
    p = float(np.mean(slopes - c[0] <= -abs(c[0]))) if c[0] < 0 else 1.0
    return dict(slope=float(c[0]), ci95=np.percentile(slopes, [2.5, 97.5]).tolist(), p_one_sided=p,
                total_drop_decades=float(c[0] * (t[-1] - t[0])))


def v1(D):
    out = {}
    for f in (1.8, 2.0):
        d = D[f]
        m = d['t'] >= 5000
        out[f] = block_bootstrap_slope(d['t'][m], d['y'][m])
    return out


def v2(D):
    arrested = [f for f in sorted(D) if f >= 1.0 and np.isfinite(D[f]['t_arr'])]
    tk = np.array([SA.tau_slope(D[f]) for f in arrested])
    ta = np.array([D[f]['t_arr'] for f in arrested])
    X = np.log(tk)
    Y = np.log(ta)
    beta, lc = np.polyfit(X, Y, 1)
    r2 = 1 - np.sum((Y - np.polyval([beta, lc], X)) ** 2) / np.sum((Y - Y.mean()) ** 2)
    bs = []
    for _ in range(4000):
        i = RNG.integers(0, len(X), len(X))
        if len(np.unique(i)) > 3:
            bs.append(np.polyfit(X[i], Y[i], 1)[0])
    ratio = ta / tk
    drop = np.array([np.interp(SA.T_TRANSIENT, D[f]['t'], D[f]['y_s']) - EQ.onset_level(D[f]) for f in arrested])
    return dict(forces=arrested, tau_K=tk.tolist(), t_arr=ta.tolist(), beta=float(beta),
                beta_ci95=np.percentile(bs, [2.5, 97.5]).tolist(), r2=float(r2),
                ratio_mean=float(ratio.mean()), ratio_cv=float(ratio.std() / ratio.mean()),
                tau_K_range=[float(tk.min()), float(tk.max())],
                decades_before_arrest=drop.tolist(), decades_mean=float(drop.mean()),
                decades_cv=float(drop.std() / drop.mean()))


def ft_surrogate(x):
    """Phase-randomised surrogate: same power spectrum, random phases (Gaussian, linear)."""
    X = np.fft.rfft(x - x.mean())
    ph = RNG.uniform(0, 2 * np.pi, len(X))
    ph[0] = 0
    if len(x) % 2 == 0:
        ph[-1] = 0
    return np.fft.irfft(np.abs(X) * np.exp(1j * ph), n=len(x)) + x.mean()


def forcing_kurtosis(dy, q, r):
    m = HV.HAVOK(q, r).fit(dy, 0.5)
    return float(kurtosis(m.forcing)), m.forcing


def v3(D, n_sur=19):
    out = {}
    for f in [g for g in sorted(D) if g >= 1.0]:
        t, dy, _, _ = HV.glass_segment(D[f])
        r = HV.select_rank(dy, 0.5, HV.Q, HV.RANKS, min(2000, len(dy) - HV.Q))[0]
        k_data, fr = forcing_kurtosis(dy, HV.Q, r)
        k_sur = [forcing_kurtosis(ft_surrogate(dy), HV.Q, r)[0] for _ in range(n_sur)]
        # block bootstrap CI of the data's forcing kurtosis
        n, b = len(fr), 400
        kb = []
        for _ in range(500):
            st = RNG.integers(0, max(1, n - b), int(np.ceil(n / b)))
            kb.append(kurtosis(np.concatenate([fr[s:s + b] for s in st])[:n]))
        rank_p = (1 + np.sum(np.array(k_sur) >= k_data)) / (n_sur + 1)
        out[f] = dict(r=int(r), kurtosis_data=k_data, kurtosis_data_ci95=np.percentile(kb, [2.5, 97.5]).tolist(),
                      kurtosis_surrogates=[float(np.min(k_sur)), float(np.median(k_sur)), float(np.max(k_sur))],
                      p_rank=float(rank_p))
        print(f"   V3 f={f}: forcing kurtosis {k_data:6.2f} [{out[f]['kurtosis_data_ci95'][0]:.2f}, "
              f"{out[f]['kurtosis_data_ci95'][1]:.2f}]  surrogates median {np.median(k_sur):5.2f} "
              f"(max {np.max(k_sur):.2f})  p = {rank_p:.2f}")
    return out


def v4(D):
    out = {}
    q0 = HV.Q
    for q in (20, 40, 80):
        HV.Q = q
        rows = [HV.analyse_force(D[f], f)[0] for f in sorted(D) if f >= 1.0]
        jam = [r for r in rows if r['f'] <= 1.65]
        out[q] = dict(
            kurtosis_jammed_median=float(np.median([r['forcing_excess_kurtosis'] for r in jam])),
            kurtosis_jammed_min=float(np.min([r['forcing_excess_kurtosis'] for r in jam])),
            kurtosis_1p8=float([r for r in rows if r['f'] == 1.8][0]['forcing_excess_kurtosis']),
            kurtosis_2p0=float([r for r in rows if r['f'] == 2.0][0]['forcing_excess_kurtosis']),
            dip_hit_over_chance_median=float(np.median([r['dip_precursor_hit'] / r['dip_precursor_chance']
                                                        for r in jam if r['dip_precursor_chance'] > 0])),
            prearrest_over_global_median=float(np.median([r['cliff_enrichment'] for r in jam])),
            prearrest_over_local_median=float(np.median([r['cliff_forcing_active_frac'] /
                                                         max(r['cliff_local_baseline_frac'], 1e-3) for r in jam])))
        print(f"   V4 q={q}: {out[q]}")
    HV.Q = q0
    return out


def v5(D, S, n_boot=200):
    forces = [f for f in sorted(D) if f >= 1.0 and np.isfinite(D[f]['t_arr'])] + [1.8]
    cl = EQ.closure(S, forces)
    coefs = []
    for _ in range(n_boot):
        pick = list(RNG.choice(forces, len(forces), replace=True))
        coefs.append(EQ.fit_model_B(D, pick, cl['a_of'], cl['tau_of']))
    coefs = np.array(coefs)
    boot = dict(names=EQ.B_NAMES, median=np.median(coefs, 0).tolist(),
                ci95_low=np.percentile(coefs, 2.5, 0).tolist(), ci95_high=np.percentile(coefs, 97.5, 0).tolist(),
                frac_selected=np.mean(coefs != 0, 0).tolist())
    uu = np.linspace(-6.5, -2.5, 41)
    gs = np.array([EQ.theta_B(uu) @ c for c in coefs])
    boot['g_band'] = dict(u=uu.tolist(), lo=np.percentile(gs, 2.5, 0).tolist(),
                          med=np.median(gs, 0).tolist(), hi=np.percentile(gs, 97.5, 0).tolist())
    # degree sensitivity, leave-one-force-out
    arrested = [f for f in forces if np.isfinite(D[f]['t_arr'])]
    gam = -S['cliff_rate']['mean']
    u_on = float(np.median([EQ.onset_level(D[f]) - cl['a_of'](f) for f in arrested]))
    deg = {}
    d0, n0 = EQ.B_DEG, EQ.B_NAMES
    for dg in (2, 3, 4):
        EQ.B_DEG = dg
        EQ.B_NAMES = ["1", "u", "u^2", "u^3", "u^4"][:dg + 1]
        errs = []
        for f in arrested:
            others = [g for g in forces if g != f]
            cl_l = EQ.closure(S, others)
            xi = EQ.fit_model_B(D, others, cl_l['a_of'], cl_l['tau_of'])
            rhs = lambda t, y, xi=xi, cl_l=cl_l, f=f: EQ.theta_B(np.atleast_1d(y) - cl_l['a_of'](f)) @ xi / cl_l['tau_of'](f)
            y0 = float(np.interp(EQ.T0, D[f]['t'], D[f]['y_s']))
            with np.errstate(all='ignore'):
                tg, yg, ta = EQ.integrate(rhs, y0, D[f]['t'][-1], cl_l['a_of'](f) + u_on, gam)
            errs.append(EQ.score(D[f], tg, yg, ta)[1])
        e = np.abs(np.nan_to_num(np.array(errs), nan=1.0))
        deg[dg] = dict(median_abs_t_arr_err=float(np.median(np.minimum(e, 1.0))),
                       n_failed=int(np.sum(~np.isfinite(np.array(errs, float)))))
        print(f"   V5 degree {dg}: LOFO median |dt_arr| = {deg[dg]['median_abs_t_arr_err']:.3f}")
    EQ.B_DEG, EQ.B_NAMES = d0, n0
    return dict(bootstrap=boot, degree=deg)


def v6(D):
    forces = [f for f in sorted(D) if f >= 1.0 and np.isfinite(D[f]['t_arr'])] + [1.8]
    true = {"1": -0.0005, "F": 0.0011}
    out = {}
    for amp in (0.05, 0.1, 0.2, 0.4):
        errs, ok = [], 0
        for seed in range(5):
            rng = np.random.default_rng(seed)
            Ds = {}
            for f in forces:
                t, y = EQ.slow_stage(D[f])
                clean = y[0] + (true["1"] + true["F"] * (f - EQ.F_C)) * (t - t[0])
                e = np.zeros(len(t))
                phi = np.exp(-0.5 / 5.0)
                for i in range(1, len(t)):
                    e[i] = phi * e[i - 1] + np.sqrt(1 - phi ** 2) * amp * rng.standard_normal()
                Ds[f] = dict(t=np.r_[0.0, t], y=np.r_[clean[0], clean + e], t_arr=t[-1] + 40.0)
            b, G, w = EQ.build_weak(Ds, forces, EQ.theta_A)
            xi, _, _ = EQ.select_lambda(G, b, w, np.logspace(-4, -0.5, 30))
            rec = dict(zip(EQ.A_NAMES, xi))
            right = set(k for k, v in rec.items() if v != 0) == {"1", "F"}
            ok += right
            errs.append(max(abs(rec["1"] - true["1"]) / abs(true["1"]), abs(rec["F"] - true["F"]) / abs(true["F"])))
        out[amp] = dict(correct_terms_fraction=ok / 5, median_max_rel_coef_err=float(np.median(errs)))
        print(f"   V6 noise {amp:.2f} decades: correct terms {ok}/5, coefficient error {np.median(errs):.3f}")
    return out


def v7(D):
    out = {}
    forces = [f for f in sorted(D) if 1.0 <= f <= 1.8]
    for start in (20.0, 100.0):
        for endfrac in (0.7, 0.85, 0.95):
            tk = []
            for f in forces:
                d = D[f]
                j0 = np.searchsorted(d['t'], start)
                j1 = int(endfrac * d['i_arr']) if np.isfinite(d['t_arr']) else len(d['t'])
                s = np.polyfit(d['t'][j0:j1], d['y'][j0:j1], 1)[0]
                tk.append(-1 / (s * np.log(10)))
            r = SA.fit_powerlaw(np.array(forces), np.array(tk))
            out[f"{start:.0f}-{endfrac}"] = dict(gamma=float(r['gamma']), fc=float(r['fc']), r2=float(r['r2']))
            print(f"   V7 window start {start:.0f}, end {endfrac} t_arr: gamma {r['gamma']:.2f}, f_c {r['fc']:.3f}, R2 {r['r2']:.3f}")
    return out


def main():
    D = load_clean()
    with open(os.path.join(PROC, "scaling_results.json")) as fh:
        S = json.load(fh)
    R = {}
    print("V1 decay significance"); R['V1'] = v1(D); print("  ", R['V1'])
    print("V2 arrest at fixed rescaled time"); R['V2'] = v2(D)
    print(f"   t_arr ~ tau_K^beta: beta = {R['V2']['beta']:.3f} {R['V2']['beta_ci95']}, R2 = {R['V2']['r2']:.3f}; "
          f"t_arr/tau_K = {R['V2']['ratio_mean']:.2f} (CV {R['V2']['ratio_cv']:.2f}) while tau_K spans "
          f"{R['V2']['tau_K_range']}; decades before arrest {R['V2']['decades_mean']:.2f} (CV {R['V2']['decades_cv']:.2f})")
    print("V3 HAVOK surrogate test"); R['V3'] = v3(D)
    print("V4 HAVOK delay-window sensitivity"); R['V4'] = v4(D)
    print("V5 equation bootstrap + degree"); R['V5'] = v5(D, S)
    b = R['V5']['bootstrap']
    for n, m, lo, hi, fs in zip(b['names'], b['median'], b['ci95_low'], b['ci95_high'], b['frac_selected']):
        print(f"   g coefficient {n}: {m:+.3f} [{lo:+.3f}, {hi:+.3f}] selected in {fs:.0%} of bootstraps")
    print("V6 SINDy noise sweep"); R['V6'] = v6(D)
    print("V7 critical-fit window robustness"); R['V7'] = v7(D)
    with open(os.path.join(PROC, "validation_results.json"), "w") as fh:
        json.dump(R, fh, indent=2, default=lambda o: float(o) if np.isscalar(o) else str(o))
    print("Saved data/processed/validation_results.json")


if __name__ == "__main__":
    main()
