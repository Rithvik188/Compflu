"""
Q1: Is there scaling? Does a master curve exist?

Pipeline
1. Extract several independent relaxation times tau(f) for every arresting force:
     - t_arr      : time at which K drops below the numerical floor (arrest time)
     - t_{-3,-4,-5}: first time the smoothed log10 K crosses -3, -4, -5
     - tau_K      : e-folding time of the slow, quasi-exponential decay of K
                    (slope of log K between t=20 and 0.85 t_arr)
     - tau_HDMD   : HDMD (existing KoopmanHDMD) applied to K in that same slow window
     - tau_mc     : time scale obtained from the master-curve collapse (step 3)
2. Fit the critical law  tau = A (f_c - f)^(-gamma)  with f_c > every force used in
   the fit (f=1.65 arrests; f=1.8 has NOT arrested within t=29505 but is still in the
   slow exponential decay, so f_c > 1.8 when tau_K(1.8) is included). Profile the fit
   over f_c, bootstrap the uncertainties and compare against VFT-like and
   Arrhenius-like forms with AIC.
3. Two-parameter master-curve collapse  log10 K(t, f) = a(f) + G(t / tau(f)),
   with a(f), tau(f) obtained by minimising the spread between rescaled curves.

Run from the project root:  python src/scaling_analysis.py
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from data_loader import load_clean

ROOT = os.path.join(os.path.dirname(__file__), "..")
DOCS = os.path.join(ROOT, "docs")
PROC = os.path.join(ROOT, "data", "processed")

FC_LOWER, FC_UPPER = 1.65, 3.0   # f=1.65 still arrests -> f_c > 1.65
ARREST_MAX = 1.65                 # largest force that arrests inside the simulated window
F_MIN_CRIT = 1.0              # forces with the plateau -> cliff structure
T_TRANSIENT = 20.0            # initial ballistic transient excluded from fits/collapse


# ---------------------------------------------------------------------------
# 1. Relaxation times
# ---------------------------------------------------------------------------
def slow_window(d):
    """Indices of the slow quasi-exponential stage: t in [T_TRANSIENT, 0.85 t_arr]."""
    j0 = np.searchsorted(d['t'], T_TRANSIENT)
    j1 = int(0.85 * d['i_arr'])
    return j0, j1


def tau_slope(d):
    """e-folding time of K in the slow window (linear fit of ln K vs t).
    Also defined for f=1.8, 2.0 (no arrest in the window): fit from t=20 to the end."""
    j0, j1 = slow_window(d) if np.isfinite(d['t_arr']) else (np.searchsorted(d['t'], T_TRANSIENT), len(d['t']))
    slope = np.polyfit(d['t'][j0:j1], d['y'][j0:j1], 1)[0]   # decades / time
    return -1.0 / (slope * np.log(10))


def tau_hdmd(d, delay=100, rank=3):
    """Existing HDMD applied to K (linear) in the slow window -> dominant decay time."""
    from hdmd_core import KoopmanHDMD
    j0, j1 = slow_window(d)
    seg = d['K'][j0:j1] / d['K'][j0]
    model = KoopmanHDMD(delay_window=delay, rank=rank, use_gpu=False)
    eigs = model.fit(seg)
    mags = np.abs(eigs)
    mags = mags[(mags < 1.0) & (mags > 0)]
    return -d['dt'] / np.log(mags.max())


def fc_grid(f, n=600):
    lo = max(FC_LOWER, np.max(f)) + 5e-3
    return np.linspace(lo, FC_UPPER, n)


def threshold_time(d, level):
    i = np.argmax(d['y_s'][:d['i_arr']] < level)
    return d['t'][i] if d['y_s'][i] < level else np.nan


def relaxation_times(D, forces):
    taus = {k: [] for k in ['t_arr', 't_-3', 't_-4', 't_-5', 'tau_K', 'tau_HDMD']}
    for f in forces:
        if not np.isfinite(D[f]['t_arr']):
            for k in taus:
                taus[k].append(np.nan)
            taus['tau_K'][-1] = tau_slope(D[f])
            continue
        d = D[f]
        taus['t_arr'].append(d['t_arr'])
        for lv in (-3, -4, -5):
            taus[f't_{lv}'].append(threshold_time(d, lv))
        taus['tau_K'].append(tau_slope(d))
        taus['tau_HDMD'].append(tau_hdmd(d))
    return {k: np.array(v) for k, v in taus.items()}


# ---------------------------------------------------------------------------
# 2. Critical fits
# ---------------------------------------------------------------------------
def fit_powerlaw_fixed_fc(f, tau, fc):
    """log tau = log A - gamma log(fc - f): linear least squares for fixed fc."""
    X = np.vstack([np.ones_like(f), -np.log(fc - f)]).T
    coef, *_ = np.linalg.lstsq(X, np.log(tau), rcond=None)
    sse = np.sum((np.log(tau) - X @ coef) ** 2)
    return coef[1], np.exp(coef[0]), sse


def profile_fc(f, tau, grid=None):
    if grid is None:
        grid = fc_grid(f)
    out = np.array([fit_powerlaw_fixed_fc(f, tau, fc) for fc in grid])
    return grid, out[:, 0], out[:, 2]


def fit_powerlaw(f, tau):
    grid, gam, sse = profile_fc(f, tau)
    k = np.argmin(sse)
    return dict(fc=grid[k], gamma=gam[k], sse=sse[k],
                A=fit_powerlaw_fixed_fc(f, tau, grid[k])[1],
                r2=1 - sse[k] / np.sum((np.log(tau) - np.log(tau).mean()) ** 2),
                at_bracket_edge=bool(k in (0, len(grid) - 1)))


def fit_vft(f, tau):
    """log tau = a + b / (fc - f)  (VFT-like essential singularity)."""
    best = None
    for fc in fc_grid(f):
        X = np.vstack([np.ones_like(f), 1.0 / (fc - f)]).T
        c, *_ = np.linalg.lstsq(X, np.log(tau), rcond=None)
        sse = np.sum((np.log(tau) - X @ c) ** 2)
        if best is None or sse < best['sse']:
            best = dict(fc=fc, a=c[0], b=c[1], sse=sse)
    return best


def fit_arrhenius(f, tau):
    """log tau = a + b f  (no divergence, 2 params)."""
    c = np.polyfit(f, np.log(tau), 1)
    return dict(a=c[1], b=c[0], sse=np.sum((np.log(tau) - np.polyval(c, f)) ** 2))


def aic(sse, n, k):
    return n * np.log(sse / n) + 2 * k


def bootstrap_powerlaw(f, tau, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    gam, fcs = [], []
    n = len(f)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(f[idx])) < 4:
            continue
        r = fit_powerlaw(f[idx], tau[idx])
        gam.append(r['gamma'])
        fcs.append(r['fc'])
    gam, fcs = np.array(gam), np.array(fcs)
    return dict(gamma_ci=np.percentile(gam, [16, 50, 84]).tolist(),
                fc_ci=np.percentile(fcs, [16, 50, 84]).tolist(),
                frac_fc_at_upper_edge=float(np.mean(fcs >= FC_UPPER - 1e-2)))


# ---------------------------------------------------------------------------
# 3. Master curve collapse
# ---------------------------------------------------------------------------
X_GRID = np.linspace(0.02, 1.0, 200)    # rescaled time x = t / tau  (tau ~ arrest time)


def _rescaled(d, tau):
    """log10 K on X_GRID with x = t / tau; NaN outside the usable (pre-arrest) range."""
    t, y = d['t'], d['y_s']
    i_end = d['i_arr']
    t_end = t[i_end - 1]
    x = X_GRID * tau
    vals = np.interp(x, t[:i_end], y[:i_end])
    vals[(x < T_TRANSIENT) | (x > t_end)] = np.nan
    return vals


def _amplitudes(curves, iref, n_iter=20):
    """Optimal vertical shifts a(f) for fixed time scales (alternating least squares,
    a(ref) = 0)."""
    a = np.zeros(len(curves))
    for _ in range(n_iter):
        G = np.nanmean(curves - a[:, None], axis=0)
        a = np.nanmean(curves - G[None, :], axis=1)
        a -= a[iref]
    return np.nan_to_num(a)


def collapse_cost(log_tau_others, D, forces, ref, mode):
    """Mean variance across forces of the rescaled curves. The reference curve keeps
    tau = t_arr(ref), a = 0. mode=1: only time rescaling, mode=2: time + amplitude."""
    n = len(forces)
    iref = forces.index(ref)
    log_tau = np.empty(n)
    log_tau[iref] = np.log(D[ref]['t_arr'])
    log_tau[[i for i in range(n) if i != iref]] = log_tau_others
    curves = np.array([_rescaled(D[f], np.exp(log_tau[i])) for i, f in enumerate(forces)])
    a = _amplitudes(curves, iref) if mode == 2 else np.zeros(n)
    curves = curves - a[:, None]
    cnt = np.sum(~np.isnan(curves), axis=0)
    ok = cnt >= 3
    var = np.nanvar(curves[:, ok], axis=0)
    # every curve must keep overlapping the grid (otherwise it 'escapes' the cost)
    cover = np.mean(~np.isnan(curves), axis=1)
    penalty = np.sum(np.clip(0.3 - cover, 0, None)) * 10.0
    return float(np.mean(var) + penalty), curves, np.exp(log_tau), a


def initial_tau(D, f, ratio):
    """t_arr if the force arrested, otherwise tau_K * <t_arr/tau_K> (prediction)."""
    return D[f]['t_arr'] if np.isfinite(D[f]['t_arr']) else tau_slope(D[f]) * ratio


def master_curve(D, forces, ref=1.4):
    arrested = [f for f in forces if np.isfinite(D[f]['t_arr'])]
    ratio = np.mean([D[f]['t_arr'] / tau_slope(D[f]) for f in arrested])
    others = [f for f in forces if f != ref]
    p0 = np.log([initial_tau(D, f, ratio) for f in others])
    out = {}
    c0, curves0, _, _ = collapse_cost(np.full(len(others), np.log(D[ref]['t_arr'])), D, forces, ref, 1)
    out['raw'] = dict(cost=c0, curves=curves0)
    for mode in (1, 2):
        res = minimize(lambda p: collapse_cost(p, D, forces, ref, mode)[0], p0, method='Powell',
                       bounds=[(v - 1.0, v + 1.0) for v in p0],
                       options=dict(maxiter=50000, xtol=1e-4, ftol=1e-7))
        c, curves, taus, a = collapse_cost(res.x, D, forces, ref, mode)
        out[f'{mode}param'] = dict(cost=c, curves=curves, tau=taus, a=a)
    out['G'] = np.nanmedian(out['2param']['curves'], axis=0)
    out['x'] = X_GRID
    out['t_arr_over_tauK'] = ratio
    return out


# ---------------------------------------------------------------------------
# Final collapse (cliff) after arrest
# ---------------------------------------------------------------------------
def cliff_profile(d, lo=-18.0, hi=-8.0):
    """Shape of the final drop: log10 K vs (t - t_cliff), and its rate in decades/time.
    t_cliff = last time the smoothed curve is above `hi` before arrest."""
    t, y = d['t'], d['y']
    i_end = d['i_arr']
    above = np.where(d['y_s'][:i_end] > hi)[0]
    j = above[-1] if len(above) else 0
    seg = slice(j, min(i_end + 5, len(t)))
    m = (y[seg] < hi) & (y[seg] > lo)
    rate = np.polyfit(t[seg][m], y[seg][m], 1)[0] if m.sum() > 3 else np.nan
    return t[seg] - t[j], y[seg], rate


def admissible_region(f, tau, d_aic=2.0):
    """f_c values whose power-law fit is within d_aic of the best one -> ranges of f_c, gamma."""
    grid, gam, sse = profile_fc(f, tau)
    n = len(f)
    aic_ = n * np.log(sse / n)
    ok = aic_ <= aic_.min() + d_aic
    return dict(fc_range=[float(grid[ok].min()), float(grid[ok].max())],
                gamma_range=[float(gam[ok].min()), float(gam[ok].max())],
                fc_hits_upper_limit=bool(grid[ok].max() >= FC_UPPER - 1e-2))


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_master(D, forces, mc, cliffs, path):
    cm = plt.cm.plasma
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    ax = ax.ravel()
    for i, f in enumerate(forces):
        d = D[f]
        c = cm(f / 2.1)
        ls = '-' if np.isfinite(d['t_arr']) else ':'
        i_end = d['i_arr']
        ax[0].plot(d['t'][:i_end], d['y_s'][:i_end], ls, color=c, lw=1.2, label=f'{f}')
        ax[1].plot(mc['x'], mc['1param']['curves'][i], ls, color=c, lw=1.2)
        ax[2].plot(mc['x'], mc['2param']['curves'][i], ls, color=c, lw=2.0 if ls == ':' else 1.2)
    ax[2].plot(mc['x'], mc['G'], 'k--', lw=2.5, label='master curve $G(x)$')
    for f, (dt_, y_, rate) in cliffs.items():
        ax[3].plot(dt_, y_, color=cm(f / 2.1), lw=1.2)
    ax[0].set(xlabel='$t$', ylabel=r'$\log_{10}K$ (smoothed)', title=f"raw data (spread {mc['raw']['cost']:.3f})")
    ax[1].set(xlabel=r'$x=t/\tau(f)$', ylabel=r'$\log_{10}K$',
              title=f"1-parameter: $t/\\tau$  (spread {mc['1param']['cost']:.4f})")
    ax[2].set(xlabel=r'$x=t/\tau(f)$', ylabel=r'$\log_{10}K - a(f)$',
              title=f"2-parameter: $a(f)+G(t/\\tau)$  (spread {mc['2param']['cost']:.4f})")
    rates = [r for (_, _, r) in cliffs.values()]
    ax[3].set(xlabel=r'$t - t_{\rm cliff}$', ylabel=r'$\log_{10}K$', xlim=(-5, 80), ylim=(-21, -6),
              title=f"final collapse: rate {np.nanmean(rates):.2f} $\\pm$ {np.nanstd(rates):.2f} decades/time")
    ax[0].legend(ncol=2, fontsize=8, title='$f$ (dotted = not yet arrested)')
    ax[2].legend()
    for a_ in ax:
        a_.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_tau_scaling(fa, taus, fits, path):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    markers = dict(zip(taus.keys(), 'osD^v<>p'))
    for k, tau in taus.items():
        ok = np.isfinite(tau) & np.isin(fa, fits[k]['forces'])
        r = fits[k]['powerlaw']
        l, = ax[0].semilogy(fa[ok], tau[ok], markers[k] + '-', ms=6, label=k)
        x = r['fc'] - fa[ok]
        ax[1].loglog(x, tau[ok], markers[k], color=l.get_color(),
                     label=fr"{k}: $\gamma$={r['gamma']:.2f}, $f_c$={r['fc']:.2f}")
        xx = np.logspace(np.log10(x.min()), np.log10(x.max()), 50)
        ax[1].loglog(xx, r['A'] * xx ** (-r['gamma']), '-', color=l.get_color(), alpha=0.6)
    ax[0].axvline(ARREST_MAX, color='crimson', ls='--', label='largest force that arrests in window')
    ax[0].set(xlabel='$f$', ylabel=r'$\tau$', title='Relaxation times (several definitions)')
    ax[1].set(xlabel='$f_c - f$', ylabel=r'$\tau$',
              title=r'$\tau = A (f_c-f)^{-\gamma}$ (best $f_c$ per definition)')
    ax[0].legend(fontsize=8)
    ax[1].legend(fontsize=7)
    for a_ in ax:
        a_.grid(alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_fc_profile(fa, taus, fits, path):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for k, tau in taus.items():
        ok = np.isfinite(tau) & np.isin(fa, fits[k]['forces'])
        grid, gam, sse = profile_fc(fa[ok], tau[ok])
        n = ok.sum()
        d_aic = n * np.log(sse / n)
        d_aic -= d_aic.min()
        l, = ax[0].plot(grid, d_aic, label=k)
        adm = d_aic <= 2
        ax[1].plot(grid, gam, color=l.get_color(), alpha=0.35)
        ax[1].plot(grid[adm], gam[adm], color=l.get_color(), lw=3, label=k)
    ax[0].axhline(2, color='k', ls=':')
    ax[0].set(xlabel='$f_c$', ylabel=r'$\Delta$AIC', ylim=(-0.5, 20),
              title=r'Fit quality vs assumed $f_c$ (below dotted = acceptable)')
    ax[1].set(xlabel='$f_c$', ylabel=r'best $\gamma$',
              title=r'$\gamma$-$f_c$ degeneracy (thick = acceptable $f_c$)')
    for a_ in ax:
        a_.legend(fontsize=8)
        a_.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _json_default(o):
    if isinstance(o, (np.floating, float)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


# ---------------------------------------------------------------------------
def run():
    D = load_clean()
    arrested = [f for f in sorted(D) if F_MIN_CRIT <= f and np.isfinite(D[f]['t_arr'])]
    slow = [f for f in sorted(D) if not np.isfinite(D[f]['t_arr'])]          # 1.8, 2.0
    forces_all = arrested + slow
    fa = np.array(forces_all)

    # ---- master curve: arrested forces + f=1.8 (still in the slow stage) ----
    forces_mc = arrested + [1.8]
    mc = master_curve(D, forces_mc)
    cliffs = {f: cliff_profile(D[f]) for f in arrested}

    # ---- relaxation times ----
    taus = relaxation_times(D, forces_all)
    taus['tau_mc'] = np.array([mc['2param']['tau'][forces_mc.index(f)] if f in forces_mc else np.nan
                               for f in forces_all])

    fits = {}
    print("\n=== Critical fits: tau = A (f_c - f)^-gamma, f_c > max(f used) ===")
    print("(f=1.8 enters only through tau_K / tau_mc, which do not need arrest)")
    for k, tau in taus.items():
        ok = np.isfinite(tau) & (fa <= 1.8)   # f=2.0 slope ~1e-6 decades/time: unreliable, reported separately
        f_, t_ = fa[ok], tau[ok]
        n = len(f_)
        pl, vf, ar = fit_powerlaw(f_, t_), fit_vft(f_, t_), fit_arrhenius(f_, t_)
        bs = bootstrap_powerlaw(f_, t_, n_boot=500)
        adm = admissible_region(f_, t_)
        windows = {}
        for fmin in (1.0, 1.15, 1.35):
            w = f_ >= fmin
            if w.sum() >= 4:
                r = fit_powerlaw(f_[w], t_[w])
                windows[fmin] = dict(gamma=r['gamma'], fc=r['fc'])
        fits[k] = dict(forces=f_.tolist(), powerlaw=pl, vft=vf, arrhenius=ar, bootstrap=bs,
                       admissible=adm, window_sensitivity=windows,
                       aic=dict(powerlaw=aic(pl['sse'], n, 3), vft=aic(vf['sse'], n, 3),
                                arrhenius=aic(ar['sse'], n, 2)))
        A = fits[k]['aic']
        print(f"{k:9s} n={n:2d} gamma={pl['gamma']:.2f} (68% CI {bs['gamma_ci'][0]:.2f}-{bs['gamma_ci'][2]:.2f}) "
              f"f_c={pl['fc']:.3f} R2={pl['r2']:.3f} | acceptable f_c {adm['fc_range'][0]:.2f}-{adm['fc_range'][1]:.2f}"
              f" -> gamma {adm['gamma_range'][0]:.2f}-{adm['gamma_range'][1]:.2f} | AIC PL {A['powerlaw']:.1f}"
              f" VFT {A['vft']:.1f} Arrh {A['arrhenius']:.1f} | windows "
              f"{ {w: round(v['gamma'], 2) for w, v in windows.items()} }")

    # censoring check: arrest-time law must predict t_arr(1.8) > end of simulation
    r = fits['t_arr']['powerlaw']
    t18 = r['A'] * (r['fc'] - 1.8) ** (-r['gamma']) if r['fc'] > 1.8 else np.inf
    print(f"\nt_arr law (fit on f<=1.65) predicts t_arr(1.8) = {t18:.3g} (simulation ends at {D[1.8]['t'][-1]:.0f})")
    print(f"<t_arr / tau_K> over arrested forces = {mc['t_arr_over_tauK']:.2f}; "
          f"=> expected arrest of f=1.8 at t ~ {mc['t_arr_over_tauK'] * taus['tau_K'][forces_all.index(1.8)]:.3g}")
    print(f"tau_K(2.0) = {taus['tau_K'][forces_all.index(2.0)]:.3g} (from a drift of only "
          f"{(D[2.0]['y'][-1000:].mean() - D[2.0]['y'][4000:5000].mean()):.3f} decades; treat as lower bound)")

    print("\n=== Master curve collapse, forces %s ===" % forces_mc)
    for m in ('raw', '1param', '2param'):
        print(f"  {m:7s} mean variance of log10 K across forces: {mc[m]['cost']:.4f}")
    a = mc['2param']['a']
    tmc = mc['2param']['tau']
    print("  a(f):  ", dict(zip(forces_mc, np.round(a, 3).tolist())))
    print("  tau(f):", dict(zip(forces_mc, np.round(tmc).tolist())))
    fm = np.array(forces_mc)
    a_lin = np.polyfit(fm, a, 1)
    print(f"  a(f) ~ linear in f: slope {a_lin[0]:.3f} decades per unit force "
          f"(residual std {np.std(a - np.polyval(a_lin, fm)):.3f})")
    rates = np.array([c[2] for c in cliffs.values()])
    print(f"  final collapse rate: {np.nanmean(rates):.3f} +- {np.nanstd(rates):.3f} decades/time "
          f"(=> e-folding {1 / (-np.nanmean(rates) * np.log(10)):.2f} time units)")

    # ---- fluid side ----
    d = D[2.0]
    n2 = len(d['K']) // 2
    fluid = dict(f=2.0, K_ss=float(d['K'][n2:].mean()), K_ss_std=float(d['K'][n2:].std()))
    print(f"\n=== f=2.0 (only near-stationary trajectory): K = {fluid['K_ss']:.4e} +- {fluid['K_ss_std']:.1e}")

    plot_master(D, forces_mc, mc, cliffs, os.path.join(DOCS, "master_curve.png"))
    plot_tau_scaling(fa, taus, fits, os.path.join(DOCS, "tau_scaling_robust.png"))
    plot_fc_profile(fa, taus, fits, os.path.join(DOCS, "fc_profile.png"))

    result = dict(
        forces=forces_all,
        taus={k: v.tolist() for k, v in taus.items()},
        fits=fits,
        t_arr_pred_1p8=t18,
        master_curve=dict(forces=forces_mc, ref=1.4, cost_raw=mc['raw']['cost'], cost_1param=mc['1param']['cost'],
                          cost_2param=mc['2param']['cost'], a=a.tolist(), tau=tmc.tolist(),
                          x=mc['x'].tolist(), G=mc['G'].tolist(), a_linear_in_f=a_lin.tolist(),
                          t_arr_over_tauK=mc['t_arr_over_tauK']),
        cliff_rate=dict(mean=float(np.nanmean(rates)), std=float(np.nanstd(rates))),
        fluid=fluid,
    )
    with open(os.path.join(PROC, "scaling_results.json"), "w") as fh:
        json.dump(result, fh, indent=2, default=_json_default)
    print("\nSaved docs/master_curve.png, docs/tau_scaling_robust.png, docs/fc_profile.png, "
          "data/processed/scaling_results.json")
    return result


if __name__ == "__main__":
    run()
