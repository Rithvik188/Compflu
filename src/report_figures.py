"""
Publication-style figures for the technical report (docs/figures/fig*.png, 300 dpi).
Uses the saved results of scaling_analysis, havok, equation_discovery, koopman_forecast and
validation; recomputes only cheap quantities (collapse curves, HAVOK forcing, bootstraps).

Run from the project root (after the other scripts):  python src/report_figures.py
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from scipy.stats import kurtosis, norm
from data_loader import load_clean
import scaling_analysis as SA
import havok as HV
import equation_discovery as EQ

ROOT = os.path.join(os.path.dirname(__file__), "..")
PROC = os.path.join(ROOT, "data", "processed")
FIG = os.path.join(ROOT, "docs", "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif', 'mathtext.fontset': 'dejavuserif', 'font.size': 9,
    'axes.labelsize': 9, 'axes.titlesize': 9, 'legend.fontsize': 7, 'xtick.labelsize': 8,
    'ytick.labelsize': 8, 'lines.linewidth': 1.2, 'axes.linewidth': 0.8, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'legend.frameon': False, 'axes.grid': False,
})
CMAP = cm.viridis
NORM = colors.Normalize(0.0, 2.1)
MODEL_STYLE = {
    'NeighbourBaseline': dict(color='0.55', ls=':', label='Neighbour baseline'),
    'HDMD-param': dict(color='tab:blue', ls='--', label='Parametric HDMD'),
    'HDMD-prefix': dict(color='tab:blue', ls='--', label='HDMD on prefix'),
    'DeepKoopman': dict(color='tab:green', ls='-.', label='Deep Koopman AE'),
    'MasterCurve': dict(color='tab:red', ls='-', label='Master-curve Koopman'),
    'ODE-B': dict(color='tab:purple', ls='--', label='Discovered ODE'),
}


def load(name):
    with open(os.path.join(PROC, name)) as fh:
        return json.load(fh)


def abs_errors(rows):
    """|relative arrest-time error| per force; a missed (or false) arrest counts as 1.0."""
    out = []
    for r in rows:
        v = r['t_arr_rel_err']
        v = 1.0 if v is None or not np.isfinite(float(v)) else min(abs(float(v)), 1.0)
        out.append(v)
    return np.array(out)


def boot_median_ci(x, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    b = [np.median(rng.choice(x, len(x))) for _ in range(n)]
    return float(np.median(x)), tuple(np.percentile(b, [2.5, 97.5]).tolist())


def letter(ax, s):
    ax.text(-0.14, 1.04, f"({s})", transform=ax.transAxes, fontweight='bold', fontsize=10, va='bottom')


def cbar(fig, ax, label='active force $f$'):
    sm = cm.ScalarMappable(cmap=CMAP, norm=NORM)
    c = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
    c.set_label(label)


def save(fig, name):
    fig.savefig(os.path.join(FIG, name))
    plt.close(fig)
    print("  saved", name)


# ---------------------------------------------------------------------------
def fig1_data(D, S):
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for f in sorted(D):
        d = D[f]
        i = min(d['i_arr'] + 80, len(d['t']))
        ax[0].plot(d['t'][1:i], np.clip(d['y_s'][1:i], -20.5, None), color=CMAP(NORM(f)), lw=0.9)
    ax[0].set(xscale='log', xlabel='time $t$', ylabel=r'$\log_{10}K$', ylim=(-21, 0), xlim=(0.5, 3.2e4))
    cbar(fig, ax[0])
    letter(ax[0], 'a')
    fa = [f for f in sorted(D) if np.isfinite(D[f]['t_arr'])]
    ax[1].semilogy(fa, [D[f]['t_arr'] for f in fa], 'o', color='k', ms=4, label='arrest observed')
    for f in (1.8, 2.0):
        ax[1].annotate('', xy=(f, 1.6e5), xytext=(f, D[f]['t'][-1]),
                       arrowprops=dict(arrowstyle='->', color='tab:red', lw=1.2))
        ax[1].plot(f, D[f]['t'][-1], 's', color='tab:red', ms=4)
    ax[1].plot([], [], 's', color='tab:red', ms=4, label='no arrest by end of run (lower bound)')
    ax[1].axhline(D[1.8]['t'][-1], color='0.6', lw=0.6, ls=':')
    ax[1].set(xlabel='active force $f$', ylabel=r'arrest time $t_{\rm arr}$', ylim=(80, 3e5))
    ax[1].legend(loc='upper left')
    letter(ax[1], 'b')
    plt.tight_layout()
    save(fig, "fig1_data.png")


def fig2_master(D, S, V):
    mcS = S['master_curve']
    forces = mcS['forces']
    mc = SA.master_curve(D, forces, ref=1.4)
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.6))
    ax = ax.ravel()
    for i, f in enumerate(forces):
        d = D[f]
        i_end = d['i_arr']
        ls = '-' if np.isfinite(d['t_arr']) else ':'
        ax[0].plot(d['t'][:i_end], d['y_s'][:i_end], ls, color=CMAP(NORM(f)), lw=0.9)
        ax[1].plot(mc['x'], mc['2param']['curves'][i], ls, color=CMAP(NORM(f)), lw=0.9)
    ax[1].plot(mc['x'], mc['G'], 'k--', lw=1.6, label='$G(x)$ (median)')
    ax[0].set(xlabel='$t$', ylabel=r'$\log_{10}K$', xlim=(0, 3e4))
    ax[0].text(0.97, 0.95, f"spread {mcS['cost_raw']:.2f}", transform=ax[0].transAxes, ha='right', va='top')
    ax[1].set(xlabel=r'$x = t/\tau(f)$', ylabel=r'$\log_{10}K - a(f)$')
    ax[1].text(0.97, 0.95, f"spread {mcS['cost_2param']:.3f}\n$R^2$ = {1 - mcS['cost_2param'] / mcS['cost_raw']:.3f}",
               transform=ax[1].transAxes, ha='right', va='top')
    ax[1].legend(loc='lower left')
    cbar(fig, ax[1])
    v2 = V['V2']
    tk, ta = np.array(v2['tau_K']), np.array(v2['t_arr'])
    for f, a_, b_ in zip(v2['forces'], tk, ta):
        ax[2].loglog(a_, b_, 'o', color=CMAP(NORM(f)), ms=5)
    xx = np.logspace(np.log10(tk.min() * 0.8), np.log10(tk.max() * 1.2), 50)
    c = np.polyfit(np.log(tk), np.log(ta), 1)
    ax[2].loglog(xx, np.exp(np.polyval(c, np.log(xx))), 'k-', lw=1,
                 label=fr"$t_{{\rm arr}}\propto\tau_K^{{\beta}}$, $\beta$={v2['beta']:.2f} "
                       fr"[{v2['beta_ci95'][0]:.2f}, {v2['beta_ci95'][1]:.2f}]")
    ax[2].loglog(xx, v2['ratio_mean'] * xx, 'k:', lw=1, label=r'$\beta = 1$')
    ax[2].set(xlabel=r'slow-stage decay time $\tau_K$ (fitted without the collapse)', ylabel=r'arrest time $t_{\rm arr}$')
    ax[2].legend(loc='upper left')
    for f in [g for g in forces if np.isfinite(D[g]['t_arr'])]:
        dt_, y_, _ = SA.cliff_profile(D[f])
        ax[3].plot(dt_, y_, color=CMAP(NORM(f)), lw=0.9)
    cr = S['cliff_rate']
    ax[3].plot([0, 35], [-7.2, -7.2 + cr['mean'] * 35], 'k--', lw=1,
               label=f"slope {cr['mean']:.2f} $\\pm$ {cr['std']:.2f} decades per t.u.")
    ax[3].set(xlabel=r'$t - t_{\rm cliff}$', ylabel=r'$\log_{10}K$', xlim=(-3, 60), ylim=(-21, -5))
    ax[3].legend(loc='upper right')
    for a_, l in zip(ax, 'abcd'):
        letter(a_, l)
    plt.tight_layout()
    save(fig, "fig2_master_curve.png")
    return mc


def fig3_critical(D, S, V):
    fa = np.array(S['forces'])
    taus = {k: np.array(v, float) for k, v in S['taus'].items()}
    use = {'t_arr': r'$t_{\rm arr}$', 't_-5': r'$t(K{=}10^{-5})$', 'tau_K': r'$\tau_K$', 'tau_mc': r'$\tau_{\rm mc}$'}
    mk = dict(zip(use, 'osD^'))
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.6))
    ax = ax.ravel()
    rng = np.random.default_rng(0)
    for k, lab in use.items():
        fit = S['fits'][k]
        ok = np.isfinite(taus[k]) & np.isin(fa, fit['forces'])
        f_, t_ = fa[ok], taus[k][ok]
        pl = fit['powerlaw']
        l, = ax[0].semilogy(f_, t_, mk[k], ms=4, label=lab)
        ff = np.linspace(f_.min(), f_.max(), 100)
        ax[0].semilogy(ff, pl['A'] * (pl['fc'] - ff) ** (-pl['gamma']), '-', color=l.get_color(), lw=0.8)
        ax[1].loglog(pl['fc'] - f_, t_, mk[k], ms=4, color=l.get_color(),
                     label=fr"{lab}: $\gamma$={pl['gamma']:.2f}, $f_c$={pl['fc']:.2f}")
        xx = np.logspace(np.log10((pl['fc'] - f_).min()), np.log10((pl['fc'] - f_).max()), 30)
        ax[1].loglog(xx, pl['A'] * xx ** (-pl['gamma']), '-', color=l.get_color(), lw=0.8)
        grid, gam, sse = SA.profile_fc(f_, t_)
        n = len(f_)
        daic = n * np.log(sse / n)
        daic -= daic.min()
        ax[2].plot(grid, daic, color=l.get_color(), label=lab)
        adm = daic <= 2
        ax[3].plot(grid, gam, color=l.get_color(), alpha=0.3, lw=0.8)
        ax[3].plot(grid[adm], gam[adm], color=l.get_color(), lw=2.2)
        if k in ('tau_K', 'tau_mc'):
            bs = []
            for _ in range(300):
                i = rng.integers(0, n, n)
                if len(np.unique(f_[i])) >= 4:
                    r = SA.fit_powerlaw(f_[i], t_[i])
                    bs.append((r['fc'], r['gamma']))
            bs = np.array(bs)
            ax[3].scatter(bs[:, 0], bs[:, 1], s=3, color=l.get_color(), alpha=0.25, lw=0)
    ax[0].set(xlabel='$f$', ylabel=r'relaxation time $\tau$')
    ax[0].legend(loc='upper left')
    ax[1].set(xlabel=r'$f_c - f$', ylabel=r'$\tau$')
    ax[1].legend(loc='lower left', fontsize=6)
    ax[2].axhline(2, color='k', ls=':', lw=0.8)
    ax[2].set(xlabel=r'assumed $f_c$', ylabel=r'$\Delta$AIC', ylim=(-0.5, 15), xlim=(1.7, 2.4))
    ax[2].legend(loc='upper right')
    ax[3].set(xlabel=r'$f_c$', ylabel=r'$\gamma$', xlim=(1.7, 2.4), ylim=(0.5, 5))
    ax[3].text(0.03, 0.95, 'lines: best $\\gamma$ for each $f_c$ (thick: $\\Delta$AIC<2)\ndots: bootstrap fits',
               transform=ax[3].transAxes, va='top', fontsize=7)
    for a_, l in zip(ax, 'abcd'):
        letter(a_, l)
    plt.tight_layout()
    save(fig, "fig3_critical_scaling.png")


def fig4_havok(D, H, V):
    rows = {r['f']: r for r in H['forces']}
    fig = plt.figure(figsize=(7.2, 5.8))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.25])
    a0 = fig.add_subplot(gs[0, :])
    a1 = fig.add_subplot(gs[1, :], sharex=a0)
    res, m, (t, dy, ev, act) = HV.analyse_force(D[1.6], 1.6)
    tc = t[HV.Q - 1:]
    a0.plot(t, dy, 'k', lw=0.4)
    a0.plot(t[ev], dy[ev], 'v', color='tab:orange', ms=3, label='dip events')
    a0.set(ylabel=r'$\delta\log_{10}K$')
    a0.legend(loc='lower left')
    a0.set_title(r'$f$ = 1.6: fluctuation of $\log_{10}K$ around its slow trend (top) and HAVOK forcing (bottom)',
                 loc='left')
    a1.plot(tc, m.forcing, 'k', lw=0.4)
    a1.plot(tc[act], m.forcing[act], '.', color='tab:red', ms=1.5, label='active forcing ($|v_r|>2.5\\sigma$)')
    a1.set(xlabel='$t$ (end of delay window)', ylabel='forcing $v_r$')
    a1.legend(loc='upper left')
    letter(a0, 'a')
    a2 = fig.add_subplot(gs[2, 0])
    bins = np.linspace(-8, 8, 61)
    for f in (1.0, 1.4, 1.65, 1.8, 2.0):
        _, mf, _ = HV.analyse_force(D[f], f)
        z = (mf.forcing - mf.forcing.mean()) / mf.forcing.std()
        h, e = np.histogram(z, bins=bins, density=True)
        c = 0.5 * (e[1:] + e[:-1])
        a2.semilogy(c[h > 0], h[h > 0], '-', color=CMAP(NORM(f)), lw=1, label=f'$f$={f}')
    a2.semilogy(bins, norm.pdf(bins), 'k--', lw=1, label='Gaussian')
    a2.set(xlabel=r'standardised forcing $v_r/\sigma$', ylabel='PDF', ylim=(1e-5, 1))
    a2.legend(loc='lower center', ncol=2, fontsize=6)
    letter(a2, 'b')
    a3 = fig.add_subplot(gs[2, 1])
    v3 = V['V3']
    fs = sorted(float(k) for k in v3)
    kd = np.array([v3[str(f) if str(f) in v3 else f]['kurtosis_data'] for f in fs])
    ci = np.array([v3[str(f) if str(f) in v3 else f]['kurtosis_data_ci95'] for f in fs])
    ks = np.array([v3[str(f) if str(f) in v3 else f]['kurtosis_surrogates'] for f in fs])
    a3.errorbar(fs, kd, yerr=[kd - ci[:, 0], ci[:, 1] - kd], fmt='o', color='k', ms=3.5, capsize=0,
                elinewidth=0.8, label='data (95% block-bootstrap CI)')
    a3.fill_between(fs, ks[:, 0], ks[:, 2], color='tab:blue', alpha=0.3, label='19 phase-randomised surrogates')
    a3.axhline(0, color='k', lw=0.6)
    a3.set_yscale('symlog', linthresh=1)
    a3.set(xlabel='$f$', ylabel='forcing kurtosis', ylim=(-1, 5000))
    a3.legend(loc='upper center', fontsize=6, ncol=1)
    letter(a3, 'c')
    plt.tight_layout()
    save(fig, "fig4_havok.png")


def fig5_equation(D, E, V, P):
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.6))
    ax = ax.ravel()
    L = E['model_L']
    band = V['V5']['bootstrap']['g_band']
    ax[0].fill_between(band['u'], band['lo'], band['hi'], color='tab:purple', alpha=0.25, label='95% bootstrap band')
    xi = np.array([E['model_B']['g_coefficients'][n] for n in EQ.B_NAMES])
    uu = np.linspace(-6.5, -2.5, 100)
    ax[0].plot(uu, EQ.theta_B(uu) @ xi, color='tab:purple', label='weak-SINDy $g(u)$')
    ax[0].plot(L['u_centers'], L['drift'], 'ko', ms=3.5, label='Kramers-Moyal drift (data)')
    ax[0].set(xlabel=r'$u = \log_{10}K - a(f)$', ylabel=r'$du/ds$', ylim=(-70, 2))
    ax[0].legend(loc='lower right')
    if P is not None:
        for f in (1.0, 1.25, 1.45, 1.6, 1.65):
            t, y = P[f"t_{f}"], P[f"y_{f}"]
            ax[1].plot(t, y, color=CMAP(NORM(f)), lw=2.5, alpha=0.35)
            ax[1].plot(t, P[f"a_{f}_ODE-B"], '--', color=CMAP(NORM(f)), lw=1)
        ax[1].plot([], [], 'k', lw=2.5, alpha=0.35, label='data')
        ax[1].plot([], [], 'k--', lw=1, label='ODE, force left out of fit')
        ax[1].set(xscale='log', xlim=(10, 3e4), ylim=(-21, -1), xlabel='$t$', ylabel=r'$\log_{10}K$')
        ax[1].legend(loc='lower left')
    uc, d2 = np.array(L['u_centers']), np.array(L['diffusion'])
    ax[2].semilogy(uc, d2, 'ko', ms=3.5, label='Kramers-Moyal $D_2$')
    ax[2].semilogy(uu, np.exp(np.polyval(L['log_D2_poly'], uu)), 'tab:purple', label='fit')
    ax[2].set(xlabel='$u$', ylabel=r'$D_2(u)$')
    ax[2].legend()
    # Langevin arrest-time distribution
    cl = dict(xi=xi)
    s_sim = EQ.langevin_arrest_times(xi, np.array(L['log_D2_poly']), float(np.median(
        [np.interp(EQ.T0, D[f]['t'], D[f]['y_s']) - (E['model_B']['a_linear'][0] * f + E['model_B']['a_linear'][1])
         for f in [g for g in sorted(D) if 1.0 <= g <= 1.65]])), E['u_on'], n_traj=2000)
    s_sim = s_sim[np.isfinite(s_sim)]
    ax[3].hist(s_sim, bins=40, density=True, color='tab:purple', alpha=0.4, label='Langevin model (2000 runs)')
    for s in L['s_arrest_obs']:
        ax[3].axvline(s, color='k', lw=0.8, ymax=0.25)
    ax[3].plot([], [], 'k', lw=0.8, label='observed (11 forces)')
    ax[3].set(xlabel=r'arrest time in rescaled units $s^* = t_{\rm arr}/\tau(f)$', ylabel='PDF', xlim=(0, 3.5))
    ax[3].legend()
    for a_, l in zip(ax, 'abcd'):
        letter(a_, l)
    plt.tight_layout()
    save(fig, "fig5_equation.png")


def fig6_forecast(Fr, P):
    fa = Fr['protocol_a']
    forces = sorted(float(k) for k in fa)
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.8))
    ax = ax.ravel()
    for f, a_ in zip((1.0, 1.4), (ax[0], ax[1])):
        t, y = P[f"t_{f}"], P[f"y_{f}"]
        a_.plot(t, y, 'k', lw=3, alpha=0.25, label='simulation (never seen)')
        for m in ('NeighbourBaseline', 'HDMD-param', 'DeepKoopman', 'MasterCurve', 'ODE-B'):
            st = MODEL_STYLE[m]
            a_.plot(t, P[f"a_{f}_{m}"], ls=st['ls'], color=st['color'], lw=1, label=st['label'])
        a_.set(xscale='log', xlim=(5, 3e4), ylim=(-21, -1), xlabel='$t$', ylabel=r'$\log_{10}K$',
               title=f'$f$ = {f}' + (' (extrapolation)' if f == 1.0 else ' (interpolation)'))
    ax[0].legend(loc='lower left', fontsize=6)
    top = 1.2e5
    for m in ('NeighbourBaseline', 'HDMD-param', 'DeepKoopman', 'MasterCurve', 'ODE-B'):
        st = MODEL_STYLE[m]
        tt, tp = [], []
        for f in forces:
            r = fa[str(f) if str(f) in fa else f][m]
            if r['t_arr_true'] is None or not np.isfinite(float(r['t_arr_true'])):
                continue
            tt.append(float(r['t_arr_true']))
            v = r['t_arr_pred']
            tp.append(top if v is None or not np.isfinite(float(v)) else float(v))
        ax[2].loglog(tt, tp, 'o' if m in ('MasterCurve', 'ODE-B') else 'x', color=st['color'], ms=4,
                     label=st['label'], mfc='none' if m == 'ODE-B' else st['color'])
    lim = (1e3, 1.5e5)
    ax[2].plot(lim, lim, 'k-', lw=0.6)
    ax[2].fill_between(lim, [l / 1.25 for l in lim], [l * 1.25 for l in lim], color='0.85', zorder=0, label='$\\pm$25%')
    ax[2].axhline(top, color='0.4', ls=':', lw=0.6)
    ax[2].text(1.1e3, top * 1.08, 'no arrest predicted', fontsize=6, color='0.3')
    ax[2].set(xlim=(1e3, 3e4), ylim=(3e2, 1.6e5), xlabel=r'true arrest time', ylabel='predicted arrest time',
              title='unseen force: all left-out forces')
    ax[2].legend(loc='upper left', fontsize=6, bbox_to_anchor=(0.0, 0.93))
    fb = Fr['protocol_b']
    ps = sorted(next(iter(fb.values())).keys(), key=float)
    for m in ('HDMD-prefix', 'DeepKoopman', 'MasterCurve', 'ODE-B'):
        st = MODEL_STYLE[m]
        med, lo, hi = [], [], []
        for p in ps:
            e = abs_errors([fb[k][p][m] for k in fb if float(k) <= 1.65])
            md, (l_, h_) = boot_median_ci(e)
            med.append(md)
            lo.append(l_)
            hi.append(h_)
        x = [100 * float(p) for p in ps]
        ax[3].plot(x, med, marker='o', color=st['color'], ls=st['ls'], label=st['label'])
        if np.all(np.isfinite(lo)):
            ax[3].fill_between(x, lo, hi, color=st['color'], alpha=0.12)
    ax[3].set(xlabel='prefix given (% of arrest time)', ylabel=r'median $|\Delta t_{\rm arr}|/t_{\rm arr}$',
              ylim=(0, 1.45), title='forecast from a prefix of the unseen run')
    ax[3].legend(loc='upper center', fontsize=6, ncol=2)
    for a_, l in zip(ax, 'abcd'):
        letter(a_, l)
    plt.tight_layout()
    save(fig, "fig6_forecast.png")


def fig7_evaluation(D):
    import koopman_forecast as KF
    KF.QUICK = False
    res, curves = KF.original_model_check(D)
    fig, ax = plt.subplots(1, 4, figsize=(7.2, 2.1), sharey=True)
    for a_, (f, (t, truth, tf, free)) in zip(ax, curves.items()):
        a_.plot(t, truth, 'k', lw=2.5, alpha=0.3, label='simulation')
        a_.plot(t, tf, '--', color='tab:blue', lw=0.9, label='one-step-ahead')
        a_.plot(t, np.clip(free, -30, 5), '-', color='tab:red', lw=0.9, label='free-running')
        a_.set(title=f'$f$ = {f}', xlabel='$t$', ylim=(-28, 1))
    ax[0].set_ylabel(r'$\log_{10}(K+10^{-12})$')
    ax[0].legend(loc='lower left', fontsize=6)
    plt.tight_layout()
    save(fig, "fig7_onestep_vs_free.png")
    return res


def figS1_methods(V):
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.4))
    HV.lorenz_check(None)
    from scipy.integrate import solve_ivp
    dt = 0.001
    t = np.arange(0, 60, dt)
    sol = solve_ivp(lambda t, s: [10 * (s[1] - s[0]), s[0] * (28 - s[2]) - s[1], s[0] * s[1] - 8 / 3 * s[2]],
                    (0, t[-1]), [-8, 8, 27], t_eval=t, rtol=1e-10, atol=1e-10)
    x = sol.y[0]
    r, err, m, _ = HV.select_rank(x, dt, 100, range(7, 16), int(20 / dt))
    act = HV.active_forcing(m.forcing)
    tc = t[99:]
    ax0 = ax[0]
    ax0.plot(tc, x[99:] / np.abs(x).max(), 'k', lw=0.5, label='$x(t)$ (scaled)')
    ax0.plot(tc, m.forcing / np.abs(m.forcing).max() - 1.6, color='0.4', lw=0.5, label='forcing $v_r$')
    ax0.plot(tc[act], m.forcing[act] / np.abs(m.forcing).max() - 1.6, '.', color='tab:red', ms=1)
    ax0.set(xlabel='$t$', yticks=[], ylim=(-2.9, 2.3), ylabel='HAVOK on Lorenz')
    ax0.legend(loc='upper center', fontsize=6, ncol=2)
    v6 = V['V6']
    amps = sorted(float(k) for k in v6)
    ax[1].plot(amps, [v6[str(a) if str(a) in v6 else a]['median_max_rel_coef_err'] for a in amps], 'o-k', ms=3.5)
    ax2 = ax[1].twinx()
    ax2.plot(amps, [v6[str(a) if str(a) in v6 else a]['correct_terms_fraction'] for a in amps], 's--',
             color='tab:red', ms=3.5)
    ax2.set_ylabel('correct terms found', color='tab:red')
    ax2.set_ylim(0, 1.05)
    ax[1].axvline(0.1, color='0.5', ls=':', lw=0.8)
    ax[1].text(0.105, 0.08, 'data noise', fontsize=6, color='0.4')
    ax[1].set_xscale('log')
    ax[1].set_xticks(amps, labels=[f'{a:g}' for a in amps])
    ax[1].minorticks_off()
    ax[1].set(xlabel='noise amplitude (decades)', ylabel='weak-SINDy coefficient error')
    v7 = V['V7']
    keys = list(v7)
    ax[2].plot(range(len(keys)), [v7[k]['gamma'] for k in keys], 'o-', color='k', ms=3.5, label=r'$\gamma$')
    ax3 = ax[2].twinx()
    ax3.plot(range(len(keys)), [v7[k]['fc'] for k in keys], 's--', color='tab:blue', ms=3.5)
    ax3.set_ylabel('$f_c$', color='tab:blue')
    ax3.set_ylim(1.8, 2.0)
    ax[2].set(xticks=range(len(keys)), xticklabels=[k.replace('-', '\n') for k in keys], ylim=(1.5, 2.3),
              ylabel=r'$\gamma$', xlabel=r'$\tau_K$ fit window (start t, end fraction)')
    ax[2].tick_params(axis='x', labelsize=6)
    for a_, l in zip(ax, 'abc'):
        letter(a_, l)
    plt.tight_layout()
    save(fig, "figS1_method_validation.png")


def main():
    D = load_clean()
    S, H, E, V = load("scaling_results.json"), load("havok_results.json"), load("equation_results.json"), load(
        "validation_results.json")
    pf = os.path.join(PROC, "forecast_predictions.npz")
    P = dict(np.load(pf)) if os.path.exists(pf) else None
    Fr = load("forecast_results.json")
    fig1_data(D, S)
    fig2_master(D, S, V)
    fig3_critical(D, S, V)
    fig4_havok(D, H, V)
    fig5_equation(D, E, V, P)
    if P is not None:
        fig6_forecast(Fr, P)
    res = fig7_evaluation(D)
    with open(os.path.join(PROC, "onestep_vs_free.json"), "w") as fh:
        json.dump({str(k): v for k, v in res.items()}, fh, indent=2)
    figS1_methods(V)


if __name__ == "__main__":
    main()
