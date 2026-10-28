"""
Slide-optimised figures for the conference talk (presentation/figures/*.png).

Each figure is drawn at the exact size (inches) it occupies on the 20 x 11.25 in slide, so
font sizes are real slide point sizes. Style follows the Canva template: charcoal on white,
transparent background, one accent colour for the key result.

Run from the project root (after the analysis scripts):  python src/slide_figures.py
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.stats import norm
from data_loader import load_clean
import scaling_analysis as SA
import havok as HV
import equation_discovery as EQ

ROOT = os.path.join(os.path.dirname(__file__), "..")
PROC = os.path.join(ROOT, "data", "processed")
OUT = os.path.join(ROOT, "presentation", "figures")
os.makedirs(OUT, exist_ok=True)

INK = "#424242"
MID = "#8a8a8a"
LIGHT = "#cfcfcf"
ACCENT = "#c0502e"
plt.rcParams.update({
    'font.family': 'Segoe UI', 'mathtext.fontset': 'dejavusans', 'font.size': 15,
    'axes.labelsize': 16, 'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 13,
    'axes.edgecolor': INK, 'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK,
    'text.color': INK, 'axes.linewidth': 1.0, 'lines.linewidth': 1.8,
    'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
    'savefig.dpi': 250, 'savefig.transparent': True,
})
GREYS = colors.LinearSegmentedColormap.from_list("g", ["#d2d2d2", "#1f1f1f"])
FNORM = colors.Normalize(1.0, 2.0)


def load(name):
    with open(os.path.join(PROC, name)) as fh:
        return json.load(fh)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)
    print("  saved", name)


def fcol(f):
    return GREYS(FNORM(min(max(f, 1.0), 2.0)))


# ---------------------------------------------------------------------------
def title_strip(D):
    fig, ax = plt.subplots(figsize=(5.0, 7.16))
    for f in sorted(D):
        if f < 1.0:
            continue
        d = D[f]
        i = min(d['i_arr'] + 60, len(d['t']))
        c = ACCENT if f == 1.65 else fcol(f)
        ax.plot(d['t'][1:i], np.clip(d['y_s'][1:i], -20, None), color=c, lw=2.2 if f == 1.65 else 1.3)
    ax.set_xscale('log')
    ax.set(xlim=(1, 3.2e4), ylim=(-20.5, -0.5))
    ax.set_xlabel('time', fontsize=14)
    ax.set_ylabel(r'$\log_{10}$ kinetic energy', fontsize=14)
    ax.tick_params(labelsize=12)
    fig.tight_layout()
    save(fig, "title_strip.png")


def data_overview(D):
    fig, ax = plt.subplots(figsize=(7.6, 8.2))
    for f in sorted(D):
        d = D[f]
        i = min(d['i_arr'] + 60, len(d['t']))
        c = '#b9b9b9' if f < 1.0 else fcol(f)
        ax.plot(d['t'][1:i], np.clip(d['y_s'][1:i], -20.5, None), color=c, lw=1.4)
    ax.set_xscale('log')
    ax.set(xlim=(0.5, 3.2e4), ylim=(-21, 0), xlabel='time  $t$', ylabel=r'$\log_{10} K$')
    ax.text(0.7, -18.5, 'f ≤ 0.75\nquench-like', color=MID, fontsize=14)
    ax.annotate('f = 1.0 … 1.65\nslow decay, then arrest', xy=(3000, -12), xytext=(18, -14.5),
                fontsize=14, arrowprops=dict(arrowstyle='->', color=INK, lw=1))
    ax.text(40, -1.2, 'f = 1.8, 2.0: no arrest in the run', fontsize=14, color=ACCENT)
    sm = plt.cm.ScalarMappable(cmap=GREYS, norm=FNORM)
    cb = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label('active force  $f$')
    cb.outline.set_edgecolor(INK)
    fig.tight_layout()
    save(fig, "data_overview.png")


def anatomy(D):
    d = D[1.4]
    fig, ax = plt.subplots(figsize=(10.2, 6.0))
    i = min(d['i_arr'] + 80, len(d['t']))
    ax.plot(d['t'][:i], d['y'][:i], color=LIGHT, lw=0.6)
    ax.plot(d['t'][:i], np.clip(d['y_s'][:i], -20, None), color=INK, lw=2.2)
    ta = d['t_arr']
    ax.axvspan(0.5, 20, color='#ececec')
    ax.axvspan(20, ta - 80, color='#f6f6f6')
    ax.axvspan(ta - 80, ta * 1.25, color='#f8e6e0')
    ax.text(0.7, -16.0, '1  transient\n    (t < 20)', fontsize=14)
    ax.text(40, -16.0, '2  slow, almost exponential decay\n    $K \\propto e^{-t/\\tau_K}$, lasts thousands of t.u.', fontsize=14)
    ax.annotate('3  abrupt collapse\n    = arrest', xy=(ta * 0.995, -13), xytext=(ta * 0.06, -9.5), fontsize=14, color=ACCENT,
                arrowprops=dict(arrowstyle='->', color=ACCENT, lw=1.4))
    ax.set_xscale('log')
    ax.set(xlim=(0.5, ta * 1.25), ylim=(-20.5, -0.5), xlabel='time  $t$  (f = 1.4)', ylabel=r'$\log_{10} K$')
    fig.tight_layout()
    save(fig, "anatomy.png")


def arrest_times(D):
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    fa = [f for f in sorted(D) if np.isfinite(D[f]['t_arr'])]
    ax.semilogy(fa, [D[f]['t_arr'] for f in fa], 'o', color=INK, ms=7)
    for f in (1.8, 2.0):
        ax.annotate('', xy=(f, 1.5e5), xytext=(f, D[f]['t'][-1]), arrowprops=dict(arrowstyle='->', color=ACCENT, lw=2))
        ax.plot(f, D[f]['t'][-1], 's', color=ACCENT, ms=7)
    ax.axhline(D[1.8]['t'][-1], color=MID, lw=0.8, ls=':')
    ax.text(0.0, 3.6e4, 'end of simulation', fontsize=12, color=MID)
    ax.set(xlabel='active force  $f$', ylabel='arrest time', ylim=(150, 3e5))
    fig.tight_layout()
    save(fig, "arrest_times.png")


def collapse(D, S):
    mcS = S['master_curve']
    forces = mcS['forces']
    mc = SA.master_curve(D, forces, ref=1.4)
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    for f in forces:
        d = D[f]
        ls = '-' if np.isfinite(d['t_arr']) else ':'
        ax.plot(d['t'][:d['i_arr']], d['y_s'][:d['i_arr']], ls, color=fcol(f), lw=1.4)
    ax.set(xlabel='time  $t$', ylabel=r'$\log_{10} K$', xlim=(0, 3e4), ylim=(-10, -1.5))
    fig.tight_layout()
    save(fig, "collapse_raw.png")
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    for i, f in enumerate(forces):
        ls = '-' if np.isfinite(D[f]['t_arr']) else ':'
        ax.plot(mc['x'], mc['2param']['curves'][i], ls, color=fcol(f), lw=1.3)
    ax.plot(mc['x'], mc['G'], color=ACCENT, lw=3, label='master curve  G(x)')
    ax.set(xlabel=r'rescaled time  $x = t/\tau(f)$', ylabel=r'$\log_{10} K - a(f)$', ylim=(-10, -1.5))
    ax.legend(loc='lower left')
    fig.tight_layout()
    save(fig, "collapse_master.png")


def collapse_tests(D, S, V):
    v2 = V['V2']
    tk, ta = np.array(v2['tau_K']), np.array(v2['t_arr'])
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    ax.loglog(tk, ta, 'o', color=INK, ms=8)
    xx = np.logspace(np.log10(tk.min() * 0.8), np.log10(tk.max() * 1.25), 50)
    c = np.polyfit(np.log(tk), np.log(ta), 1)
    ax.loglog(xx, np.exp(np.polyval(c, np.log(xx))), color=ACCENT, lw=2.4,
              label=fr"$t_{{arr}} \propto \tau_K^{{{v2['beta']:.2f}}}$,  $R^2$ = {v2['r2']:.2f}")
    ax.set(xlabel=r'slow-decay time  $\tau_K$  (no cliff used)', ylabel='arrest time  $t_{arr}$')
    ax.legend(loc='upper left')
    fig.tight_layout()
    save(fig, "tarr_vs_tauK.png")
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    for f in [g for g in S['master_curve']['forces'] if np.isfinite(D[g]['t_arr'])]:
        dt_, y_, _ = SA.cliff_profile(D[f])
        ax.plot(dt_, y_, color=fcol(f), lw=1.2)
    cr = S['cliff_rate']
    ax.plot([0, 33], [-7.2, -7.2 + cr['mean'] * 33], color=ACCENT, lw=2.6, ls='--',
            label=f"slope {cr['mean']:.2f} ± {cr['std']:.2f} decades / t.u.")
    ax.set(xlabel=r'time after the cliff starts', ylabel=r'$\log_{10} K$', xlim=(-2, 45), ylim=(-21, -5))
    ax.legend(loc='upper right')
    fig.tight_layout()
    save(fig, "cliff.png")


def critical(S):
    fa = np.array(S['forces'])
    taus = {k: np.array(v, float) for k, v in S['taus'].items()}
    use = {'t_arr': (r'$t_{arr}$', 'o', INK), 'tau_K': (r'$\tau_K$', 'D', MID), 'tau_mc': (r'$\tau_{mc}$', '^', ACCENT)}
    fig, ax = plt.subplots(figsize=(7.6, 5.3))
    fig2, ax2 = plt.subplots(figsize=(7.6, 5.3))
    for k, (lab, mk, col) in use.items():
        fit = S['fits'][k]
        ok = np.isfinite(taus[k]) & np.isin(fa, fit['forces'])
        f_, t_ = fa[ok], taus[k][ok]
        pl = fit['powerlaw']
        x = pl['fc'] - f_
        ax.loglog(x, t_, mk, color=col, ms=7, label=fr"{lab}: $\gamma$ = {pl['gamma']:.2f}, $f_c$ = {pl['fc']:.2f}")
        xx = np.logspace(np.log10(x.min()), np.log10(x.max()), 30)
        ax.loglog(xx, pl['A'] * xx ** (-pl['gamma']), color=col, lw=1.4)
        grid, gam, sse = SA.profile_fc(f_, t_)
        n = len(f_)
        daic = n * np.log(sse / n)
        daic -= daic.min()
        adm = daic <= 2
        ax2.plot(grid, gam, color=col, lw=1, alpha=0.35)
        ax2.plot(grid[adm], gam[adm], color=col, lw=5, solid_capstyle='round', label=lab)
    ax.set(xlabel=r'distance to the critical force  $f_c - f$', ylabel=r'relaxation time  $\tau$')
    ax.legend(loc='lower left', fontsize=12)
    ax2.set(xlabel=r'assumed critical force  $f_c$', ylabel=r'exponent  $\gamma$', xlim=(1.7, 2.2), ylim=(0.8, 3.2))
    ax2.text(1.71, 3.0, 'thick: statistically acceptable (ΔAIC < 2)', fontsize=12, color=MID)
    ax2.legend(loc='lower right')
    fig.tight_layout()
    fig2.tight_layout()
    save(fig, "critical_loglog.png")
    save(fig2, "critical_degeneracy.png")


def havok_figs(D, V):
    res, m, (t, dy, ev, act) = HV.analyse_force(D[1.6], 1.6)
    tc = t[HV.Q - 1:]
    fig, ax = plt.subplots(2, 1, figsize=(10.0, 4.4), sharex=True, gridspec_kw=dict(hspace=0.12))
    ax[0].plot(t, dy, color=INK, lw=0.5)
    ax[0].set_ylabel('δ log K', fontsize=14)
    ax[1].plot(tc, m.forcing, color=MID, lw=0.4)
    ax[1].plot(tc[act], m.forcing[act], '.', color=ACCENT, ms=2.5)
    ax[1].set_ylabel('forcing $v_r$', fontsize=14)
    ax[1].set_xlabel('time  $t$   (f = 1.6, arrest at t ≈ 16 700)')
    for a in ax:
        a.tick_params(labelsize=12)
    fig.align_ylabels()
    fig.subplots_adjust(left=0.1, right=0.98, top=0.97, bottom=0.17)
    save(fig, "havok_trace.png")
    v3 = V['V3']
    fs = sorted(float(k) for k in v3)
    kd = np.array([v3[str(f)]['kurtosis_data'] for f in fs])
    ci = np.array([v3[str(f)]['kurtosis_data_ci95'] for f in fs])
    ks = np.array([v3[str(f)]['kurtosis_surrogates'] for f in fs])
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.fill_between(fs, ks[:, 0], ks[:, 2], color=LIGHT, label='fake data without bursts')
    ax.errorbar(fs, kd, yerr=[kd - ci[:, 0], ci[:, 1] - kd], fmt='o', color=ACCENT, ms=7, elinewidth=1.2,
                label='simulations (95 % CI)')
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_yscale('symlog', linthresh=1)
    ax.set(xlabel='active force  $f$', ylabel='burstiness (kurtosis)', ylim=(-1, 400))
    ax.legend(loc='upper left', fontsize=12)
    fig.tight_layout()
    save(fig, "havok_kurtosis.png")
    # PDFs for backup
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    bins = np.linspace(-8, 8, 61)
    for f, c in ((1.0, INK), (1.4, MID), (1.65, ACCENT), (2.0, '#3d7a8c')):
        _, mf, _ = HV.analyse_force(D[f], f)
        z = (mf.forcing - mf.forcing.mean()) / mf.forcing.std()
        h, e = np.histogram(z, bins=bins, density=True)
        cc = 0.5 * (e[1:] + e[:-1])
        ax.semilogy(cc[h > 0], h[h > 0], color=c, lw=1.6, label=f'f = {f}')
    ax.semilogy(bins, norm.pdf(bins), 'k--', lw=1.2, label='Gaussian')
    ax.set(xlabel='standardised forcing', ylabel='probability density', ylim=(1e-5, 1))
    ax.legend(loc='upper left', fontsize=12)
    fig.tight_layout()
    save(fig, "havok_pdfs.png")


def lorenz_and_signal(D):
    from scipy.integrate import solve_ivp
    dt = 0.001
    t = np.arange(0, 40, dt)
    sol = solve_ivp(lambda t, s: [10 * (s[1] - s[0]), s[0] * (28 - s[2]) - s[1], s[0] * s[1] - 8 / 3 * s[2]],
                    (0, t[-1]), [-8, 8, 27], t_eval=t, rtol=1e-10, atol=1e-10)
    x = sol.y[0]
    r, err, m, _ = HV.select_rank(x, dt, 100, range(7, 16), int(20 / dt))
    act = HV.active_forcing(m.forcing)
    tc = t[99:]
    fig, ax = plt.subplots(figsize=(6.6, 2.7))
    ax.plot(tc, x[99:] / np.abs(x).max(), color=INK, lw=0.8, label='Lorenz x(t)')
    f_ = m.forcing / np.abs(m.forcing).max()
    ax.plot(tc, f_ - 1.9, color=MID, lw=0.6)
    ax.plot(tc[act], f_[act] - 1.9, '.', color=ACCENT, ms=2)
    ax.plot([], [], color=ACCENT, lw=2, label='HAVOK forcing (active)')
    ax.legend(loc='upper center', ncol=2, fontsize=12, bbox_to_anchor=(0.5, 1.22))
    ax.set(yticks=[], xlabel='t', ylim=(-2.95, 1.1))
    ax.spines['left'].set_visible(False)
    fig.tight_layout()
    save(fig, "lorenz.png")
    d = D[1.6]
    fig, ax = plt.subplots(figsize=(3.2, 2.0))
    ax.plot(d['t'][40:d['i_arr']:5], d['y'][40:d['i_arr']:5], color=INK, lw=0.5)
    ax.set(xticks=[], yticks=[])
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    save(fig, "signal_mini.png")


def equation_figs(D, E, V, P):
    band = V['V5']['bootstrap']['g_band']
    xi = np.array([E['model_B']['g_coefficients'][n] for n in EQ.B_NAMES])
    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    ax.fill_between(band['u'], band['lo'], band['hi'], color='#f2d6cc', label='95 % bootstrap band')
    uu = np.linspace(-6.5, -2.5, 100)
    ax.plot(uu, EQ.theta_B(uu) @ xi, color=ACCENT, lw=2.6, label='g(u) from weak SINDy')
    ax.set(xlabel=r'$u = \log_{10}K - a(f)$', ylabel=r'rate  $g(u)$', ylim=(-20, 1))
    ax.annotate('bottleneck', xy=(-4.55, -2.1), xytext=(-4.2, -9), fontsize=13,
                arrowprops=dict(arrowstyle='->', color=INK, lw=1))
    ax.legend(loc='lower right', fontsize=12)
    fig.tight_layout()
    save(fig, "g_of_u.png")
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for f in (1.0, 1.25, 1.45, 1.6, 1.65):
        t, y = P[f"t_{f}"], P[f"y_{f}"]
        ax.plot(t, y, color=fcol(f), lw=3.2, alpha=0.45)
        ax.plot(t, P[f"a_{f}_ODE-B"], '--', color=ACCENT, lw=1.5)
    ax.plot([], [], color=INK, lw=3.2, alpha=0.45, label='simulation (left out of fit)')
    ax.plot([], [], '--', color=ACCENT, lw=1.5, label='equation prediction')
    ax.set(xscale='log', xlim=(10, 3e4), ylim=(-21, -1), xlabel='time  $t$', ylabel=r'$\log_{10} K$')
    ax.legend(loc='lower left')
    fig.tight_layout()
    save(fig, "ode_lofo.png")
    L = E['model_L']
    s_sim = EQ.langevin_arrest_times(xi, np.array(L['log_D2_poly']), float(np.median(
        [np.interp(EQ.T0, D[f]['t'], D[f]['y_s']) - (E['model_B']['a_linear'][0] * f + E['model_B']['a_linear'][1])
         for f in [g for g in sorted(D) if 1.0 <= g <= 1.65]])), E['u_on'], n_traj=2000)
    s_sim = s_sim[np.isfinite(s_sim)]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.hist(s_sim, bins=40, density=True, color=LIGHT, label='Langevin model (2000 runs)')
    for s in L['s_arrest_obs']:
        ax.axvline(s, color=ACCENT, lw=1.5, ymax=0.3)
    ax.plot([], [], color=ACCENT, lw=1.5, label='observed (11 forces)')
    ax.set(xlabel=r'rescaled arrest time  $t_{arr}/\tau(f)$', ylabel='density', xlim=(0, 3.5))
    ax.legend()
    fig.tight_layout()
    save(fig, "langevin.png")


def onestep(OSC):
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.6), sharey=True)
    for a, f in zip(ax, (1.25, 1.8)):
        t, truth, tf, free = OSC[f]
        a.plot(t, truth, color=LIGHT, lw=4, label='simulation')
        a.plot(t, tf, '--', color=INK, lw=1.3, label='one-step-ahead')
        a.plot(t, np.clip(free, -30, 5), color=ACCENT, lw=1.8, label='free-running')
        a.set(title=f'f = {f}', xlabel='time  $t$', ylim=(-28, 1))
        a.tick_params(labelsize=12)
        a.title.set_size(14)
    ax[0].set_ylabel(r'$\log_{10} K$')
    ax[0].legend(loc='lower left', fontsize=11)
    fig.tight_layout()
    save(fig, "onestep_vs_free.png")


def forecast_figs(Fr, P):
    fa = Fr['protocol_a']
    forces = sorted(fa, key=float)
    style = {'NeighbourBaseline': ('x', MID, 'neighbour baseline'), 'HDMD-param': ('x', '#9bb7c4', 'parametric HDMD'),
             'DeepKoopman': ('x', '#a7a7a7', 'deep Koopman AE'), 'ODE-B': ('o', INK, 'discovered equation'),
             'MasterCurve': ('o', ACCENT, 'master-curve Koopman')}
    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    top = 1.2e5
    lim = (1.2e3, 3.2e4)
    ax.fill_between(lim, [l / 1.25 for l in lim], [l * 1.25 for l in lim], color='#efefef', label='± 25 %')
    ax.plot(lim, lim, color=INK, lw=0.8)
    for m, (mk, col, lab) in style.items():
        tt, tp = [], []
        for k in forces:
            r = fa[k][m]
            if r['t_arr_true'] is None or not np.isfinite(float(r['t_arr_true'])):
                continue
            tt.append(float(r['t_arr_true']))
            v = r['t_arr_pred']
            tp.append(top if v is None or not np.isfinite(float(v)) else float(v))
        ax.loglog(tt, tp, mk, color=col, ms=9 if m == 'MasterCurve' else 7, mew=2, label=lab,
                  mfc='none' if m == 'ODE-B' else col, zorder=5 if m == 'MasterCurve' else 3)
    ax.axhline(top, color=MID, lw=0.8, ls=':')
    ax.text(1.3e3, top * 1.15, 'never arrests', fontsize=12, color=MID)
    ax.set(xlim=lim, ylim=(4e2, 2.2e5), xlabel='true arrest time', ylabel='predicted arrest time')
    ax.legend(loc='lower right', fontsize=12)
    fig.tight_layout()
    save(fig, "parity.png")
    f, p = 1.4, '0.25'
    t, y = P[f"t_{f}"], P[f"y_{f}"]
    ta = float(fa[str(f)]['MasterCurve']['t_arr_true'])
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.axvspan(0, float(p) * ta, color='#efefef')
    ax.text(14, -8.5, 'given:\nfirst 25 %', fontsize=14, color=MID)
    ax.plot(t, y, color=LIGHT, lw=4.5, label='simulation (never seen)')
    for m, col, ls, lab in (('HDMD-prefix', '#9bb7c4', '--', 'HDMD on prefix'), ('DeepKoopman', MID, '-.', 'deep Koopman AE'),
                            ('MasterCurve', ACCENT, '-', 'master-curve Koopman')):
        ax.plot(t, P[f"b_{f}_{p}_{m}"], ls, color=col, lw=2 if m == 'MasterCurve' else 1.5, label=lab)
    ax.set(xscale='log', xlim=(10, 3e4), ylim=(-21, -1), xlabel='time  $t$   (f = 1.4)', ylabel=r'$\log_{10} K$')
    ax.legend(loc='lower left', fontsize=12, bbox_to_anchor=(0.0, 0.02))
    fig.tight_layout()
    save(fig, "prefix_example.png")


def method_backups(V):
    v6 = V['V6']
    amps = sorted(float(k) for k in v6)
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.plot(amps, [v6[str(a)]['median_max_rel_coef_err'] for a in amps], 'o-', color=INK, ms=7, label='coefficient error')
    ax2 = ax.twinx()
    ax2.plot(amps, [v6[str(a)]['correct_terms_fraction'] for a in amps], 's--', color=ACCENT, ms=7)
    ax2.set_ylabel('correct terms found', color=ACCENT)
    ax2.set_ylim(0, 1.08)
    ax2.spines['right'].set_visible(True)
    ax.axvline(0.1, color=MID, ls=':', lw=1)
    ax.text(0.105, 0.075, 'our data', fontsize=12, color=MID)
    ax.set_xscale('log')
    ax.set_xticks(amps, labels=[f'{a:g}' for a in amps])
    ax.minorticks_off()
    ax.set(xlabel='noise level (decades)', ylabel='coefficient error')
    fig.tight_layout()
    save(fig, "sindy_noise.png")


def main():
    D = load_clean()
    S, E, V, Fr = load("scaling_results.json"), load("equation_results.json"), load("validation_results.json"), load(
        "forecast_results.json")
    P = dict(np.load(os.path.join(PROC, "forecast_predictions.npz")))
    title_strip(D)
    data_overview(D)
    anatomy(D)
    arrest_times(D)
    collapse(D, S)
    collapse_tests(D, S, V)
    critical(S)
    havok_figs(D, V)
    lorenz_and_signal(D)
    equation_figs(D, E, V, P)
    forecast_figs(Fr, P)
    method_backups(V)
    import koopman_forecast as KF
    KF.QUICK = False
    _, curves = KF.original_model_check(D, forces=(1.25, 1.8))
    onestep(curves)


if __name__ == "__main__":
    main()
