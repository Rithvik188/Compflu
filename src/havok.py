"""
HAVOK: Hankel Alternative View Of Koopman  (Brunton et al., Nat. Commun. 8, 19, 2017)

Idea in one paragraph
    Stack time-delayed copies of a single measured signal into a Hankel matrix, take its
    SVD, and use the leading right-singular vectors v_1..v_r as new coordinates. Fit a
    LINEAR model for the first r-1 coordinates and treat the last one, v_r, as an
    external input:
            d/dt v_{1..r-1} = A v_{1..r-1} + B v_r
    For chaotic / intermittent systems v_r is quiet most of the time and bursts exactly
    when the dynamics does something the linear model cannot (lobe switching in Lorenz).
    Here we use it on log10 K(t) of the active glass to detect intermittent
    collapse events (cage breaking / partial arrest) and to measure how fluctuations
    around the master curve relax for every active force f.

Causality note: column j of the Hankel matrix contains y[j], ..., y[j+q-1]. Any
quantity derived from it is time-stamped at the END of its window, t[j+q-1], so a
'forcing precedes event' statement never uses information from the future.

Run from the project root:  python src/havok.py
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import lsim
from scipy.stats import kurtosis
from data_loader import load_clean

ROOT = os.path.join(os.path.dirname(__file__), "..")
DOCS = os.path.join(ROOT, "docs")
PROC = os.path.join(ROOT, "data", "processed")


# ---------------------------------------------------------------------------
# Core HAVOK
# ---------------------------------------------------------------------------
def hankel(x, q):
    n = len(x) - q + 1
    return np.array([x[i:i + n] for i in range(q)])


def derivative_4th(V, dt):
    """4th-order central difference along axis 0; returns dV for rows 2..-3."""
    return (-V[4:] + 8 * V[3:-1] - 8 * V[1:-3] + V[:-4]) / (12 * dt)


class HAVOK:
    def __init__(self, q, r):
        self.q, self.r = q, r

    def fit(self, x, dt):
        self.dt = dt
        self.mean = x.mean()
        H = hankel(x - self.mean, self.q)
        U, S, Vt = np.linalg.svd(H, full_matrices=False)
        self.U, self.S = U[:, :self.r], S[:self.r]
        self.S_all = S
        V = Vt[:self.r].T                       # (n_cols, r)
        dV = derivative_4th(V, dt)
        Vc = V[2:-2]
        # least squares  dV[:, :r-1] = Vc @ Xi
        Xi, *_ = np.linalg.lstsq(Vc, dV[:, :self.r - 1], rcond=None)
        Xi = Xi.T                                # (r-1, r)
        self.A = Xi[:, :self.r - 1]
        self.B = Xi[:, self.r - 1:]
        self.V = V
        self.forcing = V[:, -1]
        return self

    def simulate(self, v0=None, forcing=None):
        """Integrate the linear model driven by the (measured) forcing."""
        f_ = self.forcing if forcing is None else forcing
        n = len(f_)
        tt = np.arange(n) * self.dt
        v0 = self.V[0, :self.r - 1] if v0 is None else v0
        C = np.eye(self.r - 1)
        D = np.zeros((self.r - 1, 1))
        _, y, _ = lsim((self.A, self.B, C, D), f_, tt, X0=v0)
        return y

    def reconstruct(self, Vsim):
        """First Hankel row (the signal itself) from the model coordinates."""
        return (self.U[0, :self.r - 1] * self.S[:self.r - 1]) @ Vsim.T + self.mean

    def eig_continuous(self):
        return np.linalg.eigvals(self.A)


def active_forcing(forcing, k=2.5):
    """Indicator of 'active' forcing: |v_r| > k * std(v_r)."""
    return np.abs(forcing - forcing.mean()) > k * forcing.std()


def select_rank(x, dt, q, ranks, horizon):
    """Pick r: prefer (marginally) stable linear parts, i.e. growth slower than the
    record length; among those take the best forced reconstruction over `horizon`
    samples. If no rank is stable (short, noisy records) fall back to the best finite
    reconstruction and flag it. Returns (r, rmse, model, stable_flag)."""
    tol = 1.0 / (len(x) * dt)
    cands = []
    for r in ranks:
        m = HAVOK(q, r).fit(x, dt)
        n = min(horizon, len(m.forcing))
        with np.errstate(all='ignore'):
            xr = m.reconstruct(m.simulate(forcing=m.forcing[:n]))
            err = np.sqrt(np.mean((xr - x[:n]) ** 2))
        if np.isfinite(err):
            cands.append((np.max(m.eig_continuous().real) < tol, err, r, m))
    stable = [c for c in cands if c[0]]
    pool = stable if stable else cands
    flag, err, r, m = min(pool, key=lambda c: c[1])
    return r, err, m, bool(flag)


def precursor_stats(act, events, lookback, rng=None, n_null=2000):
    """Fraction of events with active forcing in the `lookback` samples before them,
    versus the same fraction for random times (chance level)."""
    rng = np.random.default_rng(0) if rng is None else rng
    n = len(act)
    ev = [e for e in events if e >= lookback]
    if not ev:
        return dict(n_events=0, hit=np.nan, chance=np.nan)
    hit = np.mean([act[e - lookback:e + 1].any() for e in ev])
    rnd = rng.integers(lookback, n, n_null)
    chance = np.mean([act[e - lookback:e + 1].any() for e in rnd])
    return dict(n_events=len(ev), hit=float(hit), chance=float(chance))


# ---------------------------------------------------------------------------
# Sanity check on Lorenz (the HAVOK paper's example)
# ---------------------------------------------------------------------------
def lorenz_check(plot_path=None):
    from scipy.integrate import solve_ivp
    dt = 0.001
    t = np.arange(0, 200, dt)
    sol = solve_ivp(lambda t, s: [10 * (s[1] - s[0]), s[0] * (28 - s[2]) - s[1], s[0] * s[1] - 8 / 3 * s[2]],
                    (0, t[-1]), [-8, 8, 27], t_eval=t, rtol=1e-10, atol=1e-10)
    x = sol.y[0]
    q = 100
    n_sim = int(20 / dt)
    r, err, m, _ = select_rank(x, dt, q, range(7, 16), n_sim)
    t_end = t[q - 1:]                       # causal time stamps
    act = active_forcing(m.forcing)
    # lobe switches: sign changes of x, detected on the causal time axis
    xs = x[q - 1:]
    sw = np.where(np.diff(np.sign(xs)) != 0)[0]
    ps = precursor_stats(act, sw, int(0.5 / dt))     # look back 0.5 time units
    res = dict(r=r, n_switches=ps['n_events'], frac_switches_preceded_by_forcing=ps['hit'],
               chance_level=ps['chance'], frac_time_forcing_active=float(act.mean()),
               forcing_excess_kurtosis=float(kurtosis(m.forcing)),
               recon_rel_rmse_20tu=float(err / x.std()))
    if plot_path:
        fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
        sl = slice(0, int(60 / dt))
        ax[0].plot(t_end[sl], xs[sl], 'k', lw=0.8)
        ax[0].set_ylabel('x (Lorenz)')
        ax[1].plot(t_end[sl], m.forcing[sl], 'k', lw=0.8)
        ax[1].plot(t_end[sl][act[sl]], m.forcing[sl][act[sl]], 'r.', ms=2)
        ax[1].set(ylabel='$v_r$ (forcing)', xlabel='t')
        ax[0].set_title('HAVOK sanity check on Lorenz: forcing (red = active) precedes lobe switches')
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
    return res


# ---------------------------------------------------------------------------
# Active glass
# ---------------------------------------------------------------------------
Q = 40                 # delay window: 40 samples = 20 time units (~4 correlation times)
RANKS = range(3, 16)
LOOKBACK = 40          # precursor look-back: 20 time units
CLIFF_LEVEL = -6.5     # smoothed log10 K below this = final collapse has started
T_TRANSIENT = 20.0
CLIFF_GAP = 40         # samples (20 t.u.) between the end of the warning window and the collapse


def glass_segment(d):
    """Fluctuations of log10 K around its slow linear trend, from t=20 up to the start
    of the final collapse (or the end of the run if the force never arrests)."""
    t, y = d['t'], d['y']
    j0 = np.searchsorted(t, T_TRANSIENT)
    if np.isfinite(d['t_arr']):
        below = np.where(d['y_s'][:d['i_arr']] < CLIFF_LEVEL)[0]
        j_cliff = below[0] if len(below) else d['i_arr']
        j_fit = int(0.85 * d['i_arr'])
    else:
        j_cliff = len(t)
        j_fit = len(t)
    trend = np.polyfit(t[j0:j_fit], y[j0:j_fit], 1)
    seg = slice(j0, j_cliff)
    dy = y[seg] - np.polyval(trend, t[seg])
    return t[seg], dy, trend, j_cliff - j0


def dip_events(dy, k=3.0, smooth=5, min_sep=40):
    """Onsets of strong downward excursions (K collapses that later recover)."""
    from scipy.ndimage import uniform_filter1d
    z = uniform_filter1d(dy, smooth)
    sig = 1.4826 * np.median(np.abs(z - np.median(z)))
    below = z < np.median(z) - k * sig
    on = np.where(below[1:] & ~below[:-1])[0] + 1
    ev, last = [], -10 ** 9
    for e in on:
        if e - last >= min_sep:
            ev.append(e)
            last = e
    return np.array(ev, dtype=int)


def analyse_force(d, f):
    t, dy, trend, n_pre = glass_segment(d)
    dt = d['dt']
    horizon = min(2000, len(dy) - Q)
    r, err, m, stable = select_rank(dy, dt, Q, RANKS, horizon)
    act = active_forcing(m.forcing)
    # causal alignment: forcing column j  <->  sample j+Q-1 of dy
    ev = dip_events(dy)
    ev_c = ev - (Q - 1)
    ev_c = ev_c[ev_c >= 0]
    ps = precursor_stats(act, ev_c, LOOKBACK)
    eig = m.eig_continuous()
    slow = eig[np.argmax(eig.real)]              # least damped (record-scale drift) mode
    out = dict(f=f, n=len(dy), r=r, stable=stable, recon_rmse=float(err), dy_std=float(dy.std()),
               dy_excess_kurtosis=float(kurtosis(dy)),
               forcing_excess_kurtosis=float(kurtosis(m.forcing)),
               frac_active=float(act.mean()),
               n_dips=int(ps['n_events']), dip_precursor_hit=ps['hit'], dip_precursor_chance=ps['chance'],
               slow_mode_rate=float(slow.real),
               slow_mode_freq=float(abs(slow.imag)),
               trend_slope_dec_per_t=float(trend[0]),
               eigs_real=eig.real.tolist(), eigs_imag=eig.imag.tolist())
    # forcing just before the final collapse: project windows that end at the cliff
    # onset onto the SAME SVD basis (v = U^T h / S), so nothing is refitted
    if np.isfinite(d['t_arr']):
        tc, yc = d['t'], d['y']
        j0 = np.searchsorted(tc, T_TRANSIENT)
        below = np.where(d['y_s'][:d['i_arr']] < CLIFF_LEVEL)[0]
        jc = below[0] if len(below) else d['i_arr']
        jstop = min(jc + 5, d['i_arr'])
        dyc = yc[j0:jstop] - np.polyval(trend, tc[j0:jstop])
        H = hankel(dyc - m.mean, Q)
        Vproj = (m.U.T @ H).T / m.S
        fc_ = Vproj[:, -1]
        thr = 2.5 * m.forcing.std()
        # windows ending between 120 and 20 time units BEFORE the smoothed curve crosses
        # CLIFF_LEVEL (the 20 t.u. gap keeps the collapse itself out of every window)
        n_c = jc - j0 - (Q - 1) - CLIFF_GAP
        win = np.abs(fc_[max(0, n_c - 200):n_c + 1] - m.forcing.mean()) > thr
        out['cliff_forcing_active_frac'] = float(win.mean()) if len(win) else np.nan
        out['cliff_enrichment'] = out['cliff_forcing_active_frac'] / max(out['frac_active'], 1e-9)
        # local baseline: the 500 t.u. just before the warning window. Fluctuations grow
        # steadily as K approaches the arrest level, so this separates a *specific*
        # pre-arrest trigger from the general rise in intermittency.
        lb = np.abs(fc_[max(0, n_c - 1200):max(0, n_c - 200)] - m.forcing.mean()) > thr
        out['cliff_local_baseline_frac'] = float(lb.mean()) if len(lb) else np.nan
        # how intermittency grows along the trajectory: active fraction in 5 equal parts
        out['active_frac_by_fifth'] = [float(c.mean()) for c in np.array_split(act, 5)]
    return out, m, (t, dy, ev, act)


def pydmd_crosscheck(dy, dt, r):
    """Least-damped eigenvalue of PyDMD's HAVOK A matrix (same q, r) for comparison."""
    try:
        from pydmd import HAVOK as PHAVOK
        tt = np.arange(len(dy)) * dt
        h = PHAVOK(svd_rank=r, delays=Q).fit(dy - dy.mean(), tt)
        ev = np.linalg.eigvals(h.A)
        return complex(ev[np.argmax(ev.real)])
    except Exception as e:                       # API differences between versions
        return repr(e)


def plot_force(f, m, data, res, path):
    t, dy, ev, act = data
    tc = t[Q - 1:]
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(3, 3)
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.semilogy(m.S_all[:20] / m.S_all.sum(), 'o-')
    ax0.axvline(m.r - 1, color='r', ls='--', label=f'r={m.r}')
    ax0.set(title='singular values', xlabel='index')
    ax0.legend()
    ax1 = fig.add_subplot(gs[0, 1:])
    ax1.plot(t, dy, 'k', lw=0.4)
    ax1.plot(t[ev], dy[ev], 'rv', ms=5, label='dip events')
    ax1.set(title=f'f={f}: fluctuation of $\\log_{{10}}K$ around slow trend', xlabel='t')
    ax1.legend()
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(tc, m.forcing, 'k', lw=0.4)
    ax2.plot(tc[act], m.forcing[act], 'r.', ms=2, label='active forcing')
    ax2.set(title=f"HAVOK forcing $v_r$ (excess kurtosis {res['forcing_excess_kurtosis']:.1f})",
            xlabel='t (end of delay window)')
    ax2.legend()
    ax3 = fig.add_subplot(gs[2, :2])
    n = min(1000, len(m.forcing))
    xr = m.reconstruct(m.simulate(forcing=m.forcing[:n]))
    ax3.plot(t[:n], dy[:n], 'k', lw=0.8, label='data')
    ax3.plot(t[:n], xr, 'r--', lw=1, label='HAVOK (linear + forcing)')
    ax3.set(title=f"reconstruction (rmse {res['recon_rmse']:.3f} decades)", xlabel='t')
    ax3.legend()
    ax4 = fig.add_subplot(gs[2, 2])
    ax4.scatter(m.V[:, 0], m.V[:, 1], c=np.abs(m.forcing), s=1, cmap='inferno')
    ax4.set(title='embedding $v_1$-$v_2$ (colour=|forcing|)', xlabel='$v_1$', ylabel='$v_2$')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_summary(rows, path):
    f = np.array([r['f'] for r in rows])
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax = ax.ravel()
    ax[0].plot(f, [r['forcing_excess_kurtosis'] for r in rows], 'o-', label='HAVOK forcing $v_r$')
    ax[0].plot(f, [r['dy_excess_kurtosis'] for r in rows], 's--', label='raw fluctuations')
    ax[0].axhline(0, color='k', lw=0.8)
    ax[0].set(yscale='symlog', xlabel='$f$', ylabel='excess kurtosis', title='Intermittency (0 = Gaussian)')
    ax[0].legend()
    hit = np.array([r['dip_precursor_hit'] for r in rows], float)
    ch = np.array([r['dip_precursor_chance'] for r in rows], float)
    ax[1].plot(f, hit, 'o-', label='dips preceded by active forcing')
    ax[1].plot(f, ch, 's--', label='chance level (random times)')
    ax[1].set(xlabel='$f$', ylabel='fraction', title='Forcing as early warning of collapse events (20 t.u.)')
    ax[1].legend()
    jam = [r for r in rows if 'cliff_forcing_active_frac' in r]
    fj = [r['f'] for r in jam]
    ax[2].plot(fj, [r['cliff_forcing_active_frac'] for r in jam], 'o-', color='crimson',
               label='window 120-20 t.u. before final collapse')
    ax[2].plot(fj, [r['cliff_local_baseline_frac'] for r in jam], '^-', color='darkorange',
               label='preceding 500 t.u. (local baseline)')
    ax[2].plot(fj, [r['frac_active'] for r in jam], 's--', color='gray', label='whole trajectory (global baseline)')
    ax[2].set(xlabel='$f$', ylabel='fraction of time forcing is active',
              title='HAVOK forcing as early warning of arrest')
    ax[2].legend()
    ax[3].semilogy(f, [r['dy_std'] for r in rows], 'o-')
    ax[3].set(xlabel='$f$', ylabel=r'std of $\delta\log_{10}K$', title='Fluctuation amplitude')
    for a in ax:
        a.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def run():
    lor = lorenz_check(os.path.join(DOCS, "havok_lorenz_check.png"))
    print("Lorenz sanity check:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in lor.items()})
    D = load_clean()
    forces = [f for f in sorted(D) if f >= 1.0]
    rows = []
    print("\n f     r   recon  kurt(dy) kurt(v_r) active  dips  dip hit/chance  pre-arrest forcing (x baseline)")
    for f in forces:
        res, m, data = analyse_force(D[f], f)
        rows.append(res)
        cf = res.get('cliff_forcing_active_frac', np.nan)
        ce = res.get('cliff_enrichment', np.nan)
        print(f"{f:5.2f} {res['r']:3d} {res['recon_rmse']:7.3f} {res['dy_excess_kurtosis']:8.1f} "
              f"{res['forcing_excess_kurtosis']:8.1f} {res['frac_active']:6.3f} {res['n_dips']:5d} "
              f"{res['dip_precursor_hit']:7.2f}/{res['dip_precursor_chance']:.2f}      {cf:.2f} (x{ce:.1f}; "
              f"local {res.get('cliff_local_baseline_frac', np.nan):.2f})  by fifth "
              f"{np.round(res.get('active_frac_by_fifth', [np.nan]), 3).tolist()}"
              + ("" if res['stable'] else "  [marginal A]"))
        if f in (1.0, 1.4, 1.65, 1.8, 2.0):
            plot_force(f, m, data, res, os.path.join(DOCS, f"havok_f{f}.png"))
        if f == 1.4:
            res['pydmd_least_damped_eig'] = str(pydmd_crosscheck(data[1], D[f]['dt'], res['r']))
            k = int(np.argmax(res['eigs_real']))
            print(f"   PyDMD cross-check (f=1.4): ours {complex(res['eigs_real'][k], res['eigs_imag'][k]):.4f}"
                  f"  PyDMD {res['pydmd_least_damped_eig']}")
    plot_summary(rows, os.path.join(DOCS, "havok_summary.png"))
    with open(os.path.join(PROC, "havok_results.json"), "w") as fh:
        json.dump(dict(lorenz=lor, q=Q, lookback=LOOKBACK, forces=rows), fh, indent=2, default=float)
    print("\nSaved docs/havok_f*.png, docs/havok_summary.png, docs/havok_lorenz_check.png, "
          "data/processed/havok_results.json")
    return rows


if __name__ == "__main__":
    if os.environ.get("HAVOK_LORENZ_ONLY"):
        print(lorenz_check(os.path.join(DOCS, "havok_lorenz_check.png")))
    else:
        run()
