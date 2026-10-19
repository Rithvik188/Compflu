"""
Governing equations for the kinetic energy K(t; f) of the active glass.

Variables:  y = log10 K,  f = active force.

Models
  A  General parametric SINDy, weak form:  dy/dt = sum_j xi_j * y^i f^k
     (library: monomials up to y^2 f^3), fitted on the slow stage of all forces.
  B  Master-curve ODE: the collapse  y = a(f) + G(t/tau(f))  implies the autonomous law
            tau(f) dy/dt = g(u),   u = y - a(f)
     with ONE universal function g. g is found with weak SINDy (polynomial in u), and
     tau(f), a(f) come from the scaling analysis -> one closed equation for all f.
  H  Handoff hypothesis  dK/dt = -A K + B f K^n   <=>   dy/dt = (-A + B f 10^{(n-1)y}) / ln10
     (scan n, fit A, B).
  L  Stochastic (Langevin) version: drift and diffusion of y estimated with
     Kramers-Moyal conditional moments; the arrest-time distribution it implies is
     compared with the observed arrest times.
  P  Cross-check with PySINDy (strong form, smoothed finite differences, f as a
     control input).

Arrest: below the force-dependent level y_on(f) the slow law stops and K decays
with the universal rate Gamma (decades/time) measured in scaling_analysis.py.

Weak SINDy (Messenger & Bortz 2021): multiply dy/dt = Theta(y) xi by smooth bump
test functions phi_k(t) and integrate by parts:  -int phi_k' y dt = int phi_k Theta xi dt.
No numerical derivative of the noisy data is ever taken.

Run from the project root:  python src/equation_discovery.py
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from data_loader import load_clean

ROOT = os.path.join(os.path.dirname(__file__), "..")
DOCS = os.path.join(ROOT, "docs")
PROC = os.path.join(ROOT, "data", "processed")
T0 = 20.0                 # start after the initial transient
LN10 = np.log(10.0)


# ---------------------------------------------------------------------------
# Weak-form SINDy engine
# ---------------------------------------------------------------------------
def weak_rows(t, y, theta_fn, width=200, stride=50, p=4):
    """Return (b, G) with b_k = -int phi_k' y dt / int phi_k, G_kj = int phi_k theta_j / int phi_k.
    Each row is therefore a smooth window-average of dy/dt (b) and of the library (G)."""
    Th = theta_fn(y)                                  # (n, m)
    rows_b, rows_G = [], []
    for s in range(0, len(t) - width, stride):
        tt = t[s:s + width]
        a, b_ = tt[0], tt[-1]
        z = (tt - a) * (b_ - tt)
        phi = z ** p
        dphi = p * z ** (p - 1) * (b_ + a - 2 * tt)
        norm = np.trapezoid(phi, tt)
        rows_b.append(-np.trapezoid(dphi * y[s:s + width], tt) / norm)
        rows_G.append(np.trapezoid(phi[:, None] * Th[s:s + width], tt, axis=0) / norm)
    return np.array(rows_b), np.array(rows_G)


def stlsq(G, b, lam, w=None, n_iter=20):
    """Sequentially thresholded least squares on column-normalised G (row weights w)."""
    w = np.ones(len(b)) if w is None else w
    Gw, bw = G * w[:, None], b * w
    scale = np.linalg.norm(Gw, axis=0)
    scale[scale == 0] = 1
    Gn = Gw / scale
    xi = np.linalg.lstsq(Gn, bw, rcond=None)[0]
    for _ in range(n_iter):
        small = np.abs(xi) < lam * np.max(np.abs(xi))
        xi[small] = 0
        big = ~small
        if big.sum() == 0:
            break
        xi[big] = np.linalg.lstsq(Gn[:, big], bw, rcond=None)[0]
    return xi / scale


def select_lambda(G, b, w, lams, overlap=4):
    """Pick the sparsity threshold by BIC on the weak-form residual. Neighbouring test
    windows overlap (width/stride = 4), so only n/overlap rows are independent."""
    best = None
    path = []
    for lam in lams:
        xi = stlsq(G, b, lam, w)
        k = int(np.sum(xi != 0))
        rss = np.sum((w * (G @ xi - b)) ** 2)
        n = len(b) / overlap
        aic_ = n * np.log(rss / n) + np.log(n) * k
        path.append((lam, k, aic_))
        if best is None or aic_ < best[2]:
            best = (lam, xi, aic_)
    return best[1], best[0], path


def fmt_eq(xi, names, lhs):
    terms = [f"{c:+.4g}·{n}" if n != '1' else f"{c:+.4g}" for c, n in zip(xi, names) if c != 0]
    return f"{lhs} = " + " ".join(terms) if terms else f"{lhs} = 0"


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def load_scaling():
    with open(os.path.join(PROC, "scaling_results.json")) as fh:
        return json.load(fh)


def onset_level(d):
    """log10 K at which the final collapse starts: smoothed value 40 t.u. before arrest."""
    i = np.searchsorted(d['t'], d['t_arr'] - 40.0)
    return float(d['y_s'][i])


def slow_stage(d):
    """Samples from T0 to 40 t.u. before arrest (whole run if no arrest)."""
    j0 = np.searchsorted(d['t'], T0)
    j1 = np.searchsorted(d['t'], d['t_arr'] - 40.0) if np.isfinite(d['t_arr']) else len(d['t'])
    return d['t'][j0:j1], d['y'][j0:j1]


# ---------------------------------------------------------------------------
# Model A: general parametric library
# ---------------------------------------------------------------------------
Y_C, F_C = -3.5, 1.4                                            # centring (conditioning)
A_POW = [(i, k) for i in range(3) for k in range(3)]           # Y^i F^k, Y=y+3.5, F=f-1.4
A_NAMES = [("1" if i == 0 and k == 0 else
            ("Y" if i == 1 else f"Y^{i}" if i else "") + ("" if k == 0 else ("*" if i else "") + ("F" if k == 1 else f"F^{k}")))
           for i, k in A_POW]


def theta_A(f):
    F = f - F_C
    return lambda y: np.stack([(y - Y_C) ** i * F ** k for i, k in A_POW], axis=1)


def build_weak(D, forces, theta_maker):
    B, G, W = [], [], []
    for f in forces:
        t, y = slow_stage(D[f])
        b, g = weak_rows(t, y, theta_maker(f))
        B.append(b)
        G.append(g)
        W.append(np.full(len(b), 1.0 / np.sqrt(len(b))))     # equal weight per force
    return np.concatenate(B), np.vstack(G), np.concatenate(W)


def fit_model_A(D, forces):
    b, G, w = build_weak(D, forces, theta_A)
    xi, lam, path = select_lambda(G, b, w, np.logspace(-4, -0.5, 30))
    return xi, lam


def rhs_A(xi, f):
    th = theta_A(f)
    return lambda t, y: th(np.atleast_1d(y)) @ xi


# ---------------------------------------------------------------------------
# Model B: universal master-curve ODE  tau(f) dy/dt = g(y - a(f))
# ---------------------------------------------------------------------------
B_DEG = 3
B_NAMES = ["1", "u", "u^2", "u^3"]


def theta_B(u):
    return np.stack([u ** i for i in range(B_DEG + 1)], axis=1)


def fit_model_B(D, forces, a_of, tau_of):
    Bs, Gs, Ws = [], [], []
    for f in forces:
        t, y = slow_stage(D[f])
        s = t / tau_of(f)                    # rescaled time
        u = y - a_of(f)
        b, g = weak_rows(s, u, theta_B)
        Bs.append(b)
        Gs.append(g)
        Ws.append(np.full(len(b), 1.0 / np.sqrt(max(len(b), 1))))
    b, G, w = np.concatenate(Bs), np.vstack(Gs), np.concatenate(Ws)
    xi, lam, _ = select_lambda(G, b, w, np.logspace(-4, -0.5, 30))
    return xi


def closure(S, forces_fit):
    """tau(f) = A (fc - f)^-gamma and a(f) linear, from the master-curve collapse."""
    mc = S['master_curve']
    fm = np.array(mc['forces'])
    sel = np.isin(fm, forces_fit)
    fm, tau, a = fm[sel], np.array(mc['tau'])[sel], np.array(mc['a'])[sel]
    best = None
    for fc in np.linspace(fm.max() + 5e-3, 3.0, 600):
        X = np.vstack([np.ones_like(fm), -np.log(fc - fm)]).T
        c, *_ = np.linalg.lstsq(X, np.log(tau), rcond=None)
        sse = np.sum((np.log(tau) - X @ c) ** 2)
        if best is None or sse < best[0]:
            best = (sse, fc, np.exp(c[0]), c[1])
    _, fc, A, gam = best
    al = np.polyfit(fm, a, 1)
    return dict(fc=fc, A=A, gamma=gam, a_lin=al.tolist(),
                tau_of=lambda f: A * (fc - f) ** (-gam), a_of=lambda f: np.polyval(al, f))


# ---------------------------------------------------------------------------
# Model H: handoff hypothesis  dK/dt = -A K + B f K^n
# ---------------------------------------------------------------------------
def fit_model_H(D, forces, ns=np.linspace(0.0, 3.0, 61)):
    best, scan = None, []
    for n in ns:
        maker = lambda f, n=n: (lambda y: np.stack([np.full_like(y, -1.0 / LN10),
                                                    f * 10 ** ((n - 1) * y) / LN10], axis=1))
        b, G, w = build_weak(D, forces, maker)
        c = np.linalg.lstsq(G * w[:, None], b * w, rcond=None)[0]
        rss = np.sum((w * (G @ c - b)) ** 2)
        scan.append((n, rss, c[0], c[1]))
        if best is None or rss < best[1]:
            best = (n, rss, c[0], c[1])
    return dict(n=best[0], A=best[2], B=best[3], rss=best[1], scan=scan)


# ---------------------------------------------------------------------------
# Integration with the arrest event
# ---------------------------------------------------------------------------
def integrate(rhs, y0, t_end, y_on, gamma_cliff, floor=-20.0):
    """Slow law until y < y_on, then universal collapse dy/dt = -gamma_cliff."""
    ev = lambda t, y: y[0] - y_on
    ev.terminal, ev.direction = True, -1
    sol = solve_ivp(rhs, (T0, t_end), [y0], events=ev, max_step=5.0, rtol=1e-6, atol=1e-8,
                    dense_output=True)
    t_arr = sol.t_events[0][0] if len(sol.t_events[0]) else np.inf
    tg = np.arange(T0, t_end, 0.5)
    yg = np.empty_like(tg)
    pre = tg <= min(t_arr, sol.t[-1])
    yg[pre] = sol.sol(tg[pre])[0]
    if np.isfinite(t_arr):
        yg[~pre] = np.maximum(y_on - gamma_cliff * (tg[~pre] - t_arr), floor)
    else:
        yg[~pre] = np.nan
    return tg, yg, t_arr


def score(d, tg, yg, t_arr_pred):
    """RMSE of log10 K on the slow stage + arrest-time error."""
    t, y = slow_stage(d)
    t_true = d['t_arr'] - 40.0 if np.isfinite(d['t_arr']) else np.inf
    keep = t <= min(t_true, t_arr_pred)            # cliff timing is scored separately
    t = t[keep]
    ys = np.interp(t, d['t'], d['y_s'])
    yp = np.interp(t, tg, yg)
    rmse = float(np.sqrt(np.nanmean((yp - ys) ** 2)))
    rel = (t_arr_pred - t_true) / t_true if np.isfinite(t_true) and np.isfinite(t_arr_pred) else np.nan
    return rmse, float(rel)


# ---------------------------------------------------------------------------
# Model L: Langevin  dy = D1(u) dt + sqrt(2 D2(u)) dW  (u = y - a(f), time in units of tau)
# ---------------------------------------------------------------------------
def kramers_moyal(D, forces, a_of, tau_of, lag=10, bins=np.linspace(-6.0, -2.4, 13)):
    """Conditional moments of the increment of u over `lag` samples, in rescaled time s."""
    U, dU, DS = [], [], []
    for f in forces:
        t, y = slow_stage(D[f])
        u = y - a_of(f)
        s = t / tau_of(f)
        U.append(u[:-lag])
        dU.append(u[lag:] - u[:-lag])
        DS.append(np.full(len(u) - lag, s[lag] - s[0]))
    U, dU, DS = map(np.concatenate, (U, dU, DS))
    centers, d1, d2, cnt = [], [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (U >= lo) & (U < hi)
        if m.sum() < 50:
            continue
        inc = dU[m] / DS[m]
        centers.append(0.5 * (lo + hi))
        d1.append(np.mean(inc))
        d2.append(np.mean((dU[m] - np.mean(dU[m])) ** 2 / (2 * DS[m])))
        cnt.append(int(m.sum()))
    return np.array(centers), np.array(d1), np.array(d2), np.array(cnt)


def langevin_arrest_times(g_xi, d2_fit, u0, u_on, n_traj=400, ds=2e-4, s_max=3.0, seed=0):
    """Simulate du = g(u) ds + sqrt(2 D2(u)) dW in rescaled time; return arrest times s*."""
    rng = np.random.default_rng(seed)
    u = np.full(n_traj, u0)
    s_arr = np.full(n_traj, np.inf)
    alive = np.ones(n_traj, bool)
    for k in range(int(s_max / ds)):
        g = theta_B(u[alive]) @ g_xi
        sig = np.sqrt(2 * np.exp(np.polyval(d2_fit, u[alive])) * ds)
        u[alive] += g * ds + sig * rng.standard_normal(alive.sum())
        hit = alive.copy()
        hit[alive] = u[alive] < u_on
        s_arr[hit] = (k + 1) * ds
        alive &= ~hit
        if not alive.any():
            break
    return s_arr


# ---------------------------------------------------------------------------
# Synthetic check: can weak SINDy recover a known law at this noise level?
# ---------------------------------------------------------------------------
def synthetic_check(D, forces):
    rng = np.random.default_rng(1)
    true = {"1": -0.0005, "F": 0.0011}               # dy/dt = -0.0005 + 0.0011 (f - 1.4)
    Ds = {}
    for f in forces:
        d = D[f]
        t, y = slow_stage(d)
        rate = true["1"] + true["F"] * (f - F_C)
        clean = y[0] + rate * (t - t[0])
        # AR(1) noise with the data's correlation time (~5 t.u.) and amplitude (~0.1 decades)
        e = np.zeros(len(t))
        phi = np.exp(-0.5 / 5.0)
        for i in range(1, len(t)):
            e[i] = phi * e[i - 1] + np.sqrt(1 - phi ** 2) * 0.1 * rng.standard_normal()
        Ds[f] = dict(t=np.concatenate([[0.0], t]), y=np.concatenate([[clean[0]], clean + e]),
                     t_arr=t[-1] + 40.0)
    b, G, w = build_weak(Ds, forces, theta_A)
    xi, lam, _ = select_lambda(G, b, w, np.logspace(-4, -0.5, 30))
    rec = {n: float(c) for n, c in zip(A_NAMES, xi) if c != 0}
    ok = set(rec) == {"1", "F"} and abs(rec["1"] - true["1"]) < 1e-4 and abs(rec["F"] - true["F"]) < 2e-4
    return dict(true={k: v for k, v in true.items() if v}, recovered=rec, passed=bool(ok))


# ---------------------------------------------------------------------------
# PySINDy cross-check (strong form)
# ---------------------------------------------------------------------------
def pysindy_check(D, forces):
    try:
        import pysindy as ps
    except ImportError:
        return "pysindy not installed"
    xs, us, ts = [], [], []
    for f in forces:
        t, y = slow_stage(D[f])
        t, y = t[::10], np.interp(t[::10], D[f]['t'], D[f]['y_s'])   # smoothed, 5 t.u. spacing
        xs.append(y[:, None] - Y_C)
        us.append(np.full((len(y), 1), f - F_C))
        ts.append(t)
    model = ps.SINDy(feature_library=ps.PolynomialLibrary(degree=2),
                     optimizer=ps.STLSQ(threshold=1e-4),
                     differentiation_method=ps.SmoothedFiniteDifference())
    model.fit(xs, t=ts, u=us, feature_names=["Y", "F"])
    return model.equations(precision=5)[0]


# ---------------------------------------------------------------------------
def run():
    D = load_clean()
    S = load_scaling()
    gamma_cliff = -S['cliff_rate']['mean']                       # decades / time
    arrested = [f for f in sorted(D) if f >= 1.0 and np.isfinite(D[f]['t_arr'])]
    forces = arrested + [1.8]
    # collapse onset in master-curve units: u_on = y_on - a(f), universal (median over f)
    cl_all = closure(S, forces)
    y_on = {f: onset_level(D[f]) for f in arrested}
    u_on = float(np.median([y_on[f] - cl_all['a_of'](f) for f in arrested]))
    out = dict(gamma_cliff=gamma_cliff, y_on=y_on, u_on=u_on)
    print(f"Universal collapse rate Gamma = {gamma_cliff:.3f} decades/time "
          f"(= {gamma_cliff * LN10:.3f} per time in ln K)")
    print("Collapse onset level y_on(f):", {f: round(v, 2) for f, v in y_on.items()},
          f"-> universal u_on = y_on - a(f) = {u_on:.2f} (median)")

    # ---- synthetic test ----
    syn = synthetic_check(D, forces)
    out['synthetic_check'] = syn
    print(f"\nSynthetic recovery test: true {syn['true']}  recovered "
          f"{ {k: round(v, 5) for k, v in syn['recovered'].items()} }  passed={syn['passed']}")

    # ---- Model A ----
    xiA, lamA = fit_model_A(D, forces)
    eqA = fmt_eq(xiA, A_NAMES, "dy/dt")
    print("\nModel A (weak SINDy, all forces):\n   " + eqA)
    out['model_A'] = dict(equation=eqA, coefficients=dict(zip(A_NAMES, xiA.tolist())), threshold=lamA)

    # ---- Model B ----
    cl = closure(S, forces)
    xiB = fit_model_B(D, forces, cl['a_of'], cl['tau_of'])
    eqB = fmt_eq(xiB, B_NAMES, "du/ds")
    print(f"\nModel B (universal):  tau(f) dy/dt = g(y - a(f)),  g: {eqB}")
    print(f"   closure: tau(f) = {cl['A']:.4g} ({cl['fc']:.3f} - f)^-{cl['gamma']:.2f},  "
          f"a(f) = {cl['a_lin'][0]:.3f} f + {cl['a_lin'][1]:.3f}")
    out['model_B'] = dict(g_equation=eqB, g_coefficients=dict(zip(B_NAMES, xiB.tolist())),
                          tau_law=dict(A=cl['A'], fc=cl['fc'], gamma=cl['gamma']), a_linear=cl['a_lin'])

    # ---- Model H ----
    H = fit_model_H(D, forces)
    print(f"\nModel H (handoff hypothesis dK/dt = -A K + B f K^n): best n = {H['n']:.2f}, "
          f"A = {H['A']:.4g}, B = {H['B']:.4g}")
    out['model_H'] = {k: v for k, v in H.items() if k != 'scan'}
    out['model_H']['scan'] = [[float(v) for v in row] for row in H['scan']]

    # ---- validation: in-sample and leave-one-force-out for A and B ----
    rhsB = lambda xi, cl, f: (lambda t, y: theta_B(np.atleast_1d(y) - cl['a_of'](f)) @ xi / cl['tau_of'](f))
    rhsH = lambda H, f: (lambda t, y: (-H['A'] + H['B'] * f * 10 ** ((H['n'] - 1) * np.atleast_1d(y))) / LN10)
    val = {}
    print("\nValidation (RMSE of log10 K on slow stage | relative arrest-time error)")
    print("   f     A(in)          A(LOFO)        B(in)          B(LOFO)        H(in)")
    curves = {}
    for f in forces:
        d = D[f]
        y0 = float(np.interp(T0, d['t'], d['y_s']))
        t_end = d['t'][-1]
        others = [g for g in forces if g != f]
        xiA_l, _ = fit_model_A(D, others)
        cl_l = closure(S, others)
        xiB_l = fit_model_B(D, others, cl_l['a_of'], cl_l['tau_of'])
        res = {}
        for name, rhs, a_ in [('A_in', rhs_A(xiA, f), cl['a_of']), ('A_lofo', rhs_A(xiA_l, f), cl_l['a_of']),
                              ('B_in', rhsB(xiB, cl, f), cl['a_of']), ('B_lofo', rhsB(xiB_l, cl_l, f), cl_l['a_of']),
                              ('H_in', rhsH(H, f), cl['a_of'])]:
            with np.errstate(all='ignore'):
                tg, yg, ta = integrate(rhs, y0, t_end, a_(f) + u_on, gamma_cliff)
            res[name] = score(d, tg, yg, ta)
            curves[(f, name)] = (tg, yg)
        val[f] = res
        print(f"  {f:4.2f} " + " ".join(f"{res[k][0]:5.2f}|{res[k][1]:+6.2f}  " for k in
                                        ['A_in', 'A_lofo', 'B_in', 'B_lofo', 'H_in']))
    out['validation'] = {str(f): {k: dict(rmse=v[0], rel_t_arr_err=v[1]) for k, v in r.items()}
                         for f, r in val.items()}

    # ---- Model L: Langevin ----
    uc, d1, d2, cnt = kramers_moyal(D, forces, cl['a_of'], cl['tau_of'])
    d2_fit = np.polyfit(uc, np.log(d2), 2)
    u0 = float(np.median([np.interp(T0, D[f]['t'], D[f]['y_s']) - cl['a_of'](f) for f in arrested]))
    s_sim = langevin_arrest_times(xiB, d2_fit, u0, u_on)
    s_obs = np.array([(D[f]['t_arr'] - 40.0) / cl['tau_of'](f) for f in arrested])
    print(f"\nModel L (Langevin, rescaled time s = t/tau): D2(u) = exp({d2_fit[0]:.3f} u^2 + {d2_fit[1]:.3f} u + {d2_fit[2]:.3f})")
    print(f"   drift from Kramers-Moyal vs g(u): " +
          ", ".join(f"u={u:.1f}: {a:.2f}/{(theta_B(np.array([u])) @ xiB)[0]:.2f}" for u, a in zip(uc, d1)))
    print(f"   arrest time s*: Langevin median {np.median(s_sim[np.isfinite(s_sim)]):.3f} "
          f"(16-84%: {np.percentile(s_sim[np.isfinite(s_sim)], 16):.3f}-{np.percentile(s_sim[np.isfinite(s_sim)], 84):.3f}),"
          f" observed {np.median(s_obs):.3f} ({np.percentile(s_obs, 16):.3f}-{np.percentile(s_obs, 84):.3f})")
    out['model_L'] = dict(u_centers=uc.tolist(), drift=d1.tolist(), diffusion=d2.tolist(), counts=cnt.tolist(),
                          log_D2_poly=d2_fit.tolist(), s_arrest_sim=np.percentile(s_sim[np.isfinite(s_sim)], [16, 50, 84]).tolist(),
                          s_arrest_obs=s_obs.tolist())

    # ---- PySINDy cross-check ----
    eqP = pysindy_check(D, forces)
    print(f"\nPySINDy (strong form, smoothed derivative, f as control; Y=y+3.5, F=f-1.4): dY/dt = {eqP}")
    out['pysindy_equation'] = eqP

    plot_validation(D, forces, curves, val, os.path.join(DOCS, "ode_validation.png"))
    plot_laws(uc, d1, d2, xiB, d2_fit, H, s_sim, s_obs, os.path.join(DOCS, "ode_laws.png"))
    with open(os.path.join(PROC, "equation_results.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=lambda o: float(o) if np.isscalar(o) else str(o))
    write_markdown(out, cl, os.path.join(DOCS, "equations.md"))
    print("\nSaved docs/ode_validation.png, docs/ode_laws.png, docs/equations.md, data/processed/equation_results.json")
    return out


def plot_validation(D, forces, curves, val, path):
    cm = plt.cm.plasma
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
    for f in forces:
        d = D[f]
        c = cm(f / 2.1)
        i = min(d['i_arr'] + 60, len(d['t']))
        for a_, key in zip(ax, ['A_lofo', 'B_lofo']):
            a_.plot(d['t'][:i], d['y_s'][:i], color=c, lw=3, alpha=0.35)
            tg, yg = curves[(f, key)]
            a_.plot(tg, yg, '--', color=c, lw=1.4)
    ax[0].set(title='Model A (general weak SINDy), leave-one-force-out', xlabel='$t$', ylabel=r'$\log_{10}K$',
              xscale='log', ylim=(-21, -1))
    ax[1].set(title=r'Model B (universal $\tau(f)\,\dot y = g(y-a(f))$), leave-one-force-out', xlabel='$t$',
              xscale='log')
    for a_ in ax:
        a_.grid(alpha=0.3)
    ax[0].plot([], [], color='k', lw=3, alpha=0.35, label='data (smoothed)')
    ax[0].plot([], [], 'k--', label='ODE prediction (force never seen in fit)')
    ax[0].legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_laws(uc, d1, d2, xiB, d2_fit, H, s_sim, s_obs, path):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    uu = np.linspace(uc.min(), uc.max(), 100)
    ax[0].plot(uc, d1, 'o', label='Kramers-Moyal drift (data)')
    ax[0].plot(uu, theta_B(uu) @ xiB, '-', label='weak-SINDy $g(u)$')
    ax[0].set(xlabel=r'$u=\log_{10}K-a(f)$', ylabel=r'$du/ds$', title='Universal drift (s = t/τ(f))')
    ax[0].legend()
    ax[1].semilogy(uc, d2, 'o', label='Kramers-Moyal diffusion')
    ax[1].semilogy(uu, np.exp(np.polyval(d2_fit, uu)), '-', label='fit')
    ax[1].set(xlabel='$u$', ylabel='$D_2(u)$', title='Noise grows as K approaches arrest')
    ax[1].legend()
    scan = np.array(H['scan'])
    ax[2].plot(scan[:, 0], scan[:, 1] / scan[:, 1].min(), '-')
    ax[2].axvline(H['n'], color='r', ls='--', label=f"best n = {H['n']:.2f}")
    ax[2].set(xlabel='$n$', ylabel='relative residual', title=r'Hypothesis $\dot K=-AK+BfK^n$', yscale='log')
    ax[2].legend()
    for a_ in ax:
        a_.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def write_markdown(out, cl, path):
    gB = out['model_B']['g_coefficients']
    lines = [
        "# Discovered governing equations (auto-generated by `src/equation_discovery.py`)",
        "",
        "Variables: $y=\\log_{10}K$, $f$ = active force, $u = y - a(f)$, $s = t/\\tau(f)$.",
        "",
        f"**Synthetic recovery test** (known law + AR(1) noise like the data): passed = {out['synthetic_check']['passed']}",
        "",
        "## Model A: general weak SINDy",
        f"`{out['model_A']['equation']}`",
        "",
        "## Model B: universal master-curve ODE",
        "$$\\tau(f)\\,\\frac{dy}{dt} = g\\big(y-a(f)\\big)$$",
        f"`g: {out['model_B']['g_equation']}`",
        "",
        f"with $\\tau(f) = {cl['A']:.4g}\\,({cl['fc']:.3f}-f)^{{-{cl['gamma']:.2f}}}$ and "
        f"$a(f) = {cl['a_lin'][0]:.3f} f {cl['a_lin'][1]:+.3f}$.",
        "",
        f"Arrest: when $u = y - a(f) < u_{{on}} = {out['u_on']:.2f}$ the law switches to "
        f"$dy/dt = -\\Gamma$, $\\Gamma = {out['gamma_cliff']:.3f}$ decades/time (universal).",
        "",
        "## Model H: handoff hypothesis $dK/dt = -AK + BfK^n$",
        f"best $n = {out['model_H']['n']:.2f}$, $A = {out['model_H']['A']:.4g}$, $B = {out['model_H']['B']:.4g}$",
        "",
        "## Model L: Langevin",
        "$du = g(u)\\,ds + \\sqrt{2D_2(u)}\\,dW$, "
        f"$\\ln D_2(u) = {out['model_L']['log_D2_poly'][0]:.3f}u^2 {out['model_L']['log_D2_poly'][1]:+.3f}u "
        f"{out['model_L']['log_D2_poly'][2]:+.3f}$",
        "",
        "## PySINDy cross-check (strong form)",
        f"`dy/dt = {out['pysindy_equation']}`",
    ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    run()
