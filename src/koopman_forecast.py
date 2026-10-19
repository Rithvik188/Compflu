"""
Q2: Can Koopman-based methods predict KE time series they have never seen?

Two protocols, both strictly free-running (no ground truth is fed back after the start):
  (a) UNSEEN FORCE  - leave-one-force-out: the model is built without force f* and must
                      predict the whole log10 K(t; f*) from f* alone (all runs share K(0);
                      the first 100 t.u. are taken as an interpolation of the neighbouring
                      training forces, never from the test trajectory).
  (b) PREFIX        - the first p = 10/25/50 % (of the arrest time) of the unseen
                      trajectory is given; forecast the rest.

Models
  HDMD-param  Parametric Hankel DMD: common SVD basis of all training Hankel matrices,
              one linear (affine) Koopman operator A(f) per training force, A(f*) obtained
              by interpolating the operators in f. Purely data-driven Koopman.
  HDMD-prefix (protocol b) affine HDMD fitted on the prefix alone (= HAVOK with the
              unknown future forcing set to zero).
  DeepKoopman Force-conditioned Koopman autoencoder retrained with a multi-step rollout
              loss (train.train_rollout), predicted by latent rollout z_{k+1} = K(f) z_k.
  MasterCurve Physics-informed: y = a(f) + G(t/tau(f)) with G, tau(f), a(f) re-derived
              WITHOUT the test force (tau = A (f_c - f)^-gamma, a linear in f); after
              x = t/tau = 1 the universal collapse rate. In Koopman language: a linear
              (exponential) Koopman mode whose eigenvalue -1/tau(f) follows the critical law,
              switched to the universal collapse mode at arrest.
  ODE-B       The discovered equation tau(f) dy/dt = g(y - a(f)) (equation_discovery.py),
              refitted without the test force, integrated forward.
  + a check of the ORIGINAL deep Koopman model: teacher-forced vs free-running error.

Run from the project root:  python src/koopman_forecast.py   [--quick]
"""
import os
import sys
import json
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data_loader import load_clean
import scaling_analysis as SA
import equation_discovery as EQ
from train import train_rollout
from deep_koopman import ForceConditionedKoopmanAE

ROOT = os.path.join(os.path.dirname(__file__), "..")
DOCS = os.path.join(ROOT, "docs")
PROC = os.path.join(ROOT, "data", "processed")

SUB = 10                     # subsample: dt = 5 time units
Y_FLOOR = -20.0
ARREST_Y = -10.0             # 'arrested' when log10 K < -10
N_INIT = 20                  # initial window = first 100 t.u.
PREFIXES = (0.10, 0.25, 0.50)
D_ARR = {}                   # force -> arrest time (for plots)
QUICK = '--quick' in sys.argv


# ---------------------------------------------------------------------------
# Data on the coarse grid
# ---------------------------------------------------------------------------
def coarse(d):
    t = d['t'][::SUB]
    y = np.clip(d['y_s'], Y_FLOOR, None)[::SUB]
    return t, y


def arrest_time(t, y):
    i = np.where(y < ARREST_Y)[0]
    return t[i[0]] if len(i) else np.inf


def interp_in_f(C, train, f, n):
    """First n samples of the unseen force: linear interpolation in f between the two
    nearest training forces (nearest one if f is outside their range)."""
    tr = sorted(train)
    lo = [g for g in tr if g <= f]
    hi = [g for g in tr if g >= f]
    if not lo or not hi:
        g = lo[-1] if lo else hi[0]
        return C[g][1][:n].copy()
    g0, g1 = lo[-1], hi[0]
    if g0 == g1:
        return C[g0][1][:n].copy()
    w = (f - g0) / (g1 - g0)
    return (1 - w) * C[g0][1][:n] + w * C[g1][1][:n]


def metrics(t, y_true, y_pred, t_from=0.0):
    m = t >= t_from
    ta_t, ta_p = arrest_time(t, y_true), arrest_time(t, y_pred)
    full = float(np.sqrt(np.nanmean((y_pred[m] - y_true[m]) ** 2)))
    ms = m & (t <= min(ta_t, ta_p))
    slow = float(np.sqrt(np.nanmean((y_pred[ms] - y_true[ms]) ** 2))) if ms.any() else np.nan
    if np.isfinite(ta_t):
        rel = (ta_p - ta_t) / ta_t if np.isfinite(ta_p) else np.inf
    else:
        rel = 0.0 if not np.isfinite(ta_p) else -np.inf   # fluid: must not predict arrest
    return dict(rmse_full=full, rmse_slow=slow, t_arr_true=ta_t, t_arr_pred=ta_p, t_arr_rel_err=float(rel))


# ---------------------------------------------------------------------------
# Parametric / prefix HDMD (affine, time-delay)
# ---------------------------------------------------------------------------
Q_HDMD, R_HDMD = 20, 6


def hankel(y, q):
    n = len(y) - q + 1
    return np.array([y[i:i + n] for i in range(q)])


def affine_op(Z):
    """Least-squares  Z[:,1:] = A Z[:,:-1] + c  ->  augmented operator on [z; 1]."""
    X = np.vstack([Z[:, :-1], np.ones(Z.shape[1] - 1)])
    M = Z[:, 1:] @ np.linalg.pinv(X)
    A = np.vstack([M, np.r_[np.zeros(Z.shape[0]), 1.0]])
    return A


def rollout_affine(A, U, h0, n):
    """Start from Hankel column h0 (last q values), iterate, return newest values."""
    z = np.r_[U.T @ h0, 1.0]
    out = []
    for _ in range(n):
        z = A @ z
        out.append((U @ z[:-1])[-1])
    return np.array(out)


def hdmd_param_predict(C, train, f, n_total):
    Hs = {g: hankel(C[g][1], Q_HDMD) for g in train}
    U = np.linalg.svd(np.hstack(list(Hs.values())), full_matrices=False)[0][:, :R_HDMD]
    ops = {g: affine_op(U.T @ Hs[g]) for g in train}
    tr = sorted(train)
    lo = [g for g in tr if g <= f]
    hi = [g for g in tr if g >= f]
    if not lo or not hi:
        A = ops[lo[-1] if lo else hi[0]]
    else:
        w = 0.0 if hi[0] == lo[-1] else (f - lo[-1]) / (hi[0] - lo[-1])
        A = (1 - w) * ops[lo[-1]] + w * ops[hi[0]]
    y0 = interp_in_f(C, train, f, Q_HDMD)
    pred = rollout_affine(A, U, y0, n_total - Q_HDMD)
    return np.clip(np.r_[y0, pred], Y_FLOOR, 0)


def hdmd_prefix_predict(yp, n_future):
    q = max(4, min(Q_HDMD, len(yp) // 3))
    H = hankel(yp, q)
    r = min(R_HDMD, q - 1)
    U = np.linalg.svd(H, full_matrices=False)[0][:, :r]
    A = affine_op(U.T @ H)
    return np.clip(rollout_affine(A, U, yp[-q:], n_future), Y_FLOOR, 0)


# ---------------------------------------------------------------------------
# Master curve + ODE (physics-informed Koopman), rebuilt without the test force
# ---------------------------------------------------------------------------
def physics_bundle(D, train):
    """Collapse, closure, cliff rate, universal ODE: everything from the training forces."""
    mc_forces = [g for g in train if g <= 1.8]
    ref = 1.4 if 1.4 in mc_forces else 1.45
    mc = SA.master_curve(D, mc_forces, ref=ref)
    tau, a = mc['2param']['tau'], mc['2param']['a']
    S_like = dict(master_curve=dict(forces=mc_forces, tau=list(tau), a=list(a)))
    cl = EQ.closure(S_like, mc_forces)
    arrested = [g for g in mc_forces if np.isfinite(D[g]['t_arr'])]
    gam = -np.nanmean([SA.cliff_profile(D[g])[2] for g in arrested])
    xiB = EQ.fit_model_B(D, mc_forces, cl['a_of'], cl['tau_of'])
    u0 = float(np.median([np.interp(EQ.T0, D[g]['t'], D[g]['y_s']) - cl['a_of'](g) for g in arrested]))
    u_on = float(np.median([EQ.onset_level(D[g]) - cl['a_of'](g) for g in arrested]))
    G = mc['G']
    okG = np.isfinite(G)
    # The ODE is autonomous in rescaled time s = (t - T0)/tau: integrate it ONCE from u0
    # to the collapse onset u_on -> universal solution U(s); every force is y = a + U(s).
    from scipy.integrate import solve_ivp
    ev = lambda s_, u: u[0] - u_on
    ev.terminal, ev.direction = True, -1
    sol = solve_ivp(lambda s_, u: EQ.theta_B(np.atleast_1d(u)) @ xiB, (0, 50), [u0], events=ev,
                    max_step=1e-3, dense_output=True)
    s_on = sol.t_events[0][0] if len(sol.t_events[0]) else sol.t[-1]
    s_grid = np.linspace(0, s_on, 2000)
    return dict(x=mc['x'][okG], G=G[okG], cl=cl, gamma=gam, xiB=xiB, u0=u0, u_on=u_on,
                s=s_grid, U=sol.sol(s_grid)[0], s_on=s_on)


def mastercurve_predict(P, f, t, y_init, tau=None, a=None):
    cl = P['cl']
    fluid = tau is None and f >= cl['fc']
    tau = cl['tau_of'](f) if tau is None and not fluid else tau
    a = cl['a_of'](f) if a is None else a
    if fluid:
        return np.where(t < t[len(y_init)], np.interp(t, t[:len(y_init)], y_init), a + P['G'][0])
    x = t / tau
    y = a + np.interp(x, P['x'], P['G'])
    end = x > P['x'][-1]
    y[end] = a + P['G'][-1] - P['gamma'] * (t[end] - tau * P['x'][-1])
    early = x < P['x'][0]
    y[early] = np.interp(t[early], t[:len(y_init)], y_init) if early.any() else y[early]
    return np.clip(y, Y_FLOOR, 0)


def ode_predict(P, f, t, y_init, tau=None, a=None):
    """y(t) = a + U((t - T0)/tau) until U reaches u_on, then the universal collapse."""
    cl = P['cl']
    if tau is None and f >= cl['fc']:
        a = cl['a_of'](f)
        return np.where(t < EQ.T0, np.interp(t, t[:len(y_init)], y_init), a + P['u0'])
    tau = cl['tau_of'](f) if tau is None else tau
    a = cl['a_of'](f) if a is None else a
    sv = (t - EQ.T0) / tau
    y = a + np.interp(sv, P['s'], P['U'])
    post = sv > P['s_on']
    y[post] = a + P['u_on'] - P['gamma'] * (t[post] - (EQ.T0 + tau * P['s_on']))
    early = t < EQ.T0
    y[early] = np.interp(t[early], t[:len(y_init)], y_init)
    return np.clip(y, Y_FLOOR, 0)


def fit_prefix_scaling(P, tp, yp, which='G'):
    """Fit tau, a of an unseen trajectory to its prefix: yp ~ a + G(t/tau) (master curve)
    or yp ~ a + U((t - T0)/tau) (ODE solution)."""
    m = tp >= EQ.T0
    xs, F, off = (P['x'], P['G'], 0.0) if which == 'G' else (P['s'], P['U'], EQ.T0)
    best = None
    for lt in np.linspace(np.log(100), np.log(5e5), 400):
        x = (tp[m] - off) / np.exp(lt)
        if x.max() > xs[-1]:
            continue
        g = np.interp(x, xs, F)
        a = np.mean(yp[m] - g)
        sse = np.sum((yp[m] - a - g) ** 2)
        if best is None or sse < best[0]:
            best = (sse, np.exp(lt), a)
    return best[1], best[2]


# ---------------------------------------------------------------------------
# Deep Koopman (rollout-trained)
# ---------------------------------------------------------------------------
NORM = lambda y: (y + 10.0) / 10.0
DENORM = lambda z: z * 10.0 - 10.0
F_NORM = lambda f: (f - 1.4) / 0.4


def dk_series(C, D, g):
    t, y = C[g]
    if np.isfinite(D[g]['t_arr']):
        n = min(len(y), np.searchsorted(t, D[g]['t_arr']) + 40)   # keep 200 t.u. of floor
        y = y[:n]
    return NORM(y)


def dk_train(C, D, train):
    ser = {g: dk_series(C, D, g) for g in train}
    return train_rollout(ser, F_NORM, window=N_INIT, horizon=40, epochs=8 if QUICK else 30, verbose=False)


def dk_predict(model, f, y_window, n_future):
    p = model.rollout(torch.tensor(NORM(y_window), dtype=torch.float32), F_NORM(f), n_future, mode='latent')
    return np.clip(DENORM(p), Y_FLOOR, 0)


# ---------------------------------------------------------------------------
# Original model: teacher forcing vs free running
# ---------------------------------------------------------------------------
def original_model_check(D, forces=(0.5, 1.25, 1.6, 1.8)):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    w = 100
    model = ForceConditionedKoopmanAE(input_dim=w, latent_dim=10, hidden_dim=64).to(dev)
    model.load_state_dict(torch.load(os.path.join(PROC, "koopman_model.pth"), map_location=dev, weights_only=True))
    model.eval()
    res, curves = {}, {}
    for f in forces:
        d = D[f]
        n = min(len(d['t']), 20000 if not QUICK else 4000)          # 10 000 t.u.
        yl = np.log10(d['K'][:n] + 1e-12)
        X = torch.tensor(np.lib.stride_tricks.sliding_window_view(yl, w)[:-1].copy(), dtype=torch.float32, device=dev)
        with torch.no_grad():
            tf = torch.cat([model(X[i:i + 4096], torch.full((len(X[i:i + 4096]), 1), f, device=dev))[1][:, -1]
                            for i in range(0, len(X), 4096)]).cpu().numpy()
        free = model.rollout(torch.tensor(yl[:w], dtype=torch.float32), f, n - w, mode='reencode')
        truth = yl[w:]
        res[f] = dict(teacher_forced_rmse=float(np.sqrt(np.mean((tf - truth) ** 2))),
                      free_running_rmse=float(np.sqrt(np.mean((free - truth) ** 2))))
        curves[f] = (d['t'][w:n], truth, tf, free)
        print(f"   original model f={f}: teacher-forced RMSE {res[f]['teacher_forced_rmse']:.3f}  "
              f"free-running RMSE {res[f]['free_running_rmse']:.3f} (decades of K)")
    return res, curves


# ---------------------------------------------------------------------------
def run():
    t_start = time.time()
    D = load_clean()
    forces = [f for f in sorted(D) if f >= 1.0]
    if QUICK:
        forces_eval = [1.0, 1.4, 1.65, 1.8]
    else:
        forces_eval = forces
    C = {f: coarse(D[f]) for f in forces}
    D_ARR.update({f: D[f]['t_arr'] for f in forces})
    models = ['HDMD-param', 'DeepKoopman', 'MasterCurve', 'ODE-B']
    lofo, preds_a, prefix, preds_b = {}, {}, {}, {}

    print("=== Check of the ORIGINAL deep Koopman model (teacher forcing vs free running) ===")
    orig, orig_curves = original_model_check(D)

    for f in forces_eval:
        train = [g for g in forces if g != f]
        t, y = C[f]
        n = len(t)
        y_init = interp_in_f(C, train, f, N_INIT)
        print(f"\n--- f = {f}  (training forces: {len(train)}) ---")
        P = physics_bundle(D, train)
        dk = dk_train(C, D, train)
        pa = {
            'HDMD-param': hdmd_param_predict(C, train, f, n),
            'DeepKoopman': np.r_[y_init, dk_predict(dk, f, y_init, n - N_INIT)],
            'MasterCurve': mastercurve_predict(P, f, t, y_init),
            'ODE-B': ode_predict(P, f, t, y_init),
        }
        preds_a[f] = pa
        lofo[f] = {m: metrics(t, y, pa[m], t_from=t[N_INIT]) for m in models}
        for m in models:
            r = lofo[f][m]
            print(f"   (a) {m:12s} RMSE full {r['rmse_full']:6.2f}  slow {r['rmse_slow']:5.2f}  "
                  f"t_arr true {r['t_arr_true']:8.0f} pred {r['t_arr_pred']:8.0f}  rel err {r['t_arr_rel_err']:+.2f}")

        # ---- protocol (b): prefixes ----
        ta = D[f]['t_arr'] if np.isfinite(D[f]['t_arr']) else t[-1]
        prefix[f], preds_b[f] = {}, {}
        for p in PREFIXES:
            k = max(np.searchsorted(t, p * ta), N_INIT + 4)
            tp, yp = t[:k], y[:k]
            tau_p, a_p = fit_prefix_scaling(P, tp, yp, 'G')
            tau_u, a_u = fit_prefix_scaling(P, tp, yp, 'U')
            pb = {
                'HDMD-prefix': np.r_[yp, hdmd_prefix_predict(yp, n - k)],
                'DeepKoopman': np.r_[yp, dk_predict(dk, f, yp[-N_INIT:], n - k)],
                'MasterCurve': np.r_[yp, mastercurve_predict(P, f, t, y_init, tau=tau_p, a=a_p)[k:]],
                'ODE-B': np.r_[yp, ode_predict(P, f, t, y_init, tau=tau_u, a=a_u)[k:]],
            }
            preds_b[f][p] = pb
            prefix[f][p] = {m: metrics(t, y, v, t_from=tp[-1]) for m, v in pb.items()}
            prefix[f][p]['fitted_tau_a'] = (float(tau_p), float(a_p))
            print(f"   (b) prefix {int(p * 100):2d}%: " + "  ".join(
                f"{m} {prefix[f][p][m]['rmse_full']:.2f}/{prefix[f][p][m]['t_arr_rel_err']:+.2f}"
                for m in pb))
        print(f"   [{time.time() - t_start:.0f}s elapsed]")

    summary = summarise(lofo, prefix, forces_eval)
    plot_lofo(C, preds_a, forces_eval, os.path.join(DOCS, "forecast_lofo.png"))
    plot_prefix(C, preds_b, forces_eval, os.path.join(DOCS, "forecast_prefix.png"), summary)
    plot_original(orig_curves, os.path.join(DOCS, "forecast_original_model.png"))
    out = dict(protocol_a=lofo, protocol_b=prefix, original_model=orig, summary=summary,
               settings=dict(sub=SUB, n_init=N_INIT, arrest_y=ARREST_Y, prefixes=PREFIXES, quick=QUICK))
    with open(os.path.join(PROC, "forecast_results.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=lambda o: None if isinstance(o, float) and not np.isfinite(o) else str(o))
    write_table(summary, orig, os.path.join(DOCS, "forecast_table.md"))
    print(f"\nSaved docs/forecast_*.png, docs/forecast_table.md, data/processed/forecast_results.json "
          f"({time.time() - t_start:.0f}s)")


def summarise(lofo, prefix, forces):
    def agg(rows, key):
        v = np.array([r[key] for r in rows], float)
        if key == 't_arr_rel_err':                 # missed / false arrest = 100 % error
            return float(np.median(np.minimum(np.abs(np.nan_to_num(v, nan=1.0, posinf=1.0, neginf=1.0)), 1.0)))
        return float(np.median(v[np.isfinite(v)]))
    S = {'a': {}, 'b': {}}
    for m in ['HDMD-param', 'DeepKoopman', 'MasterCurve', 'ODE-B']:
        rows = [lofo[f][m] for f in forces]
        S['a'][m] = dict(median_rmse_full=agg(rows, 'rmse_full'), median_rmse_slow=agg(rows, 'rmse_slow'),
                         median_abs_t_arr_err=agg(rows, 't_arr_rel_err'),
                         n_arrest_missed=int(sum(np.isinf(r['t_arr_pred']) and np.isfinite(r['t_arr_true']) for r in rows)),
                         n_false_arrest=int(sum(np.isfinite(r['t_arr_pred']) and np.isinf(r['t_arr_true']) for r in rows)))
    for p in PREFIXES:
        S['b'][p] = {}
        for m in ['HDMD-prefix', 'DeepKoopman', 'MasterCurve', 'ODE-B']:
            rows = [prefix[f][p][m] for f in forces]
            S['b'][p][m] = dict(median_rmse_full=agg(rows, 'rmse_full'), median_rmse_slow=agg(rows, 'rmse_slow'),
                                median_abs_t_arr_err=agg(rows, 't_arr_rel_err'),
                                n_arrest_missed=int(sum(np.isinf(r['t_arr_pred']) and np.isfinite(r['t_arr_true'])
                                                        for r in rows)))
    print("\n=== SUMMARY (median over forces; RMSE in decades of K; |rel. arrest-time error|) ===")
    print("(a) unseen force:")
    for m, v in S['a'].items():
        print(f"   {m:12s} RMSE full {v['median_rmse_full']:.2f}  slow {v['median_rmse_slow']:.2f}  "
              f"|dt_arr| {v['median_abs_t_arr_err']:.2f}  missed arrests {v['n_arrest_missed']}  false arrests {v['n_false_arrest']}")
    for p in PREFIXES:
        print(f"(b) prefix {int(p * 100)}%:")
        for m, v in S['b'][p].items():
            print(f"   {m:12s} RMSE full {v['median_rmse_full']:.2f}  slow {v['median_rmse_slow']:.2f}  "
                  f"|dt_arr| {v['median_abs_t_arr_err']:.2f}  missed arrests {v['n_arrest_missed']}")
    return S


COLORS = {'HDMD-param': 'tab:blue', 'HDMD-prefix': 'tab:blue', 'DeepKoopman': 'tab:green',
          'MasterCurve': 'tab:red', 'ODE-B': 'tab:purple'}


def plot_lofo(C, preds, forces, path):
    n = len(forces)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.2 * rows), sharey=True)
    ax = np.atleast_1d(ax).ravel()
    for i, f in enumerate(forces):
        t, y = C[f]
        ax[i].plot(t, y, 'k', lw=3, alpha=0.3, label='truth')
        for m, v in preds[f].items():
            ax[i].plot(t, v, '--', color=COLORS[m], lw=1.3, label=m)
        ax[i].set(title=f'f = {f} (never seen)', xscale='symlog', xlim=(0, t[-1]), ylim=(-21, -1))
        ax[i].grid(alpha=0.3)
    for j in range(n, len(ax)):
        ax[j].axis('off')
    ax[0].legend(fontsize=7)
    fig.suptitle('(a) Predicting a whole trajectory from the force alone (leave-one-force-out)')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_prefix(C, preds, forces, path, S, p=0.25):
    show = [f for f in (1.0, 1.4, 1.6, 1.65, 1.8) if f in forces][:4]
    fig, ax = plt.subplots(1, len(show) + 1, figsize=(4.4 * (len(show) + 1), 3.8))
    for i, f in enumerate(show):
        t, y = C[f]
        ax[i].plot(t, y, 'k', lw=3, alpha=0.3, label='truth')
        for m, v in preds[f][p].items():
            ax[i].plot(t, v, '--', color=COLORS[m], lw=1.3, label=m)
        ta = D_ARR[f] if np.isfinite(D_ARR[f]) else t[-1]
        ax[i].axvspan(0, p * ta, color='gold', alpha=0.2, label='given prefix')
        ax[i].set(title=f'f = {f}, prefix {int(p * 100)}%', ylim=(-21, -1))
        ax[i].grid(alpha=0.3)
    ax[0].legend(fontsize=7)
    a = ax[-1]
    ms = ['HDMD-prefix', 'DeepKoopman', 'MasterCurve', 'ODE-B']
    for m in ms:
        a.plot([int(q * 100) for q in PREFIXES], [S['b'][q][m]['median_abs_t_arr_err'] for q in PREFIXES],
               'o-', color=COLORS[m], label=m)
    a.set(xlabel='prefix given (% of arrest time)', ylabel='median |rel. error of arrest time|',
          title='Forecast skill vs prefix length', yscale='log')
    a.legend(fontsize=7)
    a.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_original(curves, path):
    fig, ax = plt.subplots(1, len(curves), figsize=(4.5 * len(curves), 3.6), sharey=True)
    for a, (f, (t, truth, tf, free)) in zip(np.atleast_1d(ax), curves.items()):
        a.plot(t, truth, 'k', lw=3, alpha=0.3, label='truth')
        a.plot(t, tf, 'b--', lw=1, label='teacher-forced (old plot)')
        a.plot(t, np.clip(free, -30, 5), 'r-', lw=1, label='free-running')
        a.set(title=f'original model, f = {f}', xlabel='t', ylim=(-28, 1))
        a.grid(alpha=0.3)
    np.atleast_1d(ax)[0].legend(fontsize=7)
    np.atleast_1d(ax)[0].set_ylabel(r'$\log_{10}K$')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def write_table(S, orig, path):
    L = ["# Forecast results (auto-generated by `src/koopman_forecast.py`)", "",
         "Median over forces. RMSE in decades of K (log10): 'full' = whole forecast incl. the collapse "
         "(dominated by arrest timing), 'slow' = up to the earlier of true/predicted arrest. "
         "|dt_arr| = median relative error of the arrest time (a missed arrest counts as 1.0).", "",
         "## (a) Unseen force (leave-one-force-out, prediction from f alone)", "",
         "| model | RMSE full | RMSE slow stage | median abs dt_arr | missed arrests | false arrests |",
         "|---|---|---|---|---|---|"]
    for m, v in S['a'].items():
        L.append(f"| {m} | {v['median_rmse_full']:.2f} | {v['median_rmse_slow']:.2f} | {v['median_abs_t_arr_err']:.2f} "
                 f"| {v['n_arrest_missed']} | {v['n_false_arrest']} |")
    for p in PREFIXES:
        L += ["", f"## (b) Forecast from the first {int(p * 100)}% of an unseen trajectory", "",
              "| model | RMSE full | RMSE slow stage | median abs dt_arr | missed arrests |", "|---|---|---|---|---|"]
        for m, v in S['b'][p].items():
            L.append(f"| {m} | {v['median_rmse_full']:.2f} | {v['median_rmse_slow']:.2f} | {v['median_abs_t_arr_err']:.2f} "
                     f"| {v['n_arrest_missed']} |")
    L += ["", "## Original deep Koopman model (data/processed/koopman_model.pth)", "",
          "| f | teacher-forced RMSE | free-running RMSE |", "|---|---|---|"]
    for f, v in orig.items():
        L.append(f"| {f} | {v['teacher_forced_rmse']:.3f} | {v['free_running_rmse']:.3f} |")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


if __name__ == "__main__":
    run()
