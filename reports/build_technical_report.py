"""
Builds reports/Technical_Report.pdf from the saved analysis results and docs/figures/.
Run from the project root after all analysis scripts and src/report_figures.py:
    python reports/build_technical_report.py
"""
import os
import sys
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, KeepTogether, CondPageBreak

sys.path.insert(0, os.path.dirname(__file__))
from pdfkit_common import (make_styles, bullets, figure, table, callout, page_decorator, load, fnum, pct, sci, r2fmt,
                           forecast_summary, FIG, GOOD, WARN, BAD, SHADE)

OUT = os.path.join(os.path.dirname(__file__), "Technical_Report.pdf")
TITLE = "Kinetic arrest in a dense active glass: scaling, Koopman prediction, HAVOK and reduced equations"
AUTHOR = "Rithvik"
DATE = "October 2026"
W = A4[0] - 36 * mm


def build():
    S = make_styles("Serif", 9.8)
    Sc = load("scaling_results.json")
    Hv = load("havok_results.json")
    Eq = load("equation_results.json")
    V = load("validation_results.json")
    OS = load("onestep_vs_free.json")
    FS = forecast_summary()
    fa = FS['a']
    fits = Sc['fits']
    mc = Sc['master_curve']
    v1, v2, v3, v4, v5, v6, v7 = (V[k] for k in ('V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7'))
    gB = Eq['model_B']
    story = []
    def P(t, st='body'):
        if st in ('h1', 'h2'):
            story.append(CondPageBreak(55 * mm))      # never leave a heading alone at the foot of a page
        story.append(Paragraph(t, S[st]))

    # ------------------------------------------------------------------ title
    story += [Spacer(1, 30 * mm), Paragraph(TITLE, S['title']), Spacer(1, 4 * mm),
              Paragraph(f"{AUTHOR}<br/>{DATE}", S['subtitle']), Spacer(1, 12 * mm)]
    best = fa['MasterCurve']
    abstract = (
        "We analyse the kinetic energy K(t) of a two-dimensional Kob–Andersen binary mixture (N = 1000) driven by "
        "a persistent active force of magnitude f, using one trajectory for each of 17 forces between 0 and 2. "
        "For 1.0 ≤ f ≤ 1.65 the energy decays quasi-exponentially and then collapses abruptly when the system "
        "arrests; f = 1.8 is still decaying at the end of the simulated window (t ≈ 3·10<super>4</super>) and "
        "f = 2.0 is nearly stationary. "
        f"(i) All trajectories with 1.0 ≤ f ≤ 1.8 collapse onto a master curve log<sub>10</sub>K = a(f) + G(t/τ(f)) "
        f"(R<super>2</super> = {1 - mc['cost_2param'] / mc['cost_raw']:.3f}); the arrest time scales with the slow-decay "
        f"time, measured without using the collapse, as t<sub>arr</sub> ∝ τ<sub>K</sub><super>{v2['beta']:.2f}</super> "
        f"(R<super>2</super> = {v2['r2']:.2f}), and the final collapse proceeds at a force-independent rate. "
        "(ii) The relaxation time grows faster than exponentially (Arrhenius rejected, ΔAIC = 11–26); power-law fits "
        "give f<sub>c</sub> ≈ 1.8–1.95 and γ ≈ 1.4–2.1, with a strong f<sub>c</sub>–γ degeneracy. "
        "(iii) HAVOK reveals strongly non-Gaussian, intermittent forcing in every arresting trajectory (excess kurtosis "
        "3.5–32, far outside phase-randomised surrogates) and Gaussian forcing at f = 2.0; the intermittency appears "
        "in the last ~20 % of the slow stage and precedes sudden drops of K at 2–4× chance. "
        "(iv) Weak-form sparse regression yields a compact reduced equation τ(f) dy/dt = g(y − a(f)), y = log<sub>10</sub>K, "
        "the differential form of the collapse. "
        "(v) In free-running, leave-one-force-out prediction, linear Hankel-DMD operators and a rollout-trained deep "
        "Koopman autoencoder do not reproduce the arrest, whereas a Koopman model built on the scaling structure "
        f"predicts the arrest time of unseen forces with a median error of {pct(best['med'])} "
        f"(95% CI {pct(best['lo'])}–{pct(best['hi'])}). All claims are accompanied by explicit validation tests "
        "(Section 5)."
    )
    story.append(callout([Paragraph("<b>Abstract.</b> " + abstract, S['body'])], S, width=W))
    story.append(PageBreak())

    # ------------------------------------------------------------------ 1 intro
    P("1. Introduction", 'h1')
    P("Dense assemblies of self-propelled particles combine glassy caging with persistent driving. Depending on "
      "the strength of the active force they either flow as an active fluid or arrest in a mechanically stable, "
      "jammed configuration. Existing theoretical descriptions, such as active extensions of mode-coupling theory "
      "[8–10] and hydrodynamic theories of active matter [16, 17], do not yet provide a closed prediction for the "
      "time dependence of global observables near this arrest. This motivates a data-driven approach.")
    P("This report addresses two questions about the mean kinetic energy K(t; f):")
    story += bullets([
        "<b>Q1 – Scaling.</b> Do trajectories at different active forces obey a common scaling form, i.e. does a master "
        "curve exist, and how do the characteristic time scales depend on f?",
        "<b>Q2 – Prediction.</b> Can Koopman-operator methods predict a kinetic-energy time series that was not used to "
        "build the model, either from the force alone or from the beginning of the trajectory?"], S)
    P("Both questions are approached together with a search for a governing differential equation (sparse "
      "identification [3, 4]) and with the Hankel Alternative View of Koopman (HAVOK) [2], which separates linear "
      "dynamics from intermittent forcing. Every claim is tested against a baseline, a null model or held-out data; "
      "Section 5 collects these tests.")

    # ------------------------------------------------------------------ 2 data
    P("2. System and data", 'h1')
    P("The system is a two-dimensional 65:35 Kob–Andersen Lennard-Jones mixture [1] of N = 1000 particles subject "
      "to an active force of magnitude f. For each f ∈ {0, 0.25, 0.5, 0.75, 1.0, 1.05, 1.15, 1.25, 1.35, 1.4, 1.45, "
      "1.5, 1.55, 1.6, 1.65, 1.8, 2.0} one trajectory of the mean kinetic energy K(t) is available, sampled at "
      "Δt = 0.5 up to t = 29 505 (59 011 points). All trajectories start from the same value K(0) = 0.1072. "
      "We work with y = log<sub>10</sub>K. After arrest, K decays to a numerical floor of ~10<super>−26</super>; values below "
      "10<super>−20</super> are discarded. The single final sample (Δt = 0.05) is dropped. For smoothed quantities a centred "
      "41-point moving average is used.")
    rows = [["f", "regime", "t<sub>arr</sub>", "τ<sub>K</sub>", "τ<sub>mc</sub>", "a(f)"]]
    forces = Sc['forces']
    taus = Sc['taus']
    amap = dict(zip(mc['forces'], mc['a']))
    tmap = dict(zip(mc['forces'], mc['tau']))
    for i, f in enumerate(forces):
        ta = taus['t_arr'][i]
        reg = "arrests" if ta is not None and np.isfinite(ta) else ("slow decay, no arrest by end" if f == 1.8 else "near-stationary")
        rows.append([f"{f:.2f}", reg, fnum(ta, 0) if ta is not None else "> 29 505", fnum(taus['tau_K'][i], 0),
                     fnum(tmap.get(f), 0) if f in tmap else "–", fnum(amap.get(f), 2) if f in amap else "–"])
    story.append(KeepTogether([table(rows, S, [14 * mm, 52 * mm, 24 * mm, 24 * mm, 24 * mm, 20 * mm]),
                               Paragraph("Table 1. Forces analysed in detail (f ≥ 1.0). t<sub>arr</sub>: time at which K falls "
                                         "below 10<super>−20</super>; τ<sub>K</sub>: e-folding time of the slow stage; τ<sub>mc</sub>, a(f): "
                                         "time scale and offset from the master-curve collapse (reference f = 1.4). Forces "
                                         "f ≤ 0.75 relax like a quench (fast decay with oscillations) and are not part of "
                                         "the scaling analysis.", S['caption'])]))
    story.append(figure(os.path.join(FIG, "fig1_data.png"), W,
                        "Figure 1. (a) log<sub>10</sub>K(t) for all 17 forces (smoothed). For f ≥ 1.0 a short transient is "
                        "followed by a slow, almost exponential decay and an abrupt collapse. (b) Arrest time versus f; "
                        "f = 1.8 and 2.0 do not arrest within the simulated window (lower bounds).", S))

    # ------------------------------------------------------------------ 3 methods
    story.append(CondPageBreak(60 * mm))
    P("3. Methods", 'h1')
    P("3.1 Relaxation times and critical fits", 'h2')
    P("Several operational definitions of a relaxation time are used so that conclusions do not depend on one "
      "choice: the arrest time t<sub>arr</sub>; the time at which the smoothed y first reaches −5; the e-folding time "
      "τ<sub>K</sub> of the slow stage, from a linear fit of ln K between t = 20 and 0.85 t<sub>arr</sub> (for non-arresting "
      "forces, to the end of the run), so that τ<sub>K</sub> never uses the collapse itself; and the time scale "
      "τ<sub>mc</sub> from the master-curve collapse. Each is fitted with τ = A(f<sub>c</sub> − f)<super>−γ</super>, "
      "profiling f<sub>c</sub> on a grid above the largest force in the fit, and compared with a Vogel–Fulcher-type form "
      "ln τ = a + b/(f<sub>c</sub> − f) and an Arrhenius-type form ln τ = a + bf using the Akaike information "
      "criterion [14]. Uncertainties come from a non-parametric bootstrap over forces (500–2000 resamples) and from "
      "the set of f<sub>c</sub> values with ΔAIC < 2.")
    P("3.2 Master-curve collapse", 'h2')
    P("Curves are compared on a grid x = t/τ(f) ∈ [0.02, 1] for t > 20. The collapse cost is the mean, over x, of the "
      "variance of y − a(f) across forces. The time scales τ(f) are optimised (Powell, bounded within a factor e of "
      "the initial guess t<sub>arr</sub>) and, for each set of time scales, the offsets a(f) are obtained by alternating "
      "least squares with a(1.4) = 0. Spread values are reported for no rescaling, time rescaling only, and time plus "
      "offset.")
    P("3.3 HAVOK", 'h2')
    P("Following [2], the fluctuation δy = y − (linear slow trend) is embedded in a Hankel matrix with q = 40 delays "
      "(20 time units). The SVD gives coordinates v<sub>1</sub> … v<sub>r</sub>; a linear model "
      "dv<sub>1..r−1</sub>/dt = A v<sub>1..r−1</sub> + B v<sub>r</sub> is fitted by least squares using fourth-order "
      "differences of the SVD time series, and v<sub>r</sub> is treated as forcing. The rank r is chosen as the "
      "(marginally) stable model with the smallest forced-reconstruction error. Forcing is 'active' when "
      "|v<sub>r</sub>| > 2.5σ. All event statistics are causal: a Hankel column is time-stamped at the end of its "
      "window, and chance levels are estimated from random times. The implementation was validated on the Lorenz "
      "system and against PyDMD (Section 5).")
    P("3.4 Equation discovery", 'h2')
    P("Sparse identification [3] is carried out in weak form [4]: dy/dt = Θ(y)ξ is multiplied by compactly supported "
      "test functions φ<sub>k</sub>(t) = [(t − a)(b − t)]<super>4</super> (width 100 time units, stride 25) and "
      "integrated by parts, so the noisy data are never differentiated. Coefficients are found by sequentially "
      "thresholded least squares, with the threshold chosen by BIC using the number of independent windows. Two "
      "libraries are used: a generic one, monomials in (y, f) up to degree 2 × 2, and a scaling-informed one, in which "
      "u = y − a(f), s = t/τ(f) and du/ds = g(u) is a polynomial. A stochastic version is obtained from "
      "Kramers–Moyal conditional moments [13] of u at a lag of 5 time units.")
    P("3.5 Prediction protocols and metrics", 'h2')
    P("All predictions are free-running: after the initial condition, the model receives no data from the trajectory "
      "being predicted. In protocol (a), each of the 13 forces with f ≥ 1.0 is left out in turn, and every fitted "
      "quantity (operators, network weights, master curve, τ(f), a(f), ODE) is rebuilt without it. The first 100 time "
      "units are taken as an interpolation of neighbouring training forces. In protocol (b) the first 10, 25 or 50 % "
      "of the arrest time of the unseen trajectory is given. Models: parametric Hankel DMD [5, 6, 15] with operators "
      "interpolated in f; HDMD fitted on the prefix (equivalent to HAVOK with unknown future forcing set to zero); a "
      "force-conditioned Koopman autoencoder [7] trained with a 40-step latent rollout loss; a master-curve Koopman "
      "model (one slow mode with eigenvalue −1/τ(f) from the critical law, followed by the collapse mode); the "
      "discovered ODE; and a non-Koopman baseline that copies the nearest training force and stretches its time axis "
      "to an interpolated arrest time. Metrics: RMSE, MAE and R<super>2</super> of y on the slow stage (up to the "
      "earlier of the true and predicted arrest), and the relative error of the arrest time (time at which y < −10). "
      "A missed arrest counts as 100 % error. Medians over forces are reported with 95 % bootstrap confidence "
      "intervals.")

    # ------------------------------------------------------------------ 4 results
    story.append(CondPageBreak(90 * mm))
    P("4. Results", 'h1')
    P("4.1 Master curve (Q1)", 'h2')
    P(f"Rescaling time alone reduces the spread of log<sub>10</sub>K across the 12 trajectories with 1.0 ≤ f ≤ 1.8 from "
      f"{mc['cost_raw']:.2f} to {mc['cost_1param']:.3f}; adding an offset a(f) reduces it to {mc['cost_2param']:.3f}, i.e. "
      f"{100 * (1 - mc['cost_2param'] / mc['cost_raw']):.1f} % of the variance is captured by a single function G(x) "
      "(Fig. 2a,b). The non-arresting trajectory f = 1.8 follows the same G over the range it covers. Because a "
      "two-parameter rescaling has considerable freedom, the collapse is tested independently. τ<sub>K</sub> is measured "
      "from the slow stage only, and the arrest time is then compared with it (Fig. 2c): "
      f"t<sub>arr</sub> ∝ τ<sub>K</sub><super>β</super> with β = {v2['beta']:.2f} (95 % CI {v2['beta_ci95'][0]:.2f}–"
      f"{v2['beta_ci95'][1]:.2f}, R<super>2</super> = {v2['r2']:.2f}). While τ<sub>K</sub> varies by a factor "
      f"{v2['tau_K_range'][1] / v2['tau_K_range'][0]:.0f}, the ratio t<sub>arr</sub>/τ<sub>K</sub> = {v2['ratio_mean']:.1f} "
      f"varies by only {pct(v2['ratio_cv'])} (coefficient of variation). The arrest therefore occurs at an almost fixed "
      "point of the slow decay, with a slight sub-linearity (β < 1). After the cliff, log<sub>10</sub>K decreases at "
      f"{-Sc['cliff_rate']['mean']:.2f} ± {Sc['cliff_rate']['std']:.2f} decades per time unit for every force "
      f"(K ∝ e<super>−{-Sc['cliff_rate']['mean'] * np.log(10):.2f} t</super>, Fig. 2d).")
    story.append(figure(os.path.join(FIG, "fig2_master_curve.png"), W,
                        "Figure 2. (a) Slow stage of log<sub>10</sub>K for 1.0 ≤ f ≤ 1.8 (dotted: f = 1.8). (b) Collapse "
                        "log<sub>10</sub>K − a(f) versus t/τ(f) with the median master curve G(x). (c) Arrest time versus "
                        "slow-stage decay time, which is fitted without using the collapse; solid: power-law fit, dotted: "
                        "proportionality. (d) Final collapse aligned at its onset, with a force-independent slope.", S))
    P("4.2 Critical scaling of the time scale (Q1)", 'h2')
    rows = [["τ definition", "n", "best f<sub>c</sub>", "best γ", "γ (68 % bootstrap)", "f<sub>c</sub>, γ with ΔAIC < 2",
             "R<super>2</super>", "ΔAIC Arrhenius", "ΔAIC VFT"]]
    names = {'t_arr': "t<sub>arr</sub>", 't_-5': "t(K = 10<super>−5</super>)", 'tau_K': "τ<sub>K</sub>", 'tau_mc': "τ<sub>mc</sub>",
             't_-4': "t(K = 10<super>−4</super>)", 't_-3': "t(K = 10<super>−3</super>)", 'tau_HDMD': "HDMD on K"}
    for k in ('t_arr', 't_-5', 'tau_K', 'tau_mc', 't_-4', 't_-3', 'tau_HDMD'):
        ft = fits[k]
        pl, ad, bs, a_ = ft['powerlaw'], ft['admissible'], ft['bootstrap'], ft['aic']
        rows.append([names[k], str(len(ft['forces'])), fnum(pl['fc']), fnum(pl['gamma']),
                     f"{fnum(bs['gamma_ci'][0])}–{fnum(bs['gamma_ci'][2])}",
                     f"{fnum(ad['fc_range'][0])}–{fnum(ad['fc_range'][1])}; {fnum(ad['gamma_range'][0])}–{fnum(ad['gamma_range'][1])}",
                     fnum(pl['r2'], 3), fnum(a_['arrhenius'] - a_['powerlaw'], 1), fnum(a_['vft'] - a_['powerlaw'], 1)])
    story.append(KeepTogether([table(rows, S, [24 * mm, 7 * mm, 14 * mm, 12 * mm, 20 * mm, 33 * mm, 12 * mm, 20 * mm, 15 * mm],
                                     highlight={5: WARN, 6: WARN, 7: WARN}),
                               Paragraph("Table 2. Fits of τ = A(f<sub>c</sub> − f)<super>−γ</super>. ΔAIC columns are relative to "
                                         "the power law (positive: power law preferred). The first four rows are the robust "
                                         "definitions; the last three (shaded) are biased: times to reach a fixed level "
                                         "mix the f-dependent offset a(f) with the time scale, and HDMD on linear K is "
                                         "dominated by the first few hundred time units.", S['caption'])]))
    story.append(figure(os.path.join(FIG, "fig3_critical_scaling.png"), W,
                        "Figure 3. (a) Relaxation times versus f with power-law fits. (b) Same data versus f<sub>c</sub> − f, each "
                        "with its own best f<sub>c</sub>. (c) ΔAIC of the power-law fit as a function of the assumed "
                        "f<sub>c</sub>. (d) Best γ for each assumed f<sub>c</sub> (thick: ΔAIC < 2) and bootstrap fits "
                        "(dots) for τ<sub>K</sub> and τ<sub>mc</sub>.", S))
    P("For the four robust definitions, an Arrhenius law is rejected (ΔAIC = 11–26), whereas a power law and a "
      "VFT-type law describe the data equally well (|ΔAIC| < 2.5). The best estimates span f<sub>c</sub> = 1.81–1.94 and "
      "γ = 1.44–2.10; Fig. 3d shows that f<sub>c</sub> and γ are strongly correlated, so that the data constrain a "
      "curve in the (f<sub>c</sub>, γ) plane rather than a point. For τ<sub>K</sub>, which includes f = 1.8, the result is "
      f"insensitive to the fitting window (γ = {min(v['gamma'] for v in v7.values()):.2f}–{max(v['gamma'] for v in v7.values()):.2f}, "
      f"f<sub>c</sub> = {min(v['fc'] for v in v7.values()):.3f}–{max(v['fc'] for v in v7.values()):.3f}; Fig. S1c). "
      f"The f = 1.8 trajectory is significantly decaying (slope {sci(v1['1.8']['slope'])} decades per time unit, 95 % CI "
      f"{sci(v1['1.8']['ci95'][0])} to {sci(v1['1.8']['ci95'][1])}, block bootstrap), which places f<sub>c</sub> above 1.8 if "
      f"this decay ends in arrest (extrapolated t<sub>arr</sub> ~ {sci(mc['t_arr_over_tauK'] * taus['tau_K'][forces.index(1.8)], 1)}). "
      f"f = 2.0 also shows a small but statistically significant downward drift ({v1['2.0']['total_drop_decades']:.3f} "
      "decades over the analysed window). If this drift were the same slow decay, f<sub>c</sub> would lie above 2.0 "
      "(≈ 2.1 with γ ≈ 3); the present data cannot exclude this.")
    P("4.3 Intermittency revealed by HAVOK", 'h2')
    rows = [["f", "r", "forcing kurtosis [95 % CI]", "surrogates (max)", "dips: preceded / chance",
             "pre-arrest active (local, global baseline)"]]
    hv = {r['f']: r for r in Hv['forces']}
    for k in sorted(v3, key=float):
        f = float(k)
        r = hv[f]
        pre = (f"{pct(r['cliff_forcing_active_frac'])} ({pct(r['cliff_local_baseline_frac'])}, {pct(r['frac_active'])})"
               if 'cliff_forcing_active_frac' in r else "–")
        rows.append([f"{f:.2f}", str(v3[k]['r']), f"{v3[k]['kurtosis_data']:.1f} [{v3[k]['kurtosis_data_ci95'][0]:.1f}, "
                                                  f"{v3[k]['kurtosis_data_ci95'][1]:.1f}]",
                     f"{v3[k]['kurtosis_surrogates'][2]:.2f}",
                     f"{pct(r['dip_precursor_hit'])} / {pct(r['dip_precursor_chance'])}" if r['n_dips'] else "–", pre])
    story.append(KeepTogether([table(rows, S, [10 * mm, 7 * mm, 38 * mm, 22 * mm, 34 * mm, 63 * mm]),
                               Paragraph("Table 3. HAVOK diagnostics (q = 40). Kurtosis: excess kurtosis of the forcing "
                                         "v<sub>r</sub>; 0 for a Gaussian. Surrogates: largest value among 19 phase-randomised "
                                         "surrogates with the same power spectrum. Dips: fraction of strong downward excursions "
                                         "of K preceded within 20 time units by active forcing, versus the same fraction at "
                                         "random times. Pre-arrest: fraction of time with active forcing 120–20 time units before "
                                         "the collapse, compared with the preceding 500 time units (local) and the whole "
                                         "trajectory (global).", S['caption'])]))
    P("The forcing is strongly non-Gaussian for every arresting force and lies far outside the surrogate "
      "distribution (all 19 surrogates have |kurtosis| < 0.7, Fig. 4c). It is weakly non-Gaussian at f = 1.8 "
      "(0.58, 95 % CI 0.39–0.78) and Gaussian at f = 2.0, where it is indistinguishable from the surrogates. "
      "Active forcing is essentially absent in the first 60 % of the slow stage and appears in its last ~20 % "
      "(x = t/τ ≥ 0.8, K ~ 10<super>−5</super>), together with growing fluctuations of log K (Fig. 4a). Sudden drops of K "
      f"are preceded by active forcing {v4['40']['dip_hit_over_chance_median']:.1f}× more often than random times "
      "(median over forces). In the 100 time units before the collapse, active forcing is "
      f"{v4['40']['prearrest_over_global_median']:.0f}× more frequent than on average over the trajectory, but only "
      f"{v4['40']['prearrest_over_local_median']:.1f}× more frequent than in the preceding 500 time units (medians). "
      "The pre-arrest enhancement thus largely reflects the general rise of intermittency as K approaches the "
      "arrest level; a specific burst immediately before arrest is clear only for f ≤ 1.45. These conclusions are "
      "unchanged for q = 20 and 80 (Table 9). The linear part of the HAVOK model is marginally stable for most "
      "forces and is used here as a diagnostic rather than as a forecaster.")
    story.append(figure(os.path.join(FIG, "fig4_havok.png"), W,
                        "Figure 4. (a) f = 1.6: fluctuation of log<sub>10</sub>K around its slow trend with detected dips, and the "
                        "HAVOK forcing v<sub>r</sub> (red: active). (b) Distribution of the standardised forcing for "
                        "several forces, with a Gaussian for reference. (c) Excess kurtosis of the forcing with 95 % "
                        "block-bootstrap intervals; the band spans 19 phase-randomised surrogates per force.", S))

    P("4.4 Reduced governing equation", 'h2')
    co = gB['g_coefficients']
    tl = gB['tau_law']
    P("With the generic library in (y, f), weak SINDy and a strong-form PySINDy fit return dense equations whose "
      "coefficients disagree with each other and which diverge when integrated for f ≥ 1.55. A polynomial in f "
      "cannot represent a rate that diverges at f<sub>c</sub>. With the scaling-informed library, the selected model is")
    P(f"τ(f) dy/dt = g(y − a(f)),  g(u) = {co['u']:.2f} u + {co['u^2']:.2f} u<super>2</super> + {co['u^3']:.3f} u<super>3</super>,", 'eq')
    P(f"with τ(f) = {tl['A']:.0f} ({tl['fc']:.3f} − f)<super>−{tl['gamma']:.2f}</super>, a(f) = {gB['a_linear'][0]:.2f} f "
      f"{gB['a_linear'][1]:+.2f}, followed by dy/dt = −{Eq['gamma_cliff']:.2f} once u < {Eq['u_on']:.1f}. This equation is "
      "the differential form of the collapse of Section 4.1, written with about eight constants. It is a reduced "
      "empirical model for one global observable, not a microscopic equation of motion.")
    b5 = v5['bootstrap']
    rows = [["term", "coefficient", "95 % bootstrap CI", "selected in"]]
    for n, m_, lo, hi, fs_ in zip(b5['names'], b5['median'], b5['ci95_low'], b5['ci95_high'], b5['frac_selected']):
        rows.append([n.replace('^2', '<super>2</super>').replace('^3', '<super>3</super>'),
                     fnum(co.get(n, 0.0), 3), f"[{lo:.2f}, {hi:.2f}]", pct(fs_)])
    story.append(KeepTogether([table(rows, S, [22 * mm, 30 * mm, 45 * mm, 25 * mm]),
                               Paragraph("Table 4. Coefficients of g(u) with bootstrap over forces (200 resamples). The "
                                         "cubic structure is selected in 89–99 % of resamples; individual coefficients are "
                                         "correlated and therefore broad, while g(u) itself is well constrained in the data "
                                         "range (Fig. 5a).", S['caption'])]))
    lof = Eq['validation']
    errs = [abs(float(lof[k]['B_lofo']['rel_t_arr_err'])) for k in lof if lof[k]['B_lofo']['rel_t_arr_err'] is not None
            and np.isfinite(float(lof[k]['B_lofo']['rel_t_arr_err']))]
    P(f"Validated leave-one-force-out, with the equation re-derived each time, the model reproduces log<sub>10</sub>K on "
      f"the slow stage with a median RMSE of {np.median([lof[k]['B_lofo']['rmse'] for k in lof]):.2f} decades and the "
      f"arrest time with a median error of {pct(np.median(errs))} (Fig. 5b). A quadratic g is clearly worse "
      f"({pct(v5['degree']['2']['median_abs_t_arr_err'])} versus {pct(v5['degree']['3']['median_abs_t_arr_err'])} for "
      f"the cubic and {pct(v5['degree']['4']['median_abs_t_arr_err'])} for a quartic, in a common leave-one-out test). "
      "|g| has a minimum near u ≈ −4.6, a bottleneck that coincides with the onset of intermittency (Section 4.3). "
      f"The hypothesis dK/dt = −AK + B f K<super>n</super> is not supported: n is not identifiable (residual changes by "
      f"< 15 % for n = 0–3), the best fit has B = {sci(Eq['model_H']['B'], 1)} < 0, and arrest times are off by 50–200 %.")
    L = Eq['model_L']
    P("The Kramers–Moyal analysis shows that the diffusion coefficient of u grows by about three orders of "
      "magnitude as K approaches the arrest level (Fig. 5c). The resulting Langevin model du = g(u) ds + "
      "(2D<sub>2</sub>(u))<super>1/2</super> dW reproduces the median rescaled arrest time "
      f"(s* = {L['s_arrest_sim'][1]:.2f} versus {np.median(L['s_arrest_obs']):.2f} observed), but its 16–84 % range "
      f"({L['s_arrest_sim'][0]:.2f}–{L['s_arrest_sim'][2]:.2f}) is wider than the observed spread "
      f"({np.percentile(L['s_arrest_obs'], 16):.2f}–{np.percentile(L['s_arrest_obs'], 84):.2f}). The diffusion estimate "
      "is inflated by the intermittent jumps, which a Gaussian Langevin model cannot represent (Fig. 5d).")
    story.append(figure(os.path.join(FIG, "fig5_equation.png"), W,
                        "Figure 5. (a) g(u) from weak SINDy with a 95 % bootstrap band, and Kramers–Moyal drift estimates "
                        "(which are biased low near arrest by jump events). (b) Leave-one-force-out integration of the "
                        "equation (dashed) against the data. (c) Kramers–Moyal diffusion coefficient. (d) Rescaled arrest "
                        "times: Langevin model versus the 11 observed values.", S))

    P("4.5 Prediction of unseen trajectories (Q2)", 'h2')
    rows = [["model", "median |Δt<sub>arr</sub>|/t<sub>arr</sub> [95 % CI]", "within 25 %", "missed arrests",
             "false arrests", "RMSE (dec.)", "MAE (dec.)", "R<super>2</super>"]]
    lab = {'NeighbourBaseline': "Neighbour baseline", 'HDMD-param': "Parametric HDMD", 'DeepKoopman': "Deep Koopman AE",
           'MasterCurve': "Master-curve Koopman", 'ODE-B': "Discovered ODE"}
    for m in ('NeighbourBaseline', 'HDMD-param', 'DeepKoopman', 'MasterCurve', 'ODE-B'):
        r = fa[m]
        rows.append([lab[m], f"{pct(r['med'])} [{pct(r['lo'])}, {pct(r['hi'])}]", f"{r['within25']}/{r['n']}",
                     f"{r['missed']}/{r['n']}", f"{r['false']}/{r['n_fluid']}", fnum(r['rmse']), fnum(r['mae']), r2fmt(r['r2'])])
    story.append(KeepTogether([table(rows, S, [32 * mm, 40 * mm, 16 * mm, 17 * mm, 16 * mm, 15 * mm, 15 * mm, 12 * mm],
                                     highlight={4: GOOD}),
                               Paragraph("Table 5. Protocol (a): prediction of a whole trajectory from the force alone, "
                                         f"leave-one-force-out over {len(FS['forces'])} forces ({fa['MasterCurve']['n']} arresting, "
                                         f"{fa['MasterCurve']['n_fluid']} non-arresting). RMSE, MAE and R<super>2</super> refer to "
                                         "log<sub>10</sub>K on the slow stage of the arresting forces (medians). A missed arrest counts "
                                         "as 100 % error.", S['caption'])]))
    rows = [["f"] + [lab[m] for m in ('NeighbourBaseline', 'HDMD-param', 'DeepKoopman', 'MasterCurve', 'ODE-B')]]
    for k in FS['forces']:
        row = [k]
        for m in ('NeighbourBaseline', 'HDMD-param', 'DeepKoopman', 'MasterCurve', 'ODE-B'):
            r = fa[m]['per_force'][k]
            tp, tt = r['t_arr_pred'], r['t_arr_true']
            tt_f = tt is not None and np.isfinite(float(tt))
            tp_f = tp is not None and np.isfinite(float(tp))
            if tt_f:
                row.append(f"{float(r['t_arr_rel_err']):+.0%}" if tp_f else "no arrest")
            else:
                row.append("correct (none)" if not tp_f else "false arrest")
        rows.append(row)
    story.append(KeepTogether([table(rows, S, [12 * mm, 28 * mm, 28 * mm, 28 * mm, 30 * mm, 28 * mm]),
                               Paragraph("Table 6a. Relative arrest-time error per left-out force (protocol a). f = 1.0 and "
                                         "f ≥ 1.65 require extrapolation in f.", S['caption'])]))
    mcA, nbA = fa['MasterCurve'], fa['NeighbourBaseline']
    P("Linear Koopman operators (parametric HDMD) decay smoothly and never produce the abrupt arrest, so they "
      "predict it far too early or not at all. The rollout-trained Koopman autoencoder drifts in free rollout and "
      "misses all 11 arrests. The master-curve Koopman model and the discovered equation predict all arrests, "
      f"with median errors of {pct(mcA['med'])} and {pct(fa['ODE-B']['med'])}, and no false arrest for f = 1.8 or 2.0. "
      f"The neighbour baseline attains a lower median arrest-time error ({pct(nbA['med'])}), because t<sub>arr</sub> "
      "varies smoothly with f between training forces. It is, however, itself a one-parameter collapse (a "
      "neighbouring trajectory stretched in time), so its success is further evidence for the common shape of the "
      "trajectories. It differs from the master-curve model in how τ(f) is supplied: by local interpolation "
      "rather than by the critical law. The critical law matters where interpolation is not available: at f = 1.0 "
      f"the baseline errs by {pct(abs(float(nbA['per_force']['1.0']['t_arr_rel_err'])))} against "
      f"{pct(abs(float(mcA['per_force']['1.0']['t_arr_rel_err'])))}, and at f = 1.8 the baseline predicts a spurious "
      f"arrest. The master-curve model also reproduces the slow stage more closely (RMSE {fnum(mcA['rmse'])} versus "
      f"{fnum(nbA['rmse'])} decades). The bootstrap intervals of the two arrest-time medians overlap; with 11 "
      "arresting forces neither model can be declared more accurate for interior forces.")
    rows = [["model"] + [f"{int(round(100 * float(p)))} % given" for p in FS['prefixes']]]
    for m in ('HDMD-prefix', 'DeepKoopman', 'MasterCurve', 'ODE-B'):
        rows.append([{'HDMD-prefix': 'HDMD on prefix (≈ HAVOK, zero forcing)'}.get(m, lab.get(m, m))] +
                    [f"{pct(FS['b'][m][p]['med'])} [{pct(FS['b'][m][p]['lo'])}, {pct(FS['b'][m][p]['hi'])}] "
                     f"({FS['b'][m][p]['missed']} missed)" for p in FS['prefixes']])
    story.append(KeepTogether([table(rows, S, [52 * mm, 42 * mm, 42 * mm, 42 * mm], highlight={3: GOOD}),
                               Paragraph("Table 6b. Protocol (b): median relative arrest-time error [95 % CI] when the first "
                                         "part of the unseen trajectory is given.", S['caption'])]))
    story.append(figure(os.path.join(FIG, "fig6_forecast.png"), W,
                        "Figure 6. Free-running prediction of trajectories never used in training. (a) f = 1.0, which requires "
                        "extrapolation in f, and (b) f = 1.4. (c) Predicted versus true arrest time for all left-out "
                        "forces (crosses at the top: no arrest predicted). (d) Median arrest-time error with 95 % bootstrap "
                        "bands as a function of the given prefix.", S))
    P("4.6 One-step-ahead versus free-running evaluation", 'h2')
    rows = [["f", "one-step-ahead RMSE", "free-running RMSE"]]
    for k, v in OS.items():
        rows.append([k, fnum(v['teacher_forced_rmse'], 3), fnum(v['free_running_rmse'], 2)])
    P("A sequence model can be evaluated by predicting one step ahead from the observed history at every step, or "
      "by running it freely from an initial window. The two can differ by orders of magnitude, so the protocol "
      "must match the question asked. For the force-conditioned autoencoder trained with a one-step loss "
      "(data/processed/koopman_model.pth), the one-step-ahead error on the original held-out forces is "
      "0.02–0.14 decades, whereas the free-running error over 10<super>4</super> time units is 3.5–11.7 decades "
      "(Table 7, Fig. 7). All prediction results in Section 4.5 are therefore free-running.")
    story.append(KeepTogether([table(rows, S, [20 * mm, 45 * mm, 45 * mm]),
                               Paragraph("Table 7. RMSE of log<sub>10</sub>(K + 10<super>−12</super>) over the first 10<super>4</super> "
                                         "time units for the one-step-trained autoencoder.", S['caption'])]))
    story.append(figure(os.path.join(FIG, "fig7_onestep_vs_free.png"), W,
                        "Figure 7. The same one-step-trained autoencoder evaluated one step ahead (dashed) and free-running "
                        "from the first window (red).", S))

    # ------------------------------------------------------------------ 5 validation
    story.append(PageBreak())
    P("5. Validation of the claims", 'h1')
    P("Each claim is listed with the test used to check it, the outcome, and a verdict. 'Supported' means the test "
      "could have failed and did not; 'Partly supported' means the claim holds only in a qualified form.")
    mcA, nbA = fa['MasterCurve'], fa['NeighbourBaseline']
    T = []
    T.append(["#", "claim", "test", "result", "verdict"])
    T += [
        ["1", "f = 1.65 arrests; f = 1.8 is still in the slow decay at the end of the run",
         "Arrest criterion K < 10<super>−20</super>; OLS slope of log K for t ≥ 5000 with moving-block bootstrap",
         f"t<sub>arr</sub>(1.65) = {fnum(taus['t_arr'][forces.index(1.65)], 0)}; slope(1.8) = {sci(v1['1.8']['slope'])} "
         f"[{sci(v1['1.8']['ci95'][0])}, {sci(v1['1.8']['ci95'][1])}], p < 0.001", "Supported"],
        ["2", "f<sub>c</sub> > 1.8", "Same as 1, plus extrapolation of the slow decay",
         "Requires the 1.8 decay to end in arrest (expected t ~ 1.5·10<super>5</super>, not simulated); f = 2.0 also drifts "
         f"slightly ({v1['2.0']['total_drop_decades']:.3f} decades)", "Partly supported"],
        ["3", "τ diverges faster than exponentially", "AIC: power law versus Arrhenius, 4 τ definitions",
         "ΔAIC = 11–26 in favour of a divergent form", "Supported"],
        ["4", "f<sub>c</sub> ≈ 1.8–1.95, γ ≈ 1.4–2.1", "Profile over f<sub>c</sub>, bootstrap over forces, fit-window variation",
         "Robust to window; f<sub>c</sub>–γ degenerate; power law ≈ VFT", "Partly supported (range only)"],
        ["5", "Master curve for 1.0 ≤ f ≤ 1.8",
         "Collapse R<super>2</super>; arrest predicted from slow-stage τ<sub>K</sub> alone; held-out prediction (Table 5)",
         f"R<super>2</super> = {1 - mc['cost_2param'] / mc['cost_raw']:.3f}; β = {v2['beta']:.2f} [{v2['beta_ci95'][0]:.2f}, "
         f"{v2['beta_ci95'][1]:.2f}]; CV(t<sub>arr</sub>/τ<sub>K</sub>) = {pct(v2['ratio_cv'])}", "Supported (β slightly < 1)"],
        ["6", "Force-independent final collapse rate", "Slope of log K after the cliff, 11 forces",
         f"{-Sc['cliff_rate']['mean']:.2f} ± {Sc['cliff_rate']['std']:.2f} decades per time unit", "Supported"],
        ["7", "Intermittent (non-Gaussian) forcing in arresting trajectories, Gaussian at f = 2.0",
         "19 phase-randomised surrogates per force; block-bootstrap CI; q = 20/40/80",
         "Data above all surrogates for all 11 arresting forces and f = 1.8; f = 2.0 consistent with surrogates",
         "Supported"],
        ["8", "Forcing precedes drops of K", "Causal hit rate versus random-time chance level",
         f"{v4['40']['dip_hit_over_chance_median']:.1f}× chance (median), stable over q", "Supported"],
        ["9", "A specific burst just before arrest", "Pre-arrest activity versus the preceding 500 time units",
         f"{v4['40']['prearrest_over_local_median']:.1f}× (median); clear only for f ≤ 1.45", "Partly supported"],
        ["10", "Reduced equation τ(f) dy/dt = g(y − a(f)) predicts unseen forces",
         "Leave-one-force-out integration; bootstrap of coefficients; polynomial degree; synthetic recovery",
         f"Median arrest error {pct(np.median(errs))}; structure kept in 89–99 % of bootstraps; recovery exact up to "
         "2× the data noise", "Supported as empirical model"],
        ["11", "dK/dt = −AK + BfK<super>n</super>", "Scan of n, sign of B, arrest-time error",
         "n not identifiable, B < 0, errors 50–200 %", "Not supported"],
        ["12", "Langevin model reproduces arrest statistics", "Simulated versus observed rescaled arrest times",
         "Median matches; spread overestimated", "Partly supported"],
        ["13", "Plain Koopman operators do not predict the arrest", "Free-running leave-one-out (Table 5)",
         f"Parametric HDMD {pct(fa['HDMD-param']['med'])}, autoencoder {pct(fa['DeepKoopman']['med'])} median error",
         "Supported"],
        ["14", "Scaling-based Koopman model predicts unseen forces",
         "Free-running leave-one-out; bootstrap CI; comparison with a neighbour baseline",
         f"{pct(mcA['med'])} [{pct(mcA['lo'])}, {pct(mcA['hi'])}], no missed or false arrests; time-stretched "
         f"neighbour baseline {pct(nbA['med'])} but one false arrest and larger errors under extrapolation",
         "Supported; no advantage over interpolation for interior forces"],
        ["15", "Method implementations are correct", "HAVOK on Lorenz; comparison with PyDMD; weak-SINDy recovery",
         f"Lorenz forcing precedes switches at {Hv['lorenz']['frac_switches_preceded_by_forcing'] / Hv['lorenz']['chance_level']:.1f}× chance; "
         "PyDMD eigenvalue agrees to < 1 %; correct terms recovered at 0.05–0.2 decades noise", "Supported"],
    ]
    hl = {}
    for i, r in enumerate(T[1:], 1):
        hl[i] = GOOD if r[4].startswith("Supported") else (WARN if r[4].startswith("Partly") else BAD)
    story.append(table(T, S, [7 * mm, 38 * mm, 44 * mm, 61 * mm, 24 * mm], zebra=False, highlight=hl))
    story.append(Paragraph("Table 8. Claims, tests and verdicts.", S['caption']))
    rows = [["delay window q", "median kurtosis (arresting)", "kurtosis f = 1.8", "kurtosis f = 2.0",
             "dips: hit/chance", "pre-arrest / local"]]
    for q in ('20', '40', '80'):
        v = v4[q]
        rows.append([q, fnum(v['kurtosis_jammed_median'], 1), fnum(v['kurtosis_1p8']), fnum(v['kurtosis_2p0']),
                     fnum(v['dip_hit_over_chance_median'], 1), fnum(v['prearrest_over_local_median'], 1)])
    story.append(KeepTogether([table(rows, S, [24 * mm, 34 * mm, 26 * mm, 26 * mm, 28 * mm, 28 * mm]),
                               Paragraph("Table 9. Sensitivity of the HAVOK results to the delay window.", S['caption'])]))
    story.append(figure(os.path.join(FIG, "figS1_method_validation.png"), W,
                        "Figure S1. Method validation. (a) HAVOK on the Lorenz system: the forcing (grey, active in red) is "
                        "concentrated at lobe switches of x(t). (b) Weak SINDy on synthetic data with the same correlated "
                        "noise as the simulations: coefficient error and fraction of correctly identified terms versus noise "
                        "amplitude (the data correspond to ≈ 0.1 decades). (c) γ and f<sub>c</sub> from τ<sub>K</sub> for six "
                        "choices of the slow-stage window.", S))

    # ------------------------------------------------------------------ 6 discussion
    P("6. Discussion", 'h1')
    P("The slow stage of the arresting trajectories is an almost exponential decay of K whose rate, 1/τ<sub>K</sub> = "
      f"1/{taus['tau_K'][0]:.0f} … 1/{taus['tau_K'][forces.index(1.8)]:.0f}, is two to four orders of magnitude smaller "
      f"than the post-arrest relaxation rate ({-Sc['cliff_rate']['mean'] * np.log(10):.2f}). In an energy balance "
      "dK/dt = −ΓK + Π<sub>act</sub>, the active power input therefore compensates all but a fraction "
      "1/(Γτ<sub>K</sub>) of the dissipation during the slow stage and vanishes once the system is trapped. As f "
      "approaches f<sub>c</sub> this compensation becomes complete, which is the origin of the diverging τ. The onset "
      "of intermittency at a fixed point of the rescaled decay, together with the bottleneck in g(u), suggests that "
      "arrest is preceded by a growing population of transient, partially arrested configurations.")
    P("These results are phenomenological. They characterise one global observable in one model system and do not "
      "replace a microscopic theory. They do provide quantitative targets for such theories: a diverging time scale "
      "with γ ≈ 1.4–2.1 (to be compared with mode-coupling-type predictions [8–11]), a common shape of the decay, a "
      "force-independent collapse rate, and intermittency that grows as arrest is approached.")
    P("Limitations", 'h3')
    story += bullets([
        "One trajectory per force; run-to-run scatter (the Langevin model suggests ±25 % in t<sub>arr</sub>) cannot be measured.",
        "One density, system size and persistence time, apparently from a single initial configuration; universality "
        "across systems is not tested.",
        "f<sub>c</sub> and γ are degenerate, and a power law cannot be distinguished from a VFT law; whether f = 1.8 and "
        "2.0 eventually arrest is not known.",
        "All validation is within this dataset (leave-one-force-out); no blind prediction of new simulations has been made.",
        "The quench-like regime f ≤ 0.75 is not covered by the scaling description."], S)

    P("7. Conclusions", 'h1')
    story.append(callout([
        Paragraph("<b>Q1 – Scaling.</b> Yes, within this dataset. For 1.0 ≤ f ≤ 1.8 the kinetic energy follows "
                  "log<sub>10</sub>K = a(f) + G(t/τ(f)) with a common shape G, an arrest at an almost fixed rescaled time, a "
                  "force-independent final collapse, and a time scale that diverges faster than exponentially "
                  "(f<sub>c</sub> ≈ 1.8–1.95, γ ≈ 1.4–2.1, degenerate).", S['box']),
        Spacer(1, 4),
        Paragraph("<b>Q2 – Koopman prediction.</b> Linear Hankel-DMD operators and a deep Koopman autoencoder, run "
                  "freely, do not predict unseen trajectories because they cannot generate the arrest. A Koopman model "
                  "built on the scaling structure does: it predicts the arrest time of left-out forces with a median "
                  f"error of {pct(mcA['med'])}, and of partially observed trajectories with "
                  f"{pct(FS['b']['MasterCurve'][FS['prefixes'][0]]['med'])} (10 % given) to "
                  f"{pct(FS['b']['MasterCurve'][FS['prefixes'][-1]]['med'])} (50 % given). For forces inside the "
                  "training range, a time-stretched copy of the nearest training trajectory, which is itself a "
                  f"one-parameter collapse, is at least as accurate ({pct(fa['NeighbourBaseline']['med'])}); the critical law "
                  "adds predictive value under extrapolation in f.", S['box']),
        Spacer(1, 4),
        Paragraph("<b>Equation and HAVOK.</b> The dynamics of log K are captured by the reduced equation "
                  "τ(f) dy/dt = g(y − a(f)), with a cubic g. HAVOK identifies intermittent forcing that is specific to "
                  "the arresting regime and switches on in the last part of the slow decay.", S['box'])], S, width=W))
    P("8. Recommended next steps", 'h2')
    story += bullets([
        "Several independent runs per force, to attach error bars to t<sub>arr</sub>, τ, f<sub>c</sub> and γ.",
        "A blind test: predict t<sub>arr</sub> for f = 1.7, 1.75, 1.85 and 1.9 with the reduced equation before simulating "
        "them, and run these forces for longer than 10<super>5</super> time units.",
        "A second system size and a second density or persistence time, to test whether G and the collapse rate are "
        "system-independent.",
        "Comparison of γ and the shape of G with active mode-coupling and random first-order transition predictions."], S)

    # ------------------------------------------------------------------ refs
    P("References", 'h1')
    refs = [
        "W. Kob and H. C. Andersen, Scaling behavior in the β-relaxation regime of a supercooled Lennard-Jones mixture, Phys. Rev. Lett. 73, 1376 (1994).",
        "S. L. Brunton, B. W. Brunton, J. L. Proctor, E. Kaiser and J. N. Kutz, Chaos as an intermittently forced linear system, Nat. Commun. 8, 19 (2017).",
        "S. L. Brunton, J. L. Proctor and J. N. Kutz, Discovering governing equations from data by sparse identification of nonlinear dynamical systems, PNAS 113, 3932 (2016).",
        "D. A. Messenger and D. M. Bortz, Weak SINDy: Galerkin-based data-driven model selection, Multiscale Model. Simul. 19, 1474 (2021).",
        "P. J. Schmid, Dynamic mode decomposition of numerical and experimental data, J. Fluid Mech. 656, 5 (2010).",
        "H. Arbabi and I. Mezić, Ergodic theory, dynamic mode decomposition, and computation of spectral properties of the Koopman operator, SIAM J. Appl. Dyn. Syst. 16, 2096 (2017).",
        "B. Lusch, J. N. Kutz and S. L. Brunton, Deep learning for universal linear embeddings of nonlinear dynamics, Nat. Commun. 9, 4950 (2018).",
        "L. Berthier and J. Kurchan, Non-equilibrium glass transitions in driven and active matter, Nat. Phys. 9, 310 (2013).",
        "G. Szamel, E. Flenner and L. Berthier, Glassy dynamics of athermal self-propelled particles: computer simulations and a nonequilibrium microscopic theory, Phys. Rev. E 91, 062304 (2015).",
        "S. K. Nandi, R. Mandal, P. J. Bhuyan, C. Dasgupta, M. Rao and N. S. Gov, A random first-order transition theory for an active glass, PNAS 115, 7688 (2018).",
        "W. Götze, Complex Dynamics of Glass-Forming Liquids: A Mode-Coupling Theory, Oxford University Press (2009).",
        "J. Theiler, S. Eubank, A. Longtin, B. Galdrikian and J. D. Farmer, Testing for nonlinearity in time series: the method of surrogate data, Physica D 58, 77 (1992).",
        "R. Friedrich, J. Peinke, M. Sahimi and M. R. R. Tabar, Approaching complexity by stochastic methods: from biological systems to turbulence, Phys. Rep. 506, 87 (2011).",
        "K. P. Burnham and D. R. Anderson, Model Selection and Multimodel Inference, 2nd ed., Springer (2002).",
        "J. N. Kutz, S. L. Brunton, B. W. Brunton and J. L. Proctor, Dynamic Mode Decomposition, SIAM (2016).",
        "J. Toner and Y. Tu, Long-range order in a two-dimensional dynamical XY model: how birds fly together, Phys. Rev. Lett. 75, 4326 (1995).",
        "M. C. Marchetti, J. F. Joanny, S. Ramaswamy, T. B. Liverpool, J. Prost, M. Rao and R. A. Simha, Hydrodynamics of soft active matter, Rev. Mod. Phys. 85, 1143 (2013).",
        "H. R. Künsch, The jackknife and the bootstrap for general stationary observations, Ann. Stat. 17, 1217 (1989).",
    ]
    for i, r in enumerate(refs, 1):
        story.append(Paragraph(f"[{i}] {r}", S['ref']))

    # ------------------------------------------------------------------ appendix
    P("Appendix A. Reproducibility", 'h1')
    P("All results are produced by the scripts in src/, run from the project root in this order: "
      "scaling_analysis.py, havok.py, equation_discovery.py, koopman_forecast.py (about 30 minutes on an NVIDIA "
      "RTX 3060), validation.py, report_figures.py. Numerical results are stored in data/processed/*.json and "
      "are read directly by reports/build_technical_report.py, which generates this document. Main settings: "
      "smoothing window 41 samples; slow stage t ∈ [20, 0.85 t<sub>arr</sub>]; HAVOK q = 40, rank 3–15 selected by "
      "stability and reconstruction, active threshold 2.5σ, look-back 20 time units; weak SINDy test-function width "
      "200 samples, stride 50, BIC on n/4 independent rows; forecasting grid Δt = 5, initial window 100 time units; "
      "autoencoder latent dimension 10, hidden width 64, 30 epochs, 40-step rollout loss; bootstrap resamples "
      "500–5000. Software: Python 3.13, NumPy, SciPy, PyTorch 2.7, PyDMD 2025.8 and PySINDy 2.1 (cross-checks).", 'small')

    doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=18 * mm,
                            bottomMargin=16 * mm, title=TITLE, author=AUTHOR,
                            subject="Scaling, Koopman prediction, HAVOK and reduced equations for an active glass")
    deco = page_decorator("Kinetic arrest in a dense active glass")
    doc.build(story, onFirstPage=deco, onLaterPages=deco)
    print("wrote", OUT)


if __name__ == "__main__":
    build()
