"""v3 figures: revised concept (Fig 1), preliminary analyses (Fig 2), decision policy (Fig 5),
testbed label (Fig 6), and schedule (Gantt). Helpers and styling come from figs.py."""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Polygon, Circle
import figs as F
from figs import (box, arrow, js_gauss, jonswap, BLUE, ORANGE, AQUA, INK, INK2, MUTED, GRID, NAVY,
                  LIGHTBLUE, LIGHTOR, LIGHTAQ, LIGHTGRAY, OUT)

STATS = {}

# ------------------------------------------------------------------ Fig 1 (revised wording)
def fig_concept():
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    fs = 7.3
    box(ax, 0.01, 0.30, 0.19, 0.62,
        "Shipboard setting\nno crew aboard\nconfined machinery\nspaces\nwave-induced roll\nand pitch\nseized fasteners,\ndegraded friction\nactuator wear\nlimited bandwidth",
        LIGHTGRAY, ec=MUTED, fs=fs)
    box(ax, 0.235, 0.30, 0.17, 0.62,
        "Robot acts\nvalve, fastener,\nand panel tasks\n\nRobot senses\ntask progress,\njoint torque,\nwrist wrench,\nship motion (IMU)",
        "white", ec=NAVY, fs=fs)
    box(ax, 0.44, 0.69, 0.27, 0.23, "Thrust 1: Influence\nDo my actions still produce\ntask progress, here and now?", LIGHTBLUE, fs=fs)
    box(ax, 0.44, 0.405, 0.27, 0.235, "Thrust 2: Attribution\nIf not, why? Strategy, actuation,\nresistance, disturbance,\nor an unknown cause", LIGHTOR, ec=ORANGE, fs=fs)
    box(ax, 0.44, 0.12, 0.27, 0.235, "Thrust 3: Decision\nPersist, probe, re-plan, or\nsuspend, with stated error\nrates; safety filter separate", LIGHTAQ, ec=AQUA, fs=fs)
    box(ax, 0.745, 0.30, 0.235, 0.62,
        "Outcomes\nfewer ineffective\npersistence events\nfewer unnecessary\nsuspensions\nforce limits enforced\nby a safety filter\ncompact cause reports\nidentifiability results",
        "white", ec=NAVY, fs=fs)
    arrow(ax, 0.202, 0.61, 0.233, 0.61)
    arrow(ax, 0.407, 0.80, 0.438, 0.80)
    arrow(ax, 0.575, 0.688, 0.575, 0.645)
    arrow(ax, 0.575, 0.403, 0.575, 0.36)
    arrow(ax, 0.712, 0.61, 0.743, 0.61)
    arrow(ax, 0.438, 0.20, 0.32, 0.297, color=AQUA, rad=-0.3)
    ax.text(0.215, 0.205, "probing and re-planning\nactions feed back", fontsize=6.9, color=INK2, ha="center", style="italic")
    ax.add_patch(FancyBboxPatch((0.03, 0.012), 0.94, 0.06, boxstyle="round,pad=0.006,rounding_size=0.015",
                                fc="#fff7e0", ec="#c98500", lw=0.8))
    ax.text(0.5, 0.042, "Inspiration from psychology: organisms judge control by comparing outcomes with and without their own action",
            ha="center", va="center", fontsize=7.0, color=INK)
    fig.savefig(os.path.join(OUT, "fig1_concept.png"))
    plt.close(fig)


# ------------------------------------------------------------------ valve simulator
U_P, TAU_K, CGAIN, S_NOISE, M_NOISE, W_NOISE = 4.0, 1.5, 0.05, 0.002, 0.15, 0.05


def simulate(rng, a_s_fn, tau_s_fn, T, dt=0.2):
    n = int(T / dt); t = np.arange(n) * dt
    a_s = a_s_fn(t); tau_s = tau_s_fn(t)
    act = rng.integers(0, 2, n)
    d = a_s * np.sin(rng.uniform(0, 2 * np.pi, n)) + rng.normal(0, 0.1, n)
    dhat = d + rng.normal(0, M_NOISE, n)
    tau = act * U_P + d
    move = np.abs(tau) > tau_s
    dth = np.where(move, np.sign(tau) * (np.abs(tau) - TAU_K) * CGAIN, 0.0) + rng.normal(0, S_NOISE, n)
    wrench = act * U_P + 0.3 * d + rng.normal(0, W_NOISE, n)          # wrist reaction torque
    return dict(t=t, act=act, d=d, dhat=dhat, move=move, dth=dth, wrench=wrench)


def cond_js(y, a, dh, floor):
    edges = np.quantile(dh, np.linspace(0, 1, 7)); vals, wts = [], []
    for b in range(6):
        mb = (dh >= edges[b]) & (dh <= edges[b + 1])
        ya, y0 = y[mb & (a == 1)], y[mb & (a == 0)]
        if len(ya) < 2 or len(y0) < 2:
            continue
        vals.append(js_gauss(ya.mean(), max(ya.std(), floor), y0.mean(), max(y0.std(), floor))); wts.append(mb.sum())
    return float(np.average(vals, weights=wts)) if vals else np.nan


def monitors(ep, win=100, step=5, t_from=20.0):
    t, act, dth, dhat, wr = ep["t"], ep["act"], ep["dth"], ep["dhat"], ep["wrench"]
    nominal = (U_P - TAU_K) * CGAIN
    out = {k: [] for k in ("t", "prog", "unc", "cond", "wrench")}
    for i in range(win, len(t) + 1, step):
        if t[i - 1] < t_from:
            continue
        sl = slice(i - win, i); a_, y_, dh_ = act[sl], dth[sl], dhat[sl]
        ya, y0 = y_[a_ == 1], y_[a_ == 0]
        out["t"].append(t[i - 1])
        out["prog"].append(np.mean(ya > 0.5 * nominal))
        out["unc"].append(js_gauss(ya.mean(), ya.std(), y0.mean(), y0.std()))
        out["cond"].append(cond_js(y_, a_, dh_, S_NOISE))
        out["wrench"].append(cond_js(wr[sl], a_, dh_, W_NOISE))
    return {k: np.array(v) for k, v in out.items()}


def fit_b6_predictor(seed=12, n=150, nb=20):
    """B6 analog: disturbance-conditioned progress predictor, fit on separate seizure-free episodes."""
    rng = np.random.default_rng(seed); eps = []
    for _ in range(n):
        a = rng.uniform(2.0, 3.5)
        eps.append(simulate(rng, lambda t, a=a: np.full_like(t, a), lambda t: np.full_like(t, 2.0), 140))
    act = np.concatenate([e["act"] for e in eps]); y = np.concatenate([e["dth"] for e in eps]); d = np.concatenate([e["dhat"] for e in eps])
    m = act == 1; edges = np.quantile(d[m], np.linspace(0, 1, nb + 1)); edges[0], edges[-1] = -np.inf, np.inf
    mu = np.array([y[m & (d >= edges[b]) & (d < edges[b + 1])].mean() for b in range(nb)])
    return edges, mu


def b6_monitor(ep, pred, win=100, step=5, t_from=20.0):
    """Observed progress on action trials divided by the predictor's expected progress, in sliding windows."""
    edges, mu = pred; t, act, dth, dhat = ep["t"], ep["act"], ep["dth"], ep["dhat"]
    yhat = mu[np.clip(np.searchsorted(edges, dhat, side="right") - 1, 0, len(mu) - 1)]
    out_t, out_v = [], []
    for i in range(win, len(t) + 1, step):
        if t[i - 1] < t_from:
            continue
        sl = slice(i - win, i); a_ = act[sl] == 1
        out_t.append(t[i - 1]); out_v.append(dth[sl][a_].sum() / max(yhat[sl][a_].sum(), 1e-9))
    return {"t": np.array(out_t), "b6": np.array(out_v)}


def fig_prelim(n_cal=150, n_test=150):
    rng = np.random.default_rng(11)
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.75), gridspec_kw={"width_ratios": [1.3, 1], "wspace": 0.30})
    # (a) single illustrative episode: calm, rough-free, rough-seizing
    ep = simulate(rng, lambda t: np.where(t < 60, 0.3, 3.0),
                  lambda t: np.where(t < 120, 2.0, np.clip(2.0 + (t - 120) / 30 * 6.0, 2.0, 8.0)), 180)
    m = monitors(ep)
    ax = axs[0]
    for x0, x1, lab, fc in ((0, 60, "calm sea,\nfree valve", "#ffffff"), (60, 120, "rough sea,\nfree valve", LIGHTGRAY),
                           (120, 180, "rough sea,\nvalve seizing", "#fbe9e1")):
        ax.axvspan(x0, x1, color=fc, lw=0, zorder=0)
        ax.text((x0 + x1) / 2, 1.16, lab, ha="center", va="center", fontsize=6.9, color=INK2)
    am = (ep["act"] == 1) & ep["move"]; t_loss = float(ep["t"][am][-1])
    ax.axvline(t_loss, color=INK2, lw=0.8, ls=":")
    ax.plot(m["t"], m["wrench"], color=AQUA, lw=1.4, ls=(0, (4, 2)), label="influence on contact wrench", zorder=2)
    ax.plot(m["t"], m["prog"], color=MUTED, lw=1.3, label="progress monitor", zorder=3)
    ax.plot(m["t"], m["unc"], color=ORANGE, lw=1.4, label="progress influence, unconditioned", zorder=4)
    ax.plot(m["t"], m["cond"], color=BLUE, lw=1.8, label="progress influence, conditioned", zorder=5)
    mb6 = b6_monitor(ep, fit_b6_predictor())
    ax.plot(mb6["t"], mb6["b6"], color=INK, lw=1.2, ls=(0, (3, 2)), label="progress predictor (B6 analog)", zorder=6)
    STATS["ep_b6_phase2_mean"] = float(np.mean(mb6["b6"][(mb6["t"] > 80) & (mb6["t"] < 120)]))
    STATS["ep_b6_range"] = [float(mb6["b6"].min()), float(mb6["b6"].max())]
    ax.set_xlim(20, 180); ax.set_ylim(-0.30, 1.25); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("time (s)"); ax.set_ylabel("monitor value")
    ax.set_title("(a) One simulated valve episode", loc="left", weight="bold", pad=16)
    ax.legend(frameon=True, framealpha=0.95, edgecolor="none", loc="lower left", fontsize=6.4, handlelength=1.8)
    m3 = m["t"] > t_loss + 5
    STATS["ep_t_loss"] = t_loss
    STATS["ep_wrench_after_loss_mean"] = float(np.nanmean(m["wrench"][m3]))
    STATS["ep_cond_after_loss_mean"] = float(np.nanmean(m["cond"][m3]))
    m2 = (m["t"] > 80) & (m["t"] < 120)
    STATS["ep_phase2"] = {k: float(np.nanmean(m[k][m2])) for k in ("prog", "unc", "cond", "wrench")}

    # (b) randomized episodes, thresholds calibrated on separate seizure-free episodes
    keys = ("prog", "unc", "cond", "wrench")
    cal_min = {k: [] for k in keys}
    for _ in range(n_cal):
        a = rng.uniform(2.0, 3.5)
        e = simulate(rng, lambda t, a=a: np.full_like(t, a), lambda t: np.full_like(t, 2.0), 140)
        mm = monitors(e)
        for k in keys:
            cal_min[k].append(np.nanmin(mm[k]))
    tests = []
    for _ in range(n_test):
        a = rng.uniform(2.0, 3.5); t_on = rng.uniform(50, 80)
        e = simulate(rng, lambda t, a=a: np.full_like(t, a),
                     lambda t, t_on=t_on: np.where(t < t_on, 2.0, np.clip(2.0 + (t - t_on) / 30 * 6.0, 2.0, 8.0)), t_on + 70)
        am = (e["act"] == 1) & e["move"]
        t_loss = float(e["t"][am][-1])
        tests.append((t_on, t_loss, monitors(e)))
    targets = np.array([0.02, 0.05, 0.10, 0.20, 0.30])
    res = {}
    for k in keys:
        mins = np.array(cal_min[k]); res[k] = []
        for r in targets:
            thr = np.quantile(mins, r)
            delays, prem, miss, fa = [], 0, 0, 0
            for t_on, t_loss, mm in tests:
                below = np.where(mm[k] < thr)[0]
                if len(below) == 0:
                    miss += 1; continue
                ta = mm["t"][below[0]]
                if ta < t_on:
                    fa += 1; continue
                delays.append(ta - t_on)
                if ta < t_loss:
                    prem += 1
            res[k].append(dict(fa_target=float(r), thr=float(thr), median_delay=float(np.median(delays)) if delays else None,
                               q25=float(np.percentile(delays, 25)) if delays else None, q75=float(np.percentile(delays, 75)) if delays else None,
                               miss=miss / n_test, premature=prem / n_test, test_fa=fa / n_test))
    STATS["tradeoff"] = res
    ax = axs[1]
    styles = {"prog": (MUTED, "progress monitor"), "unc": (ORANGE, "progress influence, unconditioned"),
              "cond": (BLUE, "progress influence, conditioned")}
    loss_after_onset = np.array([tl - to for to, tl, _ in tests])
    STATS["loss_after_onset_median"] = float(np.median(loss_after_onset))
    STATS["loss_after_onset_iqr"] = [float(np.percentile(loss_after_onset, 25)), float(np.percentile(loss_after_onset, 75))]
    ax.axhspan(np.percentile(loss_after_onset, 25), np.percentile(loss_after_onset, 75), color=LIGHTGRAY, lw=0, zorder=0)
    ax.axhline(np.median(loss_after_onset), color=INK2, lw=0.8, ls=":")
    ax.text(0.315, np.percentile(loss_after_onset, 75) + 0.8, "valve can no longer be turned in\nany motion phase (median, IQR)", fontsize=6.2, color=INK2, va="bottom", ha="right")
    for k, (col, lab) in styles.items():
        xs = [r["fa_target"] for r in res[k] if r["median_delay"] is not None]
        ys = [r["median_delay"] for r in res[k] if r["median_delay"] is not None]
        lo = [r["q25"] for r in res[k] if r["median_delay"] is not None]; hi = [r["q75"] for r in res[k] if r["median_delay"] is not None]
        ax.fill_between(xs, lo, hi, color=col, alpha=0.15, lw=0)
        ax.plot(xs, ys, color=col, lw=1.6, marker="o", ms=4.5, mfc="white", mec=col, mew=1.2, label=lab)
    b6p = os.path.join(OUT, "stats_b6.json")
    if os.path.exists(b6p):  # B6 analog from b6_compare.py, computed on the same 150 test episodes
        b6 = json.load(open(b6p))["b6"]
        ax.plot([r["fa_target"] for r in b6], [r["median"] for r in b6], color=INK, lw=1.3, ls=(0, (3, 2)), marker="D", ms=3.2,
                mfc=INK, mec=INK, zorder=6)
        ax.text(0.315, 13.6, "dashed: disturbance-conditioned\nprogress predictor (B6 analog)", ha="right", va="bottom", fontsize=6.3, color=INK)
    wmiss = [r["miss"] for r in res["wrench"]]
    STATS["wrench_miss_range"] = [float(min(wmiss)), float(max(wmiss))]
    ax.text(0.315, 3.0, "influence on contact wrench stays near 1 bit\n(uninformative about seizure by construction)", ha="right",
            va="bottom", fontsize=6.3, color=INK2)
    ax.set_xlabel("false-alarm rate per seizure-free episode")
    ax.set_ylabel("detection delay after\nseizure onset (s)")
    ax.set_xlim(0, 0.32); ax.set_ylim(0, 52)
    ax.set_title("(b) 150 randomized episodes", loc="left", weight="bold", pad=16)
    fig.savefig(os.path.join(OUT, "fig2_prelim.png"))
    plt.close(fig)


# ------------------------------------------------------------------ Fig 5: decisions with anytime-valid betting tests
def betting_test(rng, mu, a=0.05, b=0.05, d_lo=0.0, d_hi=0.6, B=2.1, sd=0.6, clip=1.5, nmax=80):
    m0 = (d_lo + B) / (2 * B); m1 = (d_hi + B) / (2 * B)
    lk, lk2 = 0.0, 0.0; xs = []; pathA, pathB = [0.0], [0.0]
    for i in range(1, nmax + 1):
        if xs:
            mh = np.mean(xs); vh = np.var(xs) + 1e-3
        else:
            mh, vh = 0.5, 0.05
        lam = np.clip((mh - m0) / (vh + (mh - m0) ** 2), 0, 0.5 / m0)
        lam2 = np.clip((m1 - mh) / (vh + (m1 - mh) ** 2), 0, 0.5 / (1 - m1))
        z = mu + np.clip(rng.normal(0, sd), -clip, clip)
        x = (z + B) / (2 * B); xs.append(x)
        lk += np.log1p(lam * (x - m0)); lk2 += np.log1p(-lam2 * (x - m1))
        pathA.append(lk); pathB.append(lk2)
        if lk >= np.log(1 / a):
            return "persist", i, pathA, pathB
        if lk2 >= np.log(1 / b):
            return "stop", i, pathA, pathB
    return "undecided", nmax, pathA, pathB


def fig_t3():
    rng = np.random.default_rng(5)
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.5), gridspec_kw={"width_ratios": [1, 1.15], "wspace": 0.32})
    ax = axs[0]; dlo, dhi = 0.40, 0.60; I = np.linspace(0, 1, 400); umax = 0.5
    per = np.clip(I - dlo, 0, umax); stp = np.clip(dhi - I, 0, umax)
    ax.fill_between(I, 0, per, color=AQUA, alpha=0.30, lw=0)
    ax.fill_between(I, 0, stp, color=ORANGE, alpha=0.28, lw=0)
    ax.fill_between(I, np.maximum(per, stp), umax, color=BLUE, alpha=0.20, lw=0)
    ax.plot(I, np.maximum(I - dlo, dhi - I), color=INK2, lw=1.0)
    ax.text(0.86, 0.05, "PERSIST\n$L_t>\\delta_{\\mathrm{lo}}+\\eta$", ha="center", fontsize=6.9, color=INK)
    ax.text(0.15, 0.05, "ABANDON\nACTION\n$U_t<\\delta_{\\mathrm{hi}}-\\eta$", ha="center", fontsize=6.9, color=INK)
    ax.text(0.5, 0.33, "PROBE\n(neither test\nhas decided)", ha="center", fontsize=7.2, color=INK)
    ax.text(0.5, 0.012, "either", ha="center", fontsize=5.8, color=INK2)
    ax.set_xlim(0, 1); ax.set_ylim(0, umax)
    ax.set_xlabel("center of confidence sequence for $\\mu$ (normalized)")
    ax.set_ylabel("half-width of confidence sequence")
    for v, lab in ((dlo, "$\\delta_{\\mathrm{lo}}+\\eta$"), (dhi, "$\\delta_{\\mathrm{hi}}-\\eta$")):
        ax.axvline(v, color=MUTED, lw=0.8, ls=":"); ax.text(v + (0.01 if v > 0.5 else -0.01), umax * 0.97, lab, fontsize=8, va="top", ha="left" if v > 0.5 else "right", color=INK2)
    ax.set_title("(a) Decision regions", loc="left", weight="bold"); ax.grid(False)

    ax = axs[1]
    thr = np.log(20)
    for mu, col in ((0.6, BLUE), (0.0, ORANGE)):
        for ep in range(5):
            dec, n, pA, pB = betting_test(rng, mu)
            path = pA if mu > 0 else pB
            ax.plot(range(len(path)), path, color=col, lw=1.1, alpha=0.9)
            ax.plot(len(path) - 1, path[-1], "o", ms=3.5, color=col)
    ax.axhline(thr, color=INK2, lw=1.1, ls="--")
    ax.text(0.5, thr + 0.15, "decision threshold $\\ln(1/\\alpha)$", ha="left", va="bottom", fontsize=7.0, color=INK)
    ax.plot([], [], color=BLUE, label="useful progress: persist test")
    ax.plot([], [], color=ORANGE, label="no useful progress: stop test")
    ax.legend(frameon=False, loc="upper left", fontsize=6.8)
    ax.set_xlim(-0.5, 30); ax.set_ylim(-1.0, 5.2)
    ax.set_xlabel("number of paired safe probes $n$"); ax.set_ylabel("log test wealth")
    ax.set_title("(b) Anytime-valid sequential tests", loc="left", weight="bold")
    # Monte Carlo error rates
    out = {}
    for name, mu in (("useful", 0.6), ("none", 0.0), ("indifference", 0.3)):
        decs, ns = [], []
        for _ in range(4000):
            d, n, _, _ = betting_test(rng, mu)
            decs.append(d); ns.append(n)
        decs = np.array(decs); ns = np.array(ns)
        out[name] = dict(persist=float(np.mean(decs == "persist")), stop=float(np.mean(decs == "stop")),
                         undecided=float(np.mean(decs == "undecided")), median_n=float(np.median(ns)), mean_n=float(np.mean(ns)))
    STATS["betting"] = out
    fig.savefig(os.path.join(OUT, "fig5_t3.png"))
    plt.close(fig)


# ------------------------------------------------------------------ Fig 6 relabel (cradle)
def fig_testbed():
    src = open(os.path.join(os.path.dirname(__file__), "figs.py")).read()
    src = src.replace('"commercial two-axis roll/pitch platform"', '"two-axis roll/pitch cradle (built in IDEELab)"')
    ns = {"__name__": "figs_tmp", "__file__": os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs.py")}
    exec(compile(src, "figs_tmp", "exec"), ns)
    ns["rng"] = np.random.default_rng(7)
    ns["fig_testbed"]()


# ------------------------------------------------------------------ Gantt
def fig_gantt():
    tasks = [
        ("T1.1 Influence and progress advantage: theory", 1, 4, BLUE),
        ("T1.2 Estimation and calibration under contact", 3, 8, BLUE),
        ("T1.3 Paired, randomized safe probes", 4, 8, BLUE),
        ("T2.1 Cause models and identifiability", 3, 7, ORANGE),
        ("T2.2 Budgeted, decision-aware probing", 6, 10, ORANGE),
        ("T3.1 Decision policy and anytime-valid tests", 5, 9, AQUA),
        ("T3.2 Robustness: misspecification, drift", 9, 11, AQUA),
        ("E1 Digital twin with roll/pitch conditions", 1, 4, MUTED),
        ("E2 Motion cradle: design, build, acceptance", 1, 2, MUTED),
        ("E3 Testbed integration and pilot trials", 3, 6, MUTED),
        ("E4 Hardware evaluation campaigns", 6, 12, MUTED),
        ("D  Open benchmark, code, and data release", 10, 12, MUTED),
    ]
    miles = [(2, "M1"), (4, "M2"), (6, "M3"), (8, "M4"), (10, "M5"), (12, "M6")]
    fig, ax = plt.subplots(figsize=(6.6, 2.55))
    for i, (lab, s, e, col) in enumerate(tasks):
        ax.barh(i, e - s + 1, left=s - 0.5, height=0.62, color=col, alpha=0.85 if col != MUTED else 0.55, edgecolor="white", lw=1.5)
    ax.set_yticks(range(len(tasks))); ax.set_yticklabels([t[0] for t in tasks], fontsize=7.2)
    ax.invert_yaxis(); ax.set_xlim(0.5, 12.5)
    ax.set_xticks(range(1, 13)); ax.set_xticklabels([f"Q{((q - 1) % 4) + 1}" for q in range(1, 13)], fontsize=7.2)
    for yr, x in ((1, 2.5), (2, 6.5), (3, 10.5)):
        ax.text(x, -1.65, f"Year {yr}  ({2026 + yr}–{2027 + yr})", ha="center", fontsize=7.6, weight="bold", color=INK)
    for x in (4.5, 8.5):
        ax.axvline(x, color=MUTED, lw=0.8)
    for q, mlab in miles:
        ax.plot(q + 0.35, len(tasks) - 0.1, marker="D", ms=5.5, color=NAVY, clip_on=False)
        ax.text(q + 0.35, len(tasks) + 0.55, mlab, ha="center", fontsize=7.2, color=NAVY, weight="bold")
    ax.plot(2.45, 8, marker="*", ms=9, color=ORANGE, zorder=5)
    ax.text(2.75, 8, "go/no-go gate (month 6)", va="center", fontsize=6.6, color=INK2)
    for q in (4, 8, 12):
        ax.plot(q + 0.35, -1.05, marker="v", ms=5.5, color=ORANGE, clip_on=False)
    ax.text(12.6, -1.05, "ONR review", fontsize=6.8, color=INK2, va="center")
    ax.set_ylim(len(tasks) + 0.9, -2.0)
    ax.grid(axis="y", visible=False); ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    fig.savefig(os.path.join(OUT, "fig7_gantt.png"))
    plt.close(fig)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "prelim":
        old = json.load(open(os.path.join(OUT, "stats_v3.json"))); STATS.update(old); fig_prelim()
    elif len(sys.argv) > 1 and sys.argv[1] == "light":
        old = json.load(open(os.path.join(OUT, "stats_v3.json"))); STATS.update(old); fig_concept(); fig_gantt()
        import numpy as _np
        fig_t3()
    else:
        fig_concept(); fig_prelim(); fig_t3(); fig_testbed(); fig_gantt()
    json.dump(STATS, open(os.path.join(OUT, "stats_v3.json"), "w"), indent=2)
    print(json.dumps(STATS, indent=1))
