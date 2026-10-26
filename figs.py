"""Figures for the ONR YIP technical proposal (Korivand).
All quantitative panels are computed here from the stated models; nothing is hand-drawn data."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Polygon, Circle, Wedge
from scipy.linalg import sqrtm
import os, json

OUT = os.path.join(os.path.dirname(__file__), "figs")
os.makedirs(OUT, exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
NAVY = "#1f3864"
LIGHTBLUE, LIGHTOR, LIGHTAQ, LIGHTGRAY = "#dbe8f8", "#fbe3d8", "#d6f2e7", "#f1f0ed"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Liberation Serif", "TeX Gyre Termes", "Times New Roman"],
    "mathtext.fontset": "stix", "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 1.6,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})
rng = np.random.default_rng(7)
STATS = {}


def box(ax, x, y, w, h, text, fc, ec=NAVY, fs=8.5, bold_first=True, lw=1.0, ha="center"):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.025", fc=fc, ec=ec, lw=lw)
    ax.add_patch(p)
    lines = text.split("\n")
    if bold_first and len(lines) > 1:
        ax.text(x + w / 2 if ha == "center" else x + 0.012, y + h - 0.035, lines[0], ha=ha, va="top",
                fontsize=fs + 0.5, weight="bold", color=INK)
        ax.text(x + w / 2 if ha == "center" else x + 0.012, y + h - 0.035 - 0.065, "\n".join(lines[1:]),
                ha=ha, va="top", fontsize=fs, color=INK, linespacing=1.25)
    else:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, linespacing=1.25)
    return p


def arrow(ax, x0, y0, x1, y1, color=NAVY, lw=1.3, style="-|>", rad=0.0, ls="-"):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=10, color=color, lw=lw,
                        connectionstyle=f"arc3,rad={rad}", linestyle=ls)
    ax.add_patch(a)


def js_gauss(m1, s1, m2, s2, n=4001):
    """Jensen-Shannon divergence (bits) between two 1-D Gaussians by numerical integration."""
    s1, s2 = max(s1, 1e-9), max(s2, 1e-9)
    lo = min(m1 - 8 * s1, m2 - 8 * s2); hi = max(m1 + 8 * s1, m2 + 8 * s2)
    x = np.linspace(lo, hi, n); dx = x[1] - x[0]
    p = np.exp(-0.5 * ((x - m1) / s1) ** 2) / (s1 * np.sqrt(2 * np.pi))
    q = np.exp(-0.5 * ((x - m2) / s2) ** 2) / (s2 * np.sqrt(2 * np.pi))
    m = 0.5 * (p + q)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(p > 0, p * np.log2(p / m), 0.0)
        b = np.where(q > 0, q * np.log2(q / m), 0.0)
    return float(np.clip(0.5 * (a.sum() + b.sum()) * dx, 0, 1))


# ---------------------------------------------------------------- Fig 1: concept
def fig_concept():
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    fs = 7.3
    box(ax, 0.01, 0.30, 0.19, 0.62,
        "Shipboard setting\nno crew aboard\nconfined machinery\nspaces\nwave-induced roll\nand pitch\nseized fasteners,\ndegraded friction\nactuator wear\nlimited bandwidth",
        LIGHTGRAY, ec=MUTED, fs=fs)
    box(ax, 0.235, 0.30, 0.17, 0.62,
        "Robot acts\nvalve, fastener,\nand panel tasks\n\nRobot senses\njoint torque,\nwrist wrench,\ntask progress,\nship motion (IMU)",
        "white", ec=NAVY, fs=fs)
    box(ax, 0.44, 0.69, 0.27, 0.23, "Thrust 1: Action influence\nDo my actions still change\nthe task outcome, here and now?", LIGHTBLUE, fs=fs)
    box(ax, 0.44, 0.405, 0.27, 0.235, "Thrust 2: Attribution\nIf not, why? Strategy, actuation,\nresistance, disturbance,\nor an unknown cause", LIGHTOR, ec=ORANGE, fs=fs)
    box(ax, 0.44, 0.12, 0.27, 0.235, "Thrust 3: Decision\nPersist, probe, re-plan,\nor suspend, with bounded\nerror rates", LIGHTAQ, ec=AQUA, fs=fs)
    box(ax, 0.745, 0.30, 0.235, 0.62,
        "Outcomes\nfewer unsafe persistence\nevents\nfewer needless\nabandonments\ncompact cause reports\nfor low-bandwidth links\ntheory linking control-\nlability, diagnosis,\nand decision",
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


# ---------------------------------------------------------------- Fig 2: preliminary analysis
def lin_gauss_case(g, sigma_d, H=20, dt=0.05, rho=1.0, n_mc=60000):
    m, k, c = 1.0, 4.0, 0.8
    Ac = np.array([[0, 1], [-k / m, -c / m]])
    A = np.eye(2) + dt * Ac
    B = g * np.array([[0.0], [dt / m]])
    E = np.array([[0.0], [dt]])
    C = np.array([[1.0, 0.0]])
    Q = (E @ E.T) * sigma_d ** 2
    Rn = 1e-8
    RH = np.hstack([np.linalg.matrix_power(A, H - 1 - j) @ B for j in range(H)])
    Sig = C @ sum(np.linalg.matrix_power(A, j) @ Q @ np.linalg.matrix_power(A, j).T for j in range(H)) @ C.T + Rn
    W = C @ RH @ RH.T @ C.T
    Sm = np.linalg.inv(sqrtm(Sig)).real
    I_closed = 0.5 * rho ** 2 * np.max(np.linalg.eigvalsh(Sm @ W @ Sm))
    # Monte Carlo with optimal input direction
    M = C @ RH
    _, _, Vt = np.linalg.svd(M)
    U = rho * Vt[0]
    ys = {}
    for lab, u in (("a", U), ("0", np.zeros(H))):
        x = np.zeros((n_mc, 2))
        for j in range(H):
            w = rng.normal(0, sigma_d, n_mc)
            x = x @ A.T + (B[:, 0] * u[j])[None, :] + np.outer(w, E[:, 0])
        ys[lab] = x[:, 0] + rng.normal(0, np.sqrt(Rn), n_mc)
    mu1, mu0 = ys["a"].mean(), ys["0"].mean()
    v1, v0 = ys["a"].var(), ys["0"].var()
    I_mc = 0.5 * (np.log(v0 / v1) + (v1 + (mu1 - mu0) ** 2) / v0 - 1)
    return I_closed, I_mc


def valve_episode():
    T, dt_tr = 180.0, 0.2
    n = int(T / dt_tr)
    t = np.arange(n) * dt_tr
    u_p, tau_k, cgain, s_noise, m_noise = 4.0, 1.5, 0.05, 0.002, 0.15
    a_s = np.where(t < 60, 0.3, 3.0)
    tau_s = np.where(t < 120, 2.0, np.clip(2.0 + (t - 120) / 30 * 6.0, 2.0, 8.0))
    act = rng.integers(0, 2, n)
    d = a_s * np.sin(rng.uniform(0, 2 * np.pi, n)) + rng.normal(0, 0.1, n)
    dhat = d + rng.normal(0, m_noise, n)
    tau = act * u_p + d
    move = np.abs(tau) > tau_s
    dth = np.where(move, np.sign(tau) * (np.abs(tau) - tau_k) * cgain, 0.0) + rng.normal(0, s_noise, n)
    nominal = (u_p - tau_k) * cgain
    win = 100
    times, prog, js_u, js_c = [], [], [], []
    for i in range(win, n + 1, 5):
        sl = slice(i - win, i)
        a_, y_, dh_ = act[sl], dth[sl], dhat[sl]
        ya, y0 = y_[a_ == 1], y_[a_ == 0]
        prog.append(np.mean(ya > 0.5 * nominal))
        js_u.append(js_gauss(ya.mean(), ya.std(), y0.mean(), y0.std()))
        # disturbance-conditioned: stratify by measured disturbance
        edges = np.quantile(dh_, np.linspace(0, 1, 7))
        vals, wts = [], []
        for b in range(6):
            mb = (dh_ >= edges[b]) & (dh_ <= edges[b + 1])
            yb_a, yb_0 = y_[mb & (a_ == 1)], y_[mb & (a_ == 0)]
            if len(yb_a) < 2 or len(yb_0) < 2:
                continue
            vals.append(js_gauss(yb_a.mean(), max(yb_a.std(), s_noise), yb_0.mean(), max(yb_0.std(), s_noise)))
            wts.append(mb.sum())
        js_c.append(np.average(vals, weights=wts))
        times.append(t[i - 1])
    am = (act == 1) & move
    STATS["valve_last_action_move_t"] = float(t[am][-1])
    STATS["valve_last_action_move_t_rough_free_minfrac"] = None
    mid = (t > 132) & (t < 142) & (act == 1)
    STATS["valve_action_success_frac_132_142"] = float(np.mean(move[mid]))
    return np.array(times), np.array(prog), np.array(js_u), np.array(js_c)


def fig_prelim():
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.55), gridspec_kw={"width_ratios": [1, 1.35], "wspace": 0.32})
    ax = axs[0]
    sig = np.geomspace(0.05, 2.0, 9)
    rel_err = []
    for g, col, lab in ((1.0, BLUE, "actuator gain 1.0"), (0.6, ORANGE, "gain 0.6"), (0.3, AQUA, "gain 0.3")):
        cl, mc = zip(*[lin_gauss_case(g, s) for s in sig])
        cl, mc = np.array(cl), np.array(mc)
        rel_err += list(np.abs(mc - cl) / cl)
        ax.plot(sig, cl, color=col, lw=1.6, label=lab)
        ax.plot(sig, mc, "o", ms=4.2, mfc="white", mec=col, mew=1.2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("disturbance intensity $\\sigma_d$")
    ax.set_ylabel("max. influence $\\mathcal{I}^{\\star}$ (nats)")
    ax.set_title("(a) Linear-Gaussian check", loc="left", weight="bold")
    ax.legend(frameon=False, loc="lower left", handlelength=1.4)
    ax.text(0.98, 0.97, "lines: closed form\nmarkers: Monte Carlo", transform=ax.transAxes, ha="right", va="top", fontsize=7.2, color=INK2)
    STATS["lg_max_rel_err"] = float(np.max(rel_err)); STATS["lg_med_rel_err"] = float(np.median(rel_err))

    ax = axs[1]
    tt, pr, ju, jc = valve_episode()
    for x0, x1, lab, fc in ((0, 60, "calm sea,\nfree valve", "#ffffff"), (60, 120, "rough sea,\nfree valve", LIGHTGRAY),
                           (120, 180, "rough sea,\nvalve seizing", "#fbe9e1")):
        ax.axvspan(x0, x1, color=fc, lw=0, zorder=0)
        ax.text((x0 + x1) / 2, 1.13, lab, ha="center", va="center", fontsize=7.2, color=INK2)
    ax.axhline(0.5, color=MUTED, lw=0.9, ls="--", zorder=1)
    ax.text(22, 0.535, "alarm threshold", ha="left", fontsize=7, color=INK2)
    ax.plot(tt, pr, color=MUTED, lw=1.3, label="progress monitor", zorder=2)
    ax.plot(tt, ju, color=ORANGE, lw=1.5, label="influence, unconditioned", zorder=3)
    ax.plot(tt, jc, color=BLUE, lw=1.8, label="influence, disturbance-conditioned", zorder=4)
    ax.set_xlim(20, 180); ax.set_ylim(-0.03, 1.22)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("time (s)"); ax.set_ylabel("monitor value")
    ax.set_title("(b) Simulated valve episode", loc="left", weight="bold", pad=16)
    ax.legend(frameon=True, framealpha=0.95, edgecolor="none", loc="lower left", fontsize=7.2, handlelength=1.4)
    m2 = (tt > 80) & (tt < 120); m3 = tt > 155
    STATS["valve_phase2_mean"] = {"progress": float(pr[m2].mean()), "uncond": float(ju[m2].mean()), "cond": float(jc[m2].mean())}
    STATS["valve_phase3_end"] = {"progress": float(pr[m3].mean()), "uncond": float(ju[m3].mean()), "cond": float(jc[m3].mean())}
    STATS["valve_phase2_min"] = {"progress": float(pr[m2].min()), "uncond": float(ju[m2].min()), "cond": float(jc[m2].min())}
    # first time conditioned drops below .5 in phase 3
    idx = np.where((tt > 120) & (jc < 0.5))[0]
    STATS["cond_detect_time"] = float(tt[idx[0]]) if len(idx) else None
    ip = np.where((tt > 120) & (pr < 0.5))[0]; STATS["prog_detect_time"] = float(tt[ip[0]]) if len(ip) else None
    iu = np.where((tt > 60) & (tt < 120) & (ju < 0.5))[0]; STATS["uncond_false_alarm_frac_phase2"] = float(len(iu) / max(1, np.sum((tt > 60) & (tt < 120))))
    iu3 = np.where((tt > 150) & (ju < 0.5))[0]; STATS["uncond_detect_time"] = float(tt[iu3[0]]) if len(iu3) else None
    fig.savefig(os.path.join(OUT, "fig2_prelim.png"))
    plt.close(fig)


# ---------------------------------------------------------------- Fig 3: Thrust 1 concept
def fig_t1():
    fig, axs = plt.subplots(1, 3, figsize=(6.6, 1.95), sharey=False, gridspec_kw={"wspace": 0.28})
    cases = [("(a) Calm sea", (0.125, 0.012), (0.0, 0.012)),
             ("(b) Rough sea, unconditioned", (0.11, 0.075), (0.0, 0.07)),
             ("(c) Rough sea, conditioned on $w$", (0.115, 0.016), (0.008, 0.016))]
    x = np.linspace(-0.25, 0.35, 800)
    for ax, (ttl, (ma, sa), (m0, s0)) in zip(axs, cases):
        pa = np.exp(-0.5 * ((x - ma) / sa) ** 2) / (sa * np.sqrt(2 * np.pi))
        p0 = np.exp(-0.5 * ((x - m0) / s0) ** 2) / (s0 * np.sqrt(2 * np.pi))
        ax.fill_between(x, p0, color=ORANGE, alpha=0.18, lw=0)
        ax.plot(x, p0, color=ORANGE, lw=1.4, label="no action $a_\\varnothing$")
        ax.fill_between(x, pa, color=BLUE, alpha=0.18, lw=0)
        ax.plot(x, pa, color=BLUE, lw=1.4, label="candidate action $a$")
        js = js_gauss(ma, sa, m0, s0)
        ax.set_title(ttl + f"\nJS divergence = {js:.2f} bit", loc="left", fontsize=8.4, weight="bold")
        ax.set_yticks([]); ax.set_xlabel("task progress $g$ (rad)")
        ax.set_xlim(-0.25, 0.35); ax.grid(False)
        ax.spines["left"].set_visible(False)
    h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, frameon=False, loc="upper center", ncol=2, fontsize=7.6, bbox_to_anchor=(0.5, 1.17), handlelength=1.6)
    fig.savefig(os.path.join(OUT, "fig3_t1.png"))
    plt.close(fig)


# ---------------------------------------------------------------- Fig 4: Thrust 2
CAUSES = ["Nominal", "Strategy", "Actuation", "Resistance", "Disturbance"]
SIGS = ["progress\ninfluence", "torque\ntracking\nerror", "axial\nwrench", "wrench\nmis-\nalignment", "ship-\nmotion\ncoherence"]
# qualitative signature levels: -1 low, 0 normal, +1 high
SIGMAT = np.array([
    [0, 0, 0, 0, 0],
    [-1, 0, 0, 1, 0],
    [-1, 1, -1, 0, 0],
    [-1, 0, 1, 0, 0],
    [0, 0, 0, 0, 1],
])


def fig_t2():
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.55), gridspec_kw={"width_ratios": [1.25, 1], "wspace": 0.55})
    ax = axs[0]
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap([BLUE, "#ecebe7", ORANGE])
    ax.imshow(SIGMAT, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    sym = {-1: "low", 0: "normal", 1: "high"}
    for i in range(5):
        for j in range(5):
            v = SIGMAT[i, j]
            ax.text(j, i, sym[v], ha="center", va="center", fontsize=7.2, color="white" if v != 0 else INK2,
                    weight="bold" if v != 0 else "normal")
    ax.set_xticks(range(5)); ax.set_xticklabels(SIGS, fontsize=6.6)
    ax.set_yticks(range(5)); ax.set_yticklabels(CAUSES, fontsize=7.8)
    ax.tick_params(length=0); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, 5), minor=True); ax.set_yticks(np.arange(-0.5, 5), minor=True)
    ax.grid(which="minor", color="white", lw=2); ax.tick_params(which="minor", length=0)
    ax.set_title("(a) Hypothesized cause signatures", loc="left", weight="bold")

    # (b) sequential Bayesian diagnosis with information-gain probe selection (simplified model)
    probes = ["P1 axial\nramp", "P2 off-axis\nwiggle", "P3 free-space\ntorque step", "P4 phase-timed\npush"]
    # mean signature (4 observed channels) for each cause under each probe, unit-noise scale
    mu = np.zeros((4, 5, 4))
    base = np.array([[2.0, 0.0, 1.0, 0.0],   # nominal
                     [0.6, 0.0, 1.0, 1.4],   # strategy
                     [0.4, 1.6, 0.4, 0.0],   # actuation
                     [0.4, 0.0, 2.2, 0.0],   # resistance
                     [1.4, 0.0, 1.0, 0.0]])  # disturbance
    for p in range(4):
        mu[p] = base.copy()
    mu[0][:, 2] *= 1.25                      # axial ramp sharpens wrench channel
    mu[1][1, 3] = 2.2                        # off-axis wiggle reveals misalignment
    mu[2][2, 1] = 2.6; mu[2][:, 0] = [0.0, 0.0, 0.0, 0.0, 0.0]  # free-space step isolates actuation
    mu[3][4, 0] = 2.4; mu[3][4, 3] = 0.0     # phase-timed push separates disturbance
    sd = 0.9
    true_c = 3
    post = np.ones(5) / 5
    hist, chosen = [post.copy()], []

    def loglik(o, p):
        return -0.5 * np.sum(((o[None, :] - mu[p]) / sd) ** 2, axis=1)

    for step in range(4):
        best, best_ig = None, -1
        for p in range(4):
            ig = 0.0
            for c in range(5):
                for _ in range(200):
                    o = mu[p][c] + rng.normal(0, sd, 4)
                    ll = loglik(o, p) + np.log(post)
                    pp = np.exp(ll - ll.max()); pp /= pp.sum()
                    ent = -np.sum(pp * np.log(pp + 1e-12))
                    ig += post[c] * ent / 200
            h0 = -np.sum(post * np.log(post + 1e-12))
            if h0 - ig > best_ig:
                best_ig, best = h0 - ig, p
        o = mu[best][true_c] + rng.normal(0, sd, 4)
        ll = loglik(o, best) + np.log(post)
        post = np.exp(ll - ll.max()); post /= post.sum()
        hist.append(post.copy()); chosen.append(best)
    H = np.array(hist).T
    ax = axs[1]
    ax.imshow(H, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#ffffff", BLUE]), vmin=0, vmax=1, aspect="auto")
    for i in range(5):
        for j in range(H.shape[1]):
            ax.text(j, i, f"{H[i, j]:.2f}", ha="center", va="center", fontsize=7.2, color="white" if H[i, j] > 0.55 else INK)
    ax.set_yticks(range(5)); ax.set_yticklabels(CAUSES, fontsize=7.8)
    labs = ["prior"] + [probes[c].split(" ")[0] for c in chosen]
    ax.set_xticks(range(H.shape[1])); ax.set_xticklabels(labs, fontsize=7.2)
    ax.set_xlabel("probe selected (max. expected information gain)", fontsize=7.6)
    ax.tick_params(length=0); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("(b) Posterior over causes", loc="left", weight="bold")
    ax.get_yticklabels()[true_c].set_weight("bold")
    STATS["t2_final_post_true"] = float(H[true_c, -1]); STATS["t2_probes"] = [probes[c].replace("\n", " ") for c in chosen]
    STATS["t2_post_after"] = [float(h) for h in H[true_c]]
    fig.savefig(os.path.join(OUT, "fig4_t2.png"))
    plt.close(fig)


# ---------------------------------------------------------------- Fig 5: Thrust 3
def fig_t3():
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.5), gridspec_kw={"width_ratios": [1, 1.15], "wspace": 0.32})
    ax = axs[0]
    k = 0.5
    I = np.linspace(0, 1, 200)
    umax = 0.5
    ax.fill_between(I, 0, np.clip(I - k, 0, umax), color=AQUA, alpha=0.30, lw=0)
    ax.fill_between(I, 0, np.clip(k - I, 0, umax), color=ORANGE, alpha=0.28, lw=0)
    ax.fill_between(I, np.maximum(np.abs(I - k), 0), umax, color=BLUE, alpha=0.20, lw=0)
    ax.plot(I, np.abs(I - k), color=INK2, lw=1.0)
    ax.text(0.86, 0.09, "PERSIST\n$\\underline{\\mathcal{I}}\\geq\\kappa$", ha="center", fontsize=7.6, color=INK)
    ax.text(0.15, 0.09, "LOSS OF\nINFLUENCE\n$\\overline{\\mathcal{I}}<\\kappa$", ha="center", fontsize=7.4, color=INK)
    ax.text(0.5, 0.36, "PROBE\n$\\underline{\\mathcal{I}}<\\kappa\\leq\\overline{\\mathcal{I}}$", ha="center", fontsize=7.6, color=INK)
    ax.set_xlim(0, 1); ax.set_ylim(0, umax)
    ax.set_xlabel("estimated influence $\\hat{\\mathcal{I}}$ (normalized)")
    ax.set_ylabel("half-width of confidence interval")
    ax.axvline(k, color=MUTED, lw=0.8, ls=":")
    ax.text(k + 0.01, umax * 0.97, "$\\kappa$", fontsize=8.5, va="top", color=INK2)
    ax.set_title("(a) Decision regions", loc="left", weight="bold")
    ax.grid(False)

    ax = axs[1]
    a_err = b_err = 0.05
    upA, loB = np.log((1 - b_err) / a_err), np.log(b_err / (1 - a_err))
    m1, m0, sd = 0.8, 0.0, 1.0
    decisions = {"eff": [], "loss": []}
    for lab, mu_true, col in (("eff", m1, BLUE), ("loss", m0, ORANGE)):
        for ep in range(5):
            L, path = 0.0, [0.0]
            for i in range(60):
                z = rng.normal(mu_true, sd)
                L += (m1 - m0) / sd ** 2 * (z - (m1 + m0) / 2)
                path.append(L)
                if L >= upA or L <= loB:
                    break
            ax.plot(range(len(path)), path, color=col, lw=1.1, alpha=0.9)
            ax.plot(len(path) - 1, path[-1], "o", ms=3.5, color=col)
    # Monte Carlo error rates
    errs = {"eff": 0, "loss": 0}; lens = {"eff": [], "loss": []}
    for lab, mu_true in (("eff", m1), ("loss", m0)):
        for ep in range(4000):
            L = 0.0
            for i in range(400):
                z = rng.normal(mu_true, sd)
                L += (m1 - m0) / sd ** 2 * (z - (m1 + m0) / 2)
                if L >= upA or L <= loB:
                    break
            lens[lab].append(i + 1)
            if lab == "eff" and L <= loB: errs["eff"] += 1
            if lab == "loss" and L >= upA: errs["loss"] += 1
    STATS["sprt_unnec_susp"] = errs["eff"] / 4000; STATS["sprt_unsafe_persist"] = errs["loss"] / 4000
    STATS["sprt_mean_probes"] = {k2: float(np.mean(v)) for k2, v in lens.items()}
    ax.axhline(upA, color=AQUA, lw=1.2, ls="--"); ax.axhline(loB, color=ORANGE, lw=1.2, ls="--")
    ax.text(29.5, upA + 0.3, "persist:  $\\Lambda_n \\geq \\ln\\frac{1-\\beta}{\\alpha}$", fontsize=7.4, color=INK, ha="right", bbox=dict(fc="white", ec="none", pad=1))
    ax.text(29.5, loB - 0.8, "suspend or re-plan:  $\\Lambda_n \\leq \\ln\\frac{\\beta}{1-\\alpha}$", fontsize=7.4, color=INK, ha="right", bbox=dict(fc="white", ec="none", pad=1))
    ax.plot([], [], color=BLUE, label="influence retained"); ax.plot([], [], color=ORANGE, label="influence lost")
    ax.legend(frameon=True, framealpha=1, edgecolor="none", loc="center right", fontsize=7.0)
    ax.set_xlim(-0.5, 30)
    ax.set_ylim(loB - 1.7, upA + 1.4)
    ax.set_xlabel("number of safe probes $n$"); ax.set_ylabel("log-likelihood ratio $\\Lambda_n$")
    ax.set_title("(b) Sequential test with bounded errors", loc="left", weight="bold")
    fig.savefig(os.path.join(OUT, "fig5_t3.png"))
    plt.close(fig)


# ---------------------------------------------------------------- Fig 6: testbed
def jonswap(w, Hs, Tp, gamma=3.3):
    wp = 2 * np.pi / Tp
    sig = np.where(w <= wp, 0.07, 0.09)
    r = np.exp(-((w - wp) ** 2) / (2 * sig ** 2 * wp ** 2))
    S = 9.81 ** 2 * w ** -5.0 * np.exp(-1.25 * (wp / w) ** 4) * gamma ** r
    m0 = np.trapezoid(S, w)
    return S * (Hs / 4) ** 2 / m0


def fig_testbed():
    fig = plt.figure(figsize=(6.6, 3.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.15], hspace=0.75, wspace=0.18)
    ax = fig.add_subplot(gs[0, 0])
    w = np.linspace(0.2, 2.5, 600)
    seas = [("sea state 3 ($H_s$ 0.88 m, $T_p$ 7.5 s)", 0.88, 7.5, AQUA),
            ("sea state 4 ($H_s$ 1.88 m, $T_p$ 8.8 s)", 1.88, 8.8, BLUE),
            ("sea state 5 ($H_s$ 3.25 m, $T_p$ 9.7 s)", 3.25, 9.7, ORANGE)]
    for lab, Hs, Tp, col in seas:
        ax.plot(w, jonswap(w, Hs, Tp), color=col, lw=1.4, label=lab)
    ax.set_xlabel("wave frequency $\\omega$ (rad/s)", fontsize=7.8); ax.set_ylabel("$S(\\omega)$ (m$^2$ s)", fontsize=7.8)
    ax.set_title("(a) JONSWAP wave spectra", loc="left", weight="bold", fontsize=8.6)
    ax.legend(frameon=False, fontsize=6.4, loc="upper right", handlelength=1.2)
    ax.set_xlim(0.2, 2.0)

    ax = fig.add_subplot(gs[1, 0])
    t = np.linspace(0, 120, 3000)
    w_i = np.linspace(0.25, 2.0, 160); dw = w_i[1] - w_i[0]
    S = jonswap(w_i, 3.25, 9.7)
    amp = np.sqrt(2 * S * dw)
    ph = rng.uniform(0, 2 * np.pi, len(w_i))
    def resp(Tn, z, gain):
        wn = 2 * np.pi / Tn; r = w_i / wn
        H = 1 / np.sqrt((1 - r ** 2) ** 2 + (2 * z * r) ** 2)
        slope = amp * w_i ** 2 / 9.81
        return gain * np.sum((H * slope)[:, None] * np.cos(np.outer(w_i, t) + ph[:, None]), axis=0)
    roll = resp(9.0, 0.12, 1.0); pitch = resp(5.0, 0.35, 1.0)
    roll *= 5.0 / roll.std(); pitch *= 2.0 / pitch.std()
    ax.plot(t, roll, color=BLUE, lw=1.0, label="roll")
    ax.plot(t, pitch, color=ORANGE, lw=1.0, label="pitch")
    ax.set_xlabel("time (s)", fontsize=7.8); ax.set_ylabel("angle (deg)", fontsize=7.8)
    ax.set_title("(b) Synthetic platform motion, sea state 5", loc="left", weight="bold", fontsize=8.6)
    ax.legend(frameon=False, fontsize=6.8, ncol=2, loc="upper right")
    ax.set_xlim(0, 120)

    ax = fig.add_subplot(gs[:, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("(c) Physical testbed (schematic)", loc="left", weight="bold", fontsize=8.6)
    ax.add_patch(Rectangle((0.08, 0.04), 0.84, 0.07, fc="#d9d8d3", ec=MUTED))
    ax.add_patch(Polygon([[0.2, 0.11], [0.8, 0.11], [0.72, 0.19], [0.28, 0.19]], fc="#ecebe7", ec=MUTED))
    ax.add_patch(Polygon([[0.06, 0.23], [0.96, 0.29], [0.96, 0.33], [0.06, 0.27]], fc=LIGHTBLUE, ec=NAVY, lw=1.0))
    ax.text(0.5, 0.075, "commercial two-axis roll/pitch platform", ha="center", va="center", fontsize=6.4, color=INK2)
    ax.annotate("", xy=(0.03, 0.40), xytext=(0.03, 0.17), arrowprops=dict(arrowstyle="<->", color=NAVY, lw=1.0))
    ax.text(0.03, 0.415, "roll/pitch", fontsize=6.2, color=NAVY, ha="left", va="bottom")
    ax.add_patch(Rectangle((0.16, 0.29), 0.09, 0.05, fc=NAVY, ec=NAVY))
    jx = [0.205, 0.22, 0.40, 0.55]; jy = [0.34, 0.60, 0.74, 0.60]
    ax.plot(jx, jy, color=NAVY, lw=5, solid_capstyle="round")
    for x_, y_ in zip(jx, jy):
        ax.add_patch(Circle((x_, y_), 0.018, fc="white", ec=NAVY, lw=1.2, zorder=5))
    ax.add_patch(Rectangle((0.535, 0.51), 0.03, 0.07, fc=ORANGE, ec=ORANGE, zorder=6))
    ax.text(0.55, 0.47, "wrist F/T", fontsize=6.4, color=INK2, ha="center", va="top")
    ax.text(0.02, 0.83, "UR10e manipulator\n(wrist F/T sensor,\njoint currents)", fontsize=6.4, color=INK2, va="top")
    ax.add_patch(Rectangle((0.68, 0.34), 0.25, 0.34, fc="white", ec=INK2, lw=1.0))
    ax.add_patch(Circle((0.745, 0.51), 0.045, fc="white", ec=BLUE, lw=1.6))
    ax.plot([0.70, 0.79], [0.51, 0.51], color=BLUE, lw=1.6)
    for (x_, y_) in ((0.845, 0.56), (0.895, 0.56), (0.845, 0.46), (0.895, 0.46)):
        ax.add_patch(Polygon([[x_ - 0.016, y_], [x_ - 0.008, y_ + 0.014], [x_ + 0.008, y_ + 0.014], [x_ + 0.016, y_], [x_ + 0.008, y_ - 0.014], [x_ - 0.008, y_ - 0.014]], fc="#bdbcb6", ec=INK2, lw=0.6))
    ax.text(0.805, 0.71, "task board: valve with\nprogrammable brake;\nbolts with adjustable preload", fontsize=6.2, ha="center", va="bottom", color=INK2)
    ax.add_patch(Rectangle((0.40, 0.33), 0.07, 0.035, fc=AQUA, ec=AQUA))
    ax.text(0.435, 0.38, "IMU", fontsize=6.4, ha="center", color=INK2)
    ax.text(0.5, 1.0, "perturbations injected with known ground truth:\nfriction, preload/seizure, torque caps, sea state",
            ha="center", va="top", fontsize=6.5, color=INK, style="italic")
    fig.savefig(os.path.join(OUT, "fig6_testbed.png"))
    plt.close(fig)


# ---------------------------------------------------------------- Fig 7: Gantt
def fig_gantt():
    tasks = [
        ("T1.1 Theory of task-relevant influence", 1, 4, BLUE),
        ("T1.2 Estimation: contact-rich, partially observed", 3, 8, BLUE),
        ("T1.3 Safe, phase-matched diagnostic probes", 5, 8, BLUE),
        ("T2.1 Cause models and identifiability", 3, 7, ORANGE),
        ("T2.2 Decision-aware diagnostic probing", 6, 10, ORANGE),
        ("T3.1 Decision policy and error bounds", 5, 9, AQUA),
        ("T3.2 Robustness to model misspecification", 9, 11, AQUA),
        ("E1 Digital twin with sea-state motion", 1, 4, MUTED),
        ("E2 Hardware testbed assembly", 2, 5, MUTED),
        ("E3 Hardware evaluation campaigns", 7, 12, MUTED),
        ("D  Open benchmark, code, and data release", 10, 12, MUTED),
    ]
    miles = [(2, "M1"), (4, "M2"), (6, "M3"), (8, "M4"), (10, "M5"), (12, "M6")]
    fig, ax = plt.subplots(figsize=(6.6, 2.45))
    for i, (lab, s, e, col) in enumerate(tasks):
        ax.barh(i, e - s + 1, left=s - 0.5, height=0.62, color=col, alpha=0.85 if col != MUTED else 0.55, edgecolor="white", lw=1.5)
    ax.set_yticks(range(len(tasks))); ax.set_yticklabels([t[0] for t in tasks], fontsize=7.4)
    ax.invert_yaxis()
    ax.set_xlim(0.5, 12.5)
    ax.set_xticks(range(1, 13)); ax.set_xticklabels([f"Q{((q - 1) % 4) + 1}" for q in range(1, 13)], fontsize=7.2)
    for yr, x in ((1, 2.5), (2, 6.5), (3, 10.5)):
        ax.text(x, -1.65, f"Year {yr}  ({2026 + yr}–{2027 + yr})", ha="center", fontsize=7.6, weight="bold", color=INK)
    for x in (4.5, 8.5):
        ax.axvline(x, color=MUTED, lw=0.8)
    for q, m in miles:
        ax.plot(q + 0.35, len(tasks) - 0.1, marker="D", ms=5.5, color=NAVY, clip_on=False)
        ax.text(q + 0.35, len(tasks) + 0.55, m, ha="center", fontsize=7.2, color=NAVY, weight="bold")
    for q in (4, 8, 12):
        ax.plot(q + 0.35, -1.05, marker="v", ms=5.5, color=ORANGE, clip_on=False)
    ax.text(12.6, -1.05, "ONR review", fontsize=6.8, color=INK2, va="center")
    ax.set_ylim(len(tasks) + 0.9, -2.0)
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    fig.savefig(os.path.join(OUT, "fig7_gantt.png"))
    plt.close(fig)


# ---------------------------------------------------------------- Fig 8: naval pathway
def fig_pathway():
    fig, ax = plt.subplots(figsize=(6.6, 1.35))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    steps = [("Basic research\n(this YIP)", "influence, attribution,\npersist/suspend theory", LIGHTBLUE, NAVY),
             ("Validated\nprinciples", "open benchmark, sensing\nrequirements, error bounds", "#e6eef9", NAVY),
             ("Applied shipboard\nrobotics", "repair and maintenance\nprograms (BAA, STTR)", LIGHTAQ, AQUA),
             ("Naval\ncapability", "longer uncrewed missions;\nsafer hazardous-space work", LIGHTOR, ORANGE)]
    x = 0.0; w = 0.25
    for i, (h, sub, fc, ec) in enumerate(steps):
        pts = [[x, 0.04], [x + w - 0.035, 0.04], [x + w, 0.5], [x + w - 0.035, 0.96], [x, 0.96]]
        if i > 0:
            pts.append([x + 0.035, 0.5])
        ax.add_patch(Polygon(pts, closed=True, fc=fc, ec=ec, lw=1.0))
        cx = x + w / 2 + (0.006 if i > 0 else -0.012)
        ax.text(cx, 0.70, h, ha="center", va="center", fontsize=7.3, weight="bold", color=INK, linespacing=1.1)
        ax.text(cx, 0.27, sub, ha="center", va="center", fontsize=6.0, color=INK, linespacing=1.15)
        x += w - 0.002
    fig.savefig(os.path.join(OUT, "fig8_pathway.png"))
    plt.close(fig)


if __name__ == "__main__":
    fig_concept(); fig_prelim(); fig_t1(); fig_t2(); fig_t3(); fig_testbed(); fig_gantt(); fig_pathway()
    json.dump(STATS, open(os.path.join(OUT, "stats.json"), "w"), indent=2)
    print(json.dumps(STATS, indent=2))
