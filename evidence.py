import numpy as np, json, sys, os, importlib
sys.path.insert(0, os.getcwd())
import figs3 as G
from scipy import stats
out = {}
# --- (a) reproduce preliminary stats in a fresh process, outputs to a temp dir
G.OUT = "repro"; os.makedirs(G.OUT, exist_ok=True)
G.STATS.clear(); G.fig_prelim()
S = G.STATS
out["repro"] = dict(ep_phase2=S["ep_phase2"], cond_5pct=[r for r in S["tradeoff"]["cond"] if r["fa_target"] == 0.05][0],
                    prog_5pct=[r for r in S["tradeoff"]["prog"] if r["fa_target"] == 0.05][0]["median_delay"],
                    unc_5pct=[r for r in S["tradeoff"]["unc"] if r["fa_target"] == 0.05][0]["median_delay"],
                    loss_after_onset_median=S["loss_after_onset_median"])
# --- (b) betting latency
def run(mu, a, b, nmax, reps, seed):
    rng = np.random.default_rng(seed); d = []; n = []
    for _ in range(reps):
        r = G.betting_test(rng, mu, a=a, b=b, nmax=nmax); d.append(r[0]); n.append(r[1])
    d = np.array(d); n = np.array(n)
    return dict(persist=float(np.mean(d == "persist")), stop=float(np.mean(d == "stop")), undecided=float(np.mean(d == "undecided")),
                median=float(np.median(n)), p90=float(np.percentile(n, 90)), p95=float(np.percentile(n, 95)))
lat = {}
for lab, a in (("single_0.05", 0.05), ("split8_0.00625", 0.05 / 8)):
    for cond, mu in (("useful", 0.6), ("none", 0.0)):
        lat[f"{lab}_{cond}_nmax200"] = run(mu, a, a, 200, 4000, 5)
        lat[f"{lab}_{cond}_nmax40"] = run(mu, a, a, 40, 4000, 6)
out["latency"] = lat
# clipped-noise check: mean and sd of clipped N(0,0.6) at +-1.5
z = np.clip(np.random.default_rng(1).normal(0, 0.6, 2_000_000), -1.5, 1.5)
out["clip_noise"] = dict(mean=float(z.mean()), sd=float(z.std()))
# --- (c) power / precision with eligible denominators (normal approximations, one-sided alpha 0.05)
za = stats.norm.ppf(0.95)
def power_two_prop(p1, p2, n):  # H1: p2 < p1 (reduction)
    se = np.sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / n); return float(stats.norm.cdf((p1 - p2) / se - za))
pw = {}
for n in (80, 120, 160):
    pw[f"persist_n{n}"] = {f"{p0:.2f}->{p0/2:.2f}": power_two_prop(p0, p0 / 2, n) for p0 in (0.2, 0.3, 0.4, 0.5)}
def ni_power(p, n, margin):  # true difference 0, test diff > -margin
    se = np.sqrt(2 * p * (1 - p) / n); return float(stats.norm.cdf(margin / se - za))
pw["completion_n400"] = {f"p{p}_m{m}": ni_power(p, 400, m) for p in (0.7, 0.8, 0.9) for m in (0.03, 0.05, 0.08, 0.10)}
pw["completion_n200"] = {f"p{p}_m{m}": ni_power(p, 200, m) for p in (0.7, 0.8) for m in (0.03, 0.08, 0.10)}
# probability that one-sided 95% Clopper-Pearson upper bound < beta for unnecessary suspension
def p_ub_below(n, p, beta=0.05):
    xs = np.arange(n + 1); ub = stats.beta.ppf(0.95, xs + 1, n - xs); ub[xs == n] = 1.0
    return float(np.sum(stats.binom.pmf(xs, n, p) * (ub < beta)))
pw["susp_ub"] = {f"n{n}_p{p}": p_ub_below(n, p) for n in (200, 240, 280) for p in (0.005, 0.01, 0.02)}
out["power"] = pw
# values quoted in Sections 1.6 and 1.7 for the full policy and the planned denominators
rng = np.random.default_rng(7); r = [G.betting_test(rng, 0.6, a=0.05 / 8, b=0.05 / 8, nmax=60) for _ in range(4000)]
out["split8_useful_budget60_unresolved"] = float(np.mean(np.array([x[0] for x in r]) == "undecided"))
pw2 = {}
pw2["completion_n320_m0.08"] = {f"p{p}": ni_power(p, 320, 0.08) for p in (0.7, 0.8, 0.9)}
pw2["persist_n160"] = {f"{p0:.2f}->{p0/2:.2f}": power_two_prop(p0, p0 / 2, 160) for p0 in (0.2, 0.3)}
pw2["susp_ub_n320"] = {f"p{p}": p_ub_below(320, p) for p in (0.01, 0.02)}
pw2["G3_joint_product"] = {"base20": pw2["persist_n160"]["0.20->0.10"] * pw2["completion_n320_m0.08"]["p0.8"] * pw2["susp_ub_n320"]["p0.01"],
                           "base30": pw2["persist_n160"]["0.30->0.15"] * pw2["completion_n320_m0.08"]["p0.8"] * pw2["susp_ub_n320"]["p0.01"]}
out["power_planned"] = pw2
json.dump(out, open("figs/evidence.json", "w"), indent=1)
print(json.dumps(out, indent=1))
