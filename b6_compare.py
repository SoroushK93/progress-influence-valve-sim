"""B6 analogue in the toy valve model: a disturbance-conditioned progress predictor.
Recreates exactly the episodes of figs3.fig_prelim (seed 11, same call order) and adds a
predictor fit on an independent set of seizure-free episodes (seed 12)."""
import numpy as np, json
import figs3 as G

# predictor training set, independent seed
pred = G.fit_b6_predictor(seed=12)

# recreate fig_prelim episodes in the same order
rng = np.random.default_rng(11)
G.simulate(rng, lambda t: np.where(t < 60, 0.3, 3.0), lambda t: np.where(t < 120, 2.0, np.clip(2.0 + (t - 120) / 30 * 6.0, 2.0, 8.0)), 180)
cal_min = {"b6": [], "cond": []}
for _ in range(150):
    a = rng.uniform(2.0, 3.5)
    e = G.simulate(rng, lambda t, a=a: np.full_like(t, a), lambda t: np.full_like(t, 2.0), 140)
    cal_min["b6"].append(np.nanmin(G.b6_monitor(e, pred)["b6"])); cal_min["cond"].append(np.nanmin(G.monitors(e)["cond"]))
tests = []
for _ in range(150):
    a = rng.uniform(2.0, 3.5); t_on = rng.uniform(50, 80)
    e = G.simulate(rng, lambda t, a=a: np.full_like(t, a), lambda t, t_on=t_on: np.where(t < t_on, 2.0, np.clip(2.0 + (t - t_on) / 30 * 6.0, 2.0, 8.0)), t_on + 70)
    mb = G.b6_monitor(e, pred); mc = G.monitors(e)
    tests.append((t_on, {"b6": mb, "cond": {"t": mc["t"], "cond": mc["cond"]}}))
res = {}
for k in ("b6", "cond"):
    mins = np.array(cal_min[k]); res[k] = []
    for r in (0.02, 0.05, 0.10, 0.20, 0.30):
        thr = np.quantile(mins, r); delays, fa, miss = [], 0, 0
        for t_on, mm in tests:
            m = mm[k]; below = np.where(m[k] < thr)[0]
            if len(below) == 0: miss += 1; continue
            ta = m["t"][below[0]]
            if ta < t_on: fa += 1; continue
            delays.append(ta - t_on)
        res[k].append(dict(fa_target=r, median=float(np.median(delays)), q25=float(np.percentile(delays, 25)), q75=float(np.percentile(delays, 75)),
                           test_fa=fa / 150, miss=miss / 150))
json.dump(res, open("figs/stats_b6.json", "w"), indent=1)
for k in res:
    print(k, [(r["fa_target"], round(r["median"], 1), round(r["test_fa"], 3), r["miss"]) for r in res[k]])
