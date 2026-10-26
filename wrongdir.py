import numpy as np, json, os
import figs3 as G
rng = np.random.default_rng(21)
res = {}
for name, sign in (("correct", 1.0), ("wrong", -1.0)):
    n = 600; t = np.arange(n) * 0.2
    act = rng.integers(0, 2, n)
    d = 0.3 * np.sin(rng.uniform(0, 2*np.pi, n)) + rng.normal(0, 0.1, n)
    dhat = d + rng.normal(0, G.M_NOISE, n)
    tau = act * sign * G.U_P + d
    move = np.abs(tau) > 2.0
    dth = np.where(move, np.sign(tau) * (np.abs(tau) - G.TAU_K) * G.CGAIN, 0.0) + rng.normal(0, G.S_NOISE, n)
    js = G.cond_js(dth, act, dhat, G.S_NOISE)
    # conditioned advantage: average within-bin mean difference
    edges = np.quantile(dhat, np.linspace(0, 1, 7)); diffs, wts = [], []
    for b in range(6):
        mb = (dhat >= edges[b]) & (dhat <= edges[b+1])
        ya, y0 = dth[mb & (act == 1)], dth[mb & (act == 0)]
        diffs.append(ya.mean() - y0.mean()); wts.append(mb.sum())
    res[name] = {"cond_js_bits": js, "cond_advantage_rad": float(np.average(diffs, weights=wts))}
# fraction of action trials that moved the valve in rough sea, free valve
ep = G.simulate(np.random.default_rng(3), lambda t: np.full_like(t, 3.0), lambda t: np.full_like(t, 2.0), 400)
a = ep["act"] == 1
res["rough_free_action_move_frac"] = float(np.mean(ep["move"][a]))
print(json.dumps(res, indent=1))
json.dump(res, open("figs/stats_wrongdir.json", "w"), indent=1)
