"""B6 analog under model mismatch: thresholds and the B6 progress model come from one valve
(development valve, nominal friction); testing is on a different valve that is stiffer or looser.
Same monitors and calibration as Fig. 2b. Seeds: 12 (B6 fit), 11 (calibration episodes, as in Fig. 2b), 13 (new-valve tests)."""
import numpy as np, json
import figs3 as G

def simulate_p(rng, a_s_fn, tau_s_fn, T, tau_k, dt=0.2):
    n = int(T / dt); t = np.arange(n) * dt
    a_s = a_s_fn(t); tau_s = tau_s_fn(t)
    act = rng.integers(0, 2, n)
    d = a_s * np.sin(rng.uniform(0, 2 * np.pi, n)) + rng.normal(0, 0.1, n)
    dhat = d + rng.normal(0, G.M_NOISE, n)
    tau = act * G.U_P + d
    move = np.abs(tau) > tau_s
    dth = np.where(move, np.sign(tau) * (np.abs(tau) - tau_k) * G.CGAIN, 0.0) + rng.normal(0, G.S_NOISE, n)
    wrench = act * G.U_P + 0.3 * d + rng.normal(0, G.W_NOISE, n)
    return dict(t=t, act=act, d=d, dhat=dhat, move=move, dth=dth, wrench=wrench)

pred = G.fit_b6_predictor(seed=12)                      # B6 model learned on the development valve
# thresholds from development-valve seizure-free episodes (identical to Fig. 2b calibration)
rng = np.random.default_rng(11)
G.simulate(rng, lambda t: np.where(t < 60, 0.3, 3.0), lambda t: np.where(t < 120, 2.0, np.clip(2.0 + (t - 120) / 30 * 6.0, 2.0, 8.0)), 180)
cal = {"b6": [], "cond": []}
for _ in range(150):
    a = rng.uniform(2.0, 3.5)
    e = G.simulate(rng, lambda t, a=a: np.full_like(t, a), lambda t: np.full_like(t, 2.0), 140)
    cal["b6"].append(np.nanmin(G.b6_monitor(e, pred)["b6"])); cal["cond"].append(np.nanmin(G.monitors(e)["cond"]))
thr = {k: float(np.quantile(cal[k], 0.05)) for k in cal}       # 5% false-alarm target on the development valve

def evaluate(name, tau_k, brk, seed=13, n=150):
    rng = np.random.default_rng(seed); out = {}
    # (i) false alarms on the new valve with no fault
    fa = {"b6": 0, "cond": 0}
    for _ in range(n):
        a = rng.uniform(2.0, 3.5)
        e = simulate_p(rng, lambda t, a=a: np.full_like(t, a), lambda t: np.full_like(t, brk), 140, tau_k)
        if np.nanmin(G.b6_monitor(e, pred)["b6"]) < thr["b6"]: fa["b6"] += 1
        if np.nanmin(G.monitors(e)["cond"]) < thr["cond"]: fa["cond"] += 1
    # (ii) detection delay on the new valve with a seizure (alarms before onset count as false alarms)
    dl = {"b6": [], "cond": []}; early = {"b6": 0, "cond": 0}
    for _ in range(n):
        a = rng.uniform(2.0, 3.5); t_on = rng.uniform(50, 80)
        e = simulate_p(rng, lambda t, a=a: np.full_like(t, a),
                       lambda t, t_on=t_on: np.where(t < t_on, brk, np.clip(brk + (t - t_on) / 30 * (8.0 - brk), brk, 8.0)), t_on + 70, tau_k)
        ms = {"b6": G.b6_monitor(e, pred), "cond": G.monitors(e)}
        for k in ("b6", "cond"):
            below = np.where(ms[k][k] < thr[k])[0]
            if len(below) == 0: continue
            ta = ms[k]["t"][below[0]]
            if ta < t_on: early[k] += 1
            else: dl[k].append(ta - t_on)
    for k in ("b6", "cond"):
        out[k] = dict(false_alarm_no_fault=fa[k] / n, alarm_before_onset=early[k] / n,
                      median_delay=float(np.median(dl[k])) if dl[k] else None, n_detected_after_onset=len(dl[k]))
    return out

res = {"thresholds_dev_valve": thr,
       "same_valve": evaluate("same", 1.5, 2.0),
       "stiffer_valve": evaluate("stiffer", 2.0, 2.5),
       "looser_valve": evaluate("looser", 1.0, 1.5)}
json.dump(res, open("figs/stats_b6_mismatch.json", "w"), indent=1)
print(json.dumps(res, indent=1))
