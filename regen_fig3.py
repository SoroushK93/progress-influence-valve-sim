"""Regenerates the attribution figure (Fig. 3) with the original random-number sequence
(seed 7, consumed first by the concept and preliminary figures of figs.py), so that the
posterior 0.20 -> 0.60 -> 0.98 is reproduced. Earlier figures are written to a scratch folder."""
import numpy as np, os
import figs as F
F.rng = np.random.default_rng(7)
real = F.OUT; F.OUT = os.path.join(os.path.dirname(__file__), "scratch_figs"); os.makedirs(F.OUT, exist_ok=True)
F.fig_concept(); F.fig_prelim(); F.fig_t1()
F.OUT = real; os.makedirs(real, exist_ok=True)
F.fig_t2()
print("posterior of true cause:", [round(x, 2) for x in F.STATS["t2_post_after"]], F.STATS["t2_probes"])
