#!/usr/bin/env python3
"""Long-horizon battery survival at lower reserve fractions (55 orbits, 10 seeds)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_common import M, apply_config

SEEDS = list(range(7, 17))
H = M.N_SLOTS_PER_ORBIT * 55

def _one(args):
    rho, sd = args
    apply_config(dict(U_TGT=0.15, RESERVE_FRAC=rho))
    r = M.run_sim(M.POLICIES["Lyapunov (ours)"], V=1.0, horizon=H, seed=sd)
    return r["blackout_slots"] / (M.N_SAT * H), float(np.min(r["log"]["battery"]))

if __name__ == "__main__":
    import concurrent.futures as cf
    print(f"{'rho':>5} {'down%(55orb)':>13} {'minB(J)':>10}", flush=True)
    for rho in [0.85, 0.8, 0.7, 0.6, 0.5, 0.4]:
        with cf.ProcessPoolExecutor(max_workers=10) as ex:
            res = list(ex.map(_one, [(rho, sd) for sd in SEEDS]))
        print(f"{rho:5.2f} {np.mean([r[0] for r in res])*100:13.3f} "
              f"{np.min([r[1] for r in res]):10.0f}", flush=True)
