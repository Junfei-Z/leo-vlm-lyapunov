#!/usr/bin/env python3
"""rho x V joint probe at U=0.15, 3 orbits, 10 seeds."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_common import M, apply_config

SEEDS = list(range(7, 17))

def _one(args):
    rho, v, sd = args
    apply_config(dict(U_TGT=0.15, RESERVE_FRAC=rho))
    r = M.run_sim(M.POLICIES["Lyapunov (ours)"], V=v, seed=sd)
    return (r["blackout_slots"] / (M.N_SAT * M.HORIZON), r["min_fill"],
            r["total_quality"], r["split_tasks"])

if __name__ == "__main__":
    import concurrent.futures as cf
    combos = [(rho, v) for rho in [0.5, 0.4, 0.3] for v in [0.3, 1.0, 3.0]]
    print(f"{'rho':>5} {'V':>5} {'down%':>7} {'maxmin':>8} {'totQ':>7} {'splits':>7}", flush=True)
    for rho, v in combos:
        with cf.ProcessPoolExecutor(max_workers=10) as ex:
            res = list(ex.map(_one, [(rho, v, sd) for sd in SEEDS]))
        down = np.mean([r[0] for r in res]) * 100
        mf = np.mean([r[1] for r in res]); tq = np.mean([r[2] for r in res])
        sp = np.mean([r[3] for r in res])
        print(f"{rho:5.2f} {v:5.1f} {down:7.2f} {mf:8.3f} {tq:7.0f} {sp:7.0f}", flush=True)
