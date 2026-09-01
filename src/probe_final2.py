#!/usr/bin/env python3
"""3-orbit table stats + U ceiling at low-rho candidate configs."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_common import M, apply_config

SEEDS = list(range(7, 17))

def _one(args):
    rho, v, u, sd = args
    apply_config(dict(U_TGT=u, RESERVE_FRAC=rho))
    r = M.run_sim(M.POLICIES["Lyapunov (ours)"], V=v, seed=sd)
    return (r["blackout_slots"] / (M.N_SAT * M.HORIZON), r["min_fill"],
            r["total_quality"], r["split_tasks"], r["service_rate"])

if __name__ == "__main__":
    import concurrent.futures as cf
    cands = [(0.3, 0.3, 0.30), (0.3, 0.3, 0.33), (0.3, 0.3, 0.35),
             (0.3, 1.0, 0.30), (0.2, 0.3, 0.30), (0.4, 0.3, 0.28)]
    print(f"{'rho':>5} {'V':>5} {'U':>5} {'down%':>7} {'maxmin':>8} {'totQ':>7} {'splits':>7} {'serv':>6}", flush=True)
    for rho, v, u in cands:
        with cf.ProcessPoolExecutor(max_workers=10) as ex:
            res = list(ex.map(_one, [(rho, v, u, sd) for sd in SEEDS]))
        down = np.mean([r[0] for r in res]) * 100
        mf = np.mean([r[1] for r in res]); tq = np.mean([r[2] for r in res])
        sp = np.mean([r[3] for r in res]); sv = np.mean([r[4] for r in res])
        print(f"{rho:5.2f} {v:5.1f} {u:5.2f} {down:7.2f} {mf:8.3f} {tq:7.0f} "
              f"{sp:7.0f} {sv:6.3f}", flush=True)
