#!/usr/bin/env python3
"""U ceiling at rho=0.3 over 20 orbits (V=1), to fix the final U^tgt."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_common import M, apply_config

SEEDS = list(range(7, 17))
H20 = M.N_SLOTS_PER_ORBIT * 20

def _one(args):
    u, v, sd = args
    apply_config(dict(U_TGT=u, RESERVE_FRAC=0.3))
    r = M.run_sim(M.POLICIES["Lyapunov (ours)"], V=v, horizon=H20, seed=sd)
    return r["min_fill"], float(np.max(r["QU_end"])) / H20, r["total_quality"]

if __name__ == "__main__":
    import concurrent.futures as cf
    for v in [0.3, 1.0]:
        print(f"== V={v}, rho=0.3, 20 orbits ==", flush=True)
        for u in [0.30, 0.33, 0.35, 0.38]:
            with cf.ProcessPoolExecutor(max_workers=10) as ex:
                res = list(ex.map(_one, [(u, v, sd) for sd in SEEDS]))
            mf = np.mean([r[0] for r in res]); qu = np.mean([r[1] for r in res])
            print(f"  U={u}: maxmin={mf:.3f} QUend/H={qu:.4f} totQ={np.mean([r[2] for r in res]):.0f}", flush=True)
