#!/usr/bin/env python3
"""Final tuning probes: long-horizon rho safety + U boundary at low rho."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_common import M, apply_config

SEEDS = list(range(7, 17))
H55 = M.N_SLOTS_PER_ORBIT * 55
H20 = M.N_SLOTS_PER_ORBIT * 20

def _long(args):
    rho, sd = args
    apply_config(dict(U_TGT=0.15, RESERVE_FRAC=rho))
    r = M.run_sim(M.POLICIES["Lyapunov (ours)"], V=0.3, horizon=H55, seed=sd)
    return r["blackout_slots"] / (M.N_SAT * H55)

def _ubound(args):
    rho, u, sd = args
    apply_config(dict(U_TGT=u, RESERVE_FRAC=rho))
    r = M.run_sim(M.POLICIES["Lyapunov (ours)"], V=0.3, horizon=H20, seed=sd)
    return r["min_fill"], float(np.max(r["QU_end"]) / H20), r["total_quality"]

if __name__ == "__main__":
    import concurrent.futures as cf
    print("== long-horizon (55 orbits) down% at V=0.3 ==", flush=True)
    for rho in [0.5, 0.4, 0.3, 0.2]:
        with cf.ProcessPoolExecutor(max_workers=10) as ex:
            res = list(ex.map(_long, [(rho, sd) for sd in SEEDS]))
        print(f"  rho={rho}: down={np.mean(res)*100:.3f}%", flush=True)
    print("== U sweep at rho=0.4, V=0.3, 20 orbits ==", flush=True)
    for u in [0.15, 0.18, 0.20, 0.22, 0.25, 0.28]:
        with cf.ProcessPoolExecutor(max_workers=10) as ex:
            res = list(ex.map(_ubound, [(0.4, u, sd) for sd in SEEDS]))
        mf = np.mean([r[0] for r in res]); qu = np.mean([r[1] for r in res])
        tq = np.mean([r[2] for r in res])
        print(f"  U={u}: maxmin={mf:.3f} QUend/H={qu:.4f} totQ={tq:.0f}", flush=True)
    print("== U sweep at rho=0.3, V=0.3, 20 orbits ==", flush=True)
    for u in [0.15, 0.20, 0.25, 0.30]:
        with cf.ProcessPoolExecutor(max_workers=10) as ex:
            res = list(ex.map(_ubound, [(0.3, u, sd) for sd in SEEDS]))
        mf = np.mean([r[0] for r in res]); qu = np.mean([r[1] for r in res])
        tq = np.mean([r[2] for r in res])
        print(f"  U={u}: maxmin={mf:.3f} QUend/H={qu:.4f} totQ={tq:.0f}", flush=True)
