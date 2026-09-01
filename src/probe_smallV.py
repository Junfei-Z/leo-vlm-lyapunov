#!/usr/bin/env python3
"""Probe: does V < 1 raise max-min / total quality at U=0.15 (mainline, 3 orbits)?"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_common import M, apply_config  # noqa: E402

SEEDS = list(range(7, 17))


def stat(V):
    a = {k: [] for k in ["blackout_slots", "min_fill", "total_quality", "split_tasks", "service_rate"]}
    for sd in SEEDS:
        apply_config(dict(U_TGT=0.15))
        r = M.run_sim(M.POLICIES["Lyapunov (ours)"], V=V, seed=sd)
        for k in a:
            a[k].append(r[k])
    TOT = M.N_SAT * M.HORIZON
    return dict(down=np.mean(a["blackout_slots"]) / TOT * 100,
                mf=np.mean(a["min_fill"]), mf_sd=np.std(a["min_fill"]),
                tq=np.mean(a["total_quality"]), sp=np.mean(a["split_tasks"]),
                sv=np.mean(a["service_rate"]))


if __name__ == "__main__":
    print(f"{'V':>6} {'down%':>7} {'maxmin':>8} {'totQ':>7} {'splits':>7} {'serv':>6}")
    for V in [0.0, 0.1, 0.3, 0.5, 1.0, 3.0, 10.0]:
        st = stat(V)
        print(f"{V:6.1f} {st['down']:7.2f} {st['mf']:8.3f} {st['tq']:7.0f} "
              f"{st['sp']:7.0f} {st['sv']:6.3f}", flush=True)
