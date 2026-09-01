#!/usr/bin/env python3
"""ISL-availability sweep at RESERVE_FRAC=0.3, U_TGT=0.30 (mirrors src/14 isl_sweep)."""
import os, sys, csv
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_common import M, apply_config, ISL, T  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
SEEDS = list(range(7, 17))
BASE_ISL = ISL.copy()


def apply_u15():
    apply_config(dict(RESERVE_FRAC=0.3, U_TGT=0.30))


def stat(p_isl):
    a = {k: [] for k in ["blackout_slots", "min_fill", "split_tasks", "total_quality"]}
    for sd in SEEDS:
        apply_u15()

        def isl(t, s1, s2, _p=p_isl):
            if s1 == s2 or not BASE_ISL[min(t, T - 1), s1, s2]:
                return False
            a_, b = (s1, s2) if s1 < s2 else (s2, s1)
            w = t // 300
            h = ((a_ * 73856093) ^ (b * 19349663) ^ (w * 83492791)) & 0xFFFFFFFF
            return (h / 0xFFFFFFFF) < _p
        M.isl_connected = isl
        r = M.run_sim(M.POLICIES["Lyapunov (ours)"], V=10, seed=sd)
        for k in a:
            a[k].append(r[k])
    TOT = M.N_SAT * M.HORIZON
    return (np.mean(a["split_tasks"]), np.std(a["split_tasks"]),
            np.mean(a["min_fill"]), np.std(a["min_fill"]),
            np.mean(a["total_quality"]), np.mean(a["blackout_slots"]) / TOT * 100)


if __name__ == "__main__":
    rows = []
    for p in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        st = stat(p)
        rows.append([p, *[round(x, 4) for x in st]])
        print(f"  p_isl={p}: split {st[0]:.0f} mf {st[2]:.3f} totQ {st[4]:.0f} down {st[5]:.2f}%", flush=True)
    with open(os.path.join(OUT, "tleP30_isl.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["p_isl", "split_mean", "split_std", "minfill_mean", "minfill_std",
                    "totQ_mean", "down_mean"])
        w.writerows(rows)
    print("wrote tleP30_isl.csv", flush=True)
