#!/usr/bin/env python3
"""Staggered-eclipse contrast frontier at U_TGT=0.15 (mirrors src/18)."""
import os, sys, csv
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_common import M, apply_config  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = list(range(7, 17))
OUT = os.path.join(HERE, "..", "data")


def apply_u15():
    """Contrast schedule = module defaults (staggered synthetic eclipse,
    P_solar=13, B_max=18000, h=0) exactly as src/18; only the target changes."""
    M.U_TGT = 0.30; M.RESERVE_FRAC = 0.3


def stat(name, V):
    a = {k: [] for k in ["blackout_slots", "min_fill", "mean_fill", "total_quality", "split_tasks"]}
    for sd in SEEDS:
        apply_u15()
        r = M.run_sim(M.POLICIES[name], V=V, seed=sd)
        for k in a:
            a[k].append(r[k])
    TOT = M.N_SAT * M.HORIZON
    return (np.mean(a["blackout_slots"]) / TOT * 100, np.std(a["blackout_slots"]) / TOT * 100,
            np.mean(a["min_fill"]), np.std(a["min_fill"]),
            np.mean(a["mean_fill"]), np.mean(a["total_quality"]),
            np.std(a["total_quality"]), np.mean(a["split_tasks"]))


if __name__ == "__main__":
    rows = []
    for V in [1, 3, 10, 30, 100, 300]:
        st = stat("Lyapunov (ours)", V)
        rows.append(["Proposed", V, *[round(x, 4) for x in st]])
        print(f"  V={V}: down {st[0]:.1f}% mf {st[2]:.3f} meanQ {st[4]:.3f} "
              f"totQ {st[5]:.0f}", flush=True)
    for name in ["Greedy-Q", "Greedy-E", "Random", "Static"]:
        st = stat(name, 100)
        rows.append([name, "-", *[round(x, 4) for x in st]])
        print(f"  {name}: down {st[0]:.1f}% mf {st[2]:.3f} meanQ {st[4]:.3f} "
              f"totQ {st[5]:.0f}", flush=True)
    with open(os.path.join(OUT, "contrast_frontier_u30.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "V", "down_mean", "down_std", "minfill_mean",
                    "minfill_std", "meanQ_mean", "totQ_mean", "totQ_std", "split_mean"])
        w.writerows(rows)
    print("wrote data/contrast_frontier_u30.csv", flush=True)
