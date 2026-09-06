#!/usr/bin/env python3
"""Synthetic cross-plane contrast data generator (measured Q_SRC).
Replaces the previously hand-transcribed numbers in plot_frontier.py /
plot_Vsweep.py with a CSV (figure-reproducibility discipline).

Config = src/04 defaults exactly (staggered synthetic schedule, P_solar=13,
B_max=18000, h=0 legacy with band ends verified separately). 10 seeds.
Output: data/contrast_frontier.csv
  rows: label, V_or_-, down_mean, down_std, minfill_mean, minfill_std,
        totQ_mean, totQ_std, split_mean
"""
import importlib.util, os, sys, csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("splitpipe", os.path.join(HERE, "04_split_pipeline.py"))
M = importlib.util.module_from_spec(spec); sys.modules["splitpipe"] = M; spec.loader.exec_module(M)

SEEDS = list(range(7, 17))

def stat(name, V):
    a = {k: [] for k in ["blackout_slots", "min_fill", "total_quality", "split_tasks"]}
    for sd in SEEDS:
        r = M.run_sim(M.POLICIES[name], V=V, seed=sd)
        for k in a:
            a[k].append(r[k])
    TOT = M.N_SAT * M.HORIZON
    return (np.mean(a["blackout_slots"]) / TOT * 100, np.std(a["blackout_slots"]) / TOT * 100,
            np.mean(a["min_fill"]), np.std(a["min_fill"]),
            np.mean(a["total_quality"]), np.std(a["total_quality"]),
            np.mean(a["split_tasks"]))

if __name__ == "__main__":
    rows = []
    for V in [1, 3, 10, 30, 100, 300]:
        st = stat("Lyapunov (ours)", V)
        rows.append([f"Proposed", V, *[round(x, 4) for x in st]])
        print(f"  V={V}: down {st[0]:.1f}% mf {st[2]:.3f} totQ {st[4]:.0f}", flush=True)
    for name in ["Greedy-Q", "Greedy-E", "Random", "Static"]:
        st = stat(name, 100)
        rows.append([name, "-", *[round(x, 4) for x in st]])
        print(f"  {name}: down {st[0]:.1f}% mf {st[2]:.3f} totQ {st[4]:.0f}", flush=True)
    with open(os.path.join(HERE, "..", "data", "contrast_frontier.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "V", "down_mean", "down_std", "minfill_mean",
                    "minfill_std", "totQ_mean", "totQ_std", "split_mean"])
        w.writerows(rows)
    print("wrote data/contrast_frontier.csv", flush=True)
