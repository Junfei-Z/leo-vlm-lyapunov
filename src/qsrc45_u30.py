#!/usr/bin/env python3
"""45-source (src45) core at RESERVE_FRAC=0.3, U_TGT=0.30 (mirrors src/17 src45 core)."""
import os, sys, json, csv
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_common import M, apply_config  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
Q45 = json.load(open(os.path.join(HERE, "..", "data", "q_src45.json")))
SEEDS = list(range(7, 17))
OUT = os.path.join(HERE, "..", "data", "qsrc")


def apply_src45():
    apply_config(dict(RESERVE_FRAC=0.3, U_TGT=0.30))
    classes = sorted(Q45)
    M.SRC_CLASS = classes
    M.Q_SRC = [dict(Q45[c]) for c in classes]
    M.N_SOURCES = 45
    M.VIS = [set(M.ALL_SATS) for _ in range(45)]
    M.ARRIVAL_PROB = 0.10 * 6 / 45


def stat(name, V):
    a = {k: [] for k in ["blackout_slots", "min_fill", "total_quality", "split_tasks"]}
    fills = []
    for sd in SEEDS:
        apply_src45()
        r = M.run_sim(M.POLICIES[name], V=V, seed=sd)
        for k in a:
            a[k].append(r[k])
        fills.append(r["fill"])
    fills = np.mean(np.array(fills), axis=0)      # per-source mean fill (45)
    TOT = M.N_SAT * M.HORIZON
    return (np.mean(a["blackout_slots"]) / TOT * 100, np.mean(a["min_fill"]),
            np.mean(a["total_quality"]), np.mean(a["split_tasks"]),
            float(np.percentile(fills, 10)), float(np.median(fills)), fills)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    unserv = [c for c in Q45 if max(Q45[c].values()) < 0.15]
    servable = [c for c in Q45 if c not in unserv]
    print(f"  unservable classes at U=0.30 (max_m Q < 0.15): {len(unserv)} -> {unserv}", flush=True)
    rows = []
    for name, V in [("Lyapunov (ours)", 1), ("Lyapunov (ours)", 10),
                    ("Lyapunov (ours)", 100), ("Greedy-Q", 100),
                    ("Greedy-E", 100), ("Random", 100), ("Static", 100)]:
        down, mf, totq, sp, q10, med, fills = stat(name, V)
        # servable-set max-min: min mean fill over the servable classes
        idx_serv = [i for i, c in enumerate(sorted(Q45)) if c in servable]
        serv_mf = float(fills[idx_serv].min())
        lab = f"Proposed(V={V})" if "Lyap" in name else name
        rows.append([lab, round(down, 4), round(mf, 4), round(serv_mf, 4),
                     round(totq, 1), round(sp, 1), round(q10, 4), round(med, 4)])
        print(f"  {lab:18s} down {down:.1f}% mf {mf:.3f} servable-min {serv_mf:.3f} "
              f"totQ {totq:.0f} split {sp:.0f} med {med:.3f}", flush=True)
    with open(os.path.join(OUT, "src45_core_u30.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "down_mean", "minfill_mean", "servable_minfill",
                    "totQ_mean", "split_mean", "fill_q10", "fill_med"])
        w.writerows(rows)
    print("wrote data/qsrc/src45_core_u30.csv", flush=True)
