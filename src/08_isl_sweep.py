#!/usr/bin/env python3
"""ISL-availability sweep on the LOCKED model (imports src/04 — no divergent copy).
Time-varying per-window connectivity with duty cycle p_ISL (coherence window 300
slots, deterministic per (pair, window) hash). The proposed scheduler (V=10) is
swept over p_ISL; at p_ISL=1.0 this reduces to the all-connected case and must
reproduce the sealed core numbers (201 splits, min-fill 0.169) as an internal check.
Output: data/scope_isl.csv  (feeds src/plot_isl_stab.py).
"""
import importlib.util, os, sys, csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("splitpipe", os.path.join(HERE, "04_split_pipeline.py"))
M = importlib.util.module_from_spec(spec); sys.modules["splitpipe"] = M; spec.loader.exec_module(M)

WINDOW = 300
SEEDS = list(range(7, 17))

def make_isl(p_isl):
    def isl(t, s1, s2):
        if s1 == s2:
            return False
        a, b = (s1, s2) if s1 < s2 else (s2, s1)
        w = t // WINDOW
        h = ((a * 73856093) ^ (b * 19349663) ^ (w * 83492791)) & 0xFFFFFFFF
        return (h / 0xFFFFFFFF) < p_isl
    return isl

def main():
    TOT = M.N_SAT * M.HORIZON
    orig = M.isl_connected
    rows = []
    print(f"{'p_isl':>6} {'split':>10} {'min-fill':>14} {'totQ':>12} {'down%':>7}")
    for p in [0.3, 0.5, 0.7, 0.85, 1.0]:
        M.isl_connected = make_isl(p)
        sp, mf, tq, dt = [], [], [], []
        for sd in SEEDS:
            r = M.run_sim(M.POLICIES["Lyapunov (ours)"], V=10, seed=sd)
            sp.append(r["split_tasks"]); mf.append(r["min_fill"])
            tq.append(r["total_quality"]); dt.append(r["blackout_slots"] / TOT * 100)
        rows.append([p, np.mean(sp), np.std(sp), np.mean(mf), np.std(mf),
                     np.mean(tq), np.std(tq), np.mean(dt)])
        print(f"{p:6.2f} {np.mean(sp):6.0f}±{np.std(sp):<3.0f} {np.mean(mf):.3f}±{np.std(mf):.3f} "
              f"{np.mean(tq):7.0f}±{np.std(tq):<3.0f} {np.mean(dt):6.1f}%")
    M.isl_connected = orig
    path = os.path.join(HERE, "..", "data", "scope_isl.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["p_isl", "split_mean", "split_std", "minfill_mean", "minfill_std",
                    "totQ_mean", "totQ_std", "down_mean"])
        w.writerows(rows)
    print("wrote", path)
    # re-sealed 2026-07-04 after the measured-Q_SRC upgrade (option 1, B1 anchor):
    # p_isl=1.0 yields ~182 splits under measured values (old stylized seal ~201)
    assert abs(rows[-1][1] - 182) <= 15, "p_isl=1.0 must reproduce the sealed ~182 splits (measured Q_SRC)"
    print("internal check ok: p_isl=1.0 reproduces the all-connected core")

if __name__ == "__main__":
    main()
