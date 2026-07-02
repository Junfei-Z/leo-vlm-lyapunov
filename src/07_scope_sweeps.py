#!/usr/bin/env python3
"""Scope-evidence sweeps for the locked config, driven by the SAME locked model as
src/04 (imported, not re-implemented, so the two can never diverge). Each sweep sets
the relevant global (and its derived globals) on the src/04 module and runs 10 seeds.
Saves data/scope_{param}.csv. These are the honest-scope sweeps: they show WHERE the
deployability advantage holds and where it does not (narrow bands), not a universal win.
"""
import importlib.util, os, sys, csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("splitpipe", os.path.join(HERE, "04_split_pipeline.py"))
M = importlib.util.module_from_spec(spec); sys.modules["splitpipe"] = M; spec.loader.exec_module(M)

SEEDS = list(range(7, 17))
V_PROP = 10   # a fairness-favouring operating point; the paper reports the full V curve elsewhere
POLICIES = ["Lyapunov (ours)", "Greedy-Q", "Greedy-E", "Random", "Static"]

def metrics(name):
    V = V_PROP if name == "Lyapunov (ours)" else 100
    tot = M.N_SAT * M.HORIZON          # dynamic: N_SAT may be swept
    b, mf, sp = [], [], []
    for sd in SEEDS:
        r = M.run_sim(M.POLICIES[name], V=V, seed=sd)
        b.append(r["blackout_slots"] / tot * 100); mf.append(r["min_fill"]); sp.append(r["split_tasks"])
    return np.mean(b), np.std(b), np.mean(mf), np.std(mf), np.mean(sp)

def set_param(field, v):
    if field == "B_MAX":
        M.B_MAX = float(v); M.B_INIT = 0.6 * float(v)          # derived global must track
    elif field == "N_SAT":
        M.N_SAT = int(v)
        M.SAT_PHASE = [int(k * M.N_SLOTS_PER_ORBIT / int(v)) for k in range(int(v))]  # derived
    else:
        setattr(M, field, float(v) if field != "N_SOURCES" else int(v))

def restore():
    M.P_BASE, M.P_SOLAR, M.B_MAX = 7.0, 13.0, 18000.0; M.B_INIT = 0.6 * 18000
    M.ARRIVAL_PROB = 0.10; M.N_SAT = 4
    M.SAT_PHASE = [int(k * M.N_SLOTS_PER_ORBIT / 4) for k in range(4)]

SWEEPS = [
    ("pbase",  "P_BASE",       [6, 7, 8, 9, 10, 11]),
    ("battery","B_MAX",        [16000, 17000, 18000, 19000, 20000, 24000]),
    ("panel",  "P_SOLAR",      [11, 12, 13, 15, 18, 25]),
    ("arrival","ARRIVAL_PROB", [0.05, 0.10, 0.20, 0.40, 0.70]),
    ("nsat",   "N_SAT",        [2, 4, 8, 16]),
]

def main():
    for tag, field, vals in SWEEPS:
        rows = []
        for v in vals:
            restore(); set_param(field, v)
            for name in POLICIES:
                db, ds, mf, mfs, sp = metrics(name)
                rows.append([v, name, db, ds, mf, mfs, sp])
        restore()
        path = os.path.join(HERE, "..", "data", f"scope_{tag}.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["value", "policy", "down_mean", "down_std", "minfill_mean", "minfill_std", "split_mean"])
            w.writerows(rows)
        print(f"wrote scope_{tag}.csv ({len(vals)} pts x {len(POLICIES)} policies)")

if __name__ == "__main__":
    main()
