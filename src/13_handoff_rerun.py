#!/usr/bin/env python3
"""P0-1 adjudication step: run the TLE-trace core at the declared-rule sizes with
the MEASURED handoff band accounted in the split timing.
Three points (net of instrument overhead, tier minus proxy0):
  0.25 s  -- band low end (2 ms one-way, measured 1G/2ms delta 0.246 s)
  0.60 s  -- band high end (5 ms one-way, measured 1G/5ms delta 0.577 s)
  0.69 s  -- 6 ms extrapolation via the 56-round-trip model (TLE same-plane
             neighbor at ~1800 km): EXPECTED to overflow the 0.625 s front-slot
             slack and shift the rear start by one slot -- run it honestly.
Slack threshold: front compute 1.375 s + h vs N_front*tau = 2.0 s -> h* = 0.625 s.
"""
import importlib.util, os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("splitpipe", os.path.join(HERE, "04_split_pipeline.py"))
M = importlib.util.module_from_spec(spec); sys.modules["splitpipe"] = M; spec.loader.exec_module(M)

npz = np.load(os.path.join(HERE, "..", "data", "tle_traces.npz"))
SUNLIT = npz["sunlit"]; ISL = npz["isl"]; T, S = SUNLIT.shape
S2 = np.concatenate([SUNLIT, SUNLIT], axis=0)
F = np.zeros((T, S), dtype=np.int32)
for s in range(S):
    col = S2[:, s]
    for t in range(T):
        if col[t]:
            j = t
            while j < 2 * T and col[j]:
                j += 1
            k = j
            while k < 2 * T and not col[k]:
                k += 1
            F[t, s] = k - j
        else:
            j = t
            while j < 2 * T and not col[j]:
                j += 1
            F[t, s] = j - t
M.sunlit_indicator = lambda t, sat: 1 if SUNLIT[t % T, sat] else 0
M.isl_connected = lambda t, s1, s2: bool(ISL[min(t, T - 1), s1, s2]) if s1 != s2 else False
M.eclipse_slots_to_sunrise = lambda t, s: int(F[t % T, s])
M.P_SOLAR = 12.2
M.B_MAX = 16580.0
M.B_INIT = 0.6 * 16580.0

SEEDS = list(range(7, 17))

def stat(name, V):
    a = {k: [] for k in ["blackout_slots", "min_fill", "split_tasks", "total_quality", "service_rate"]}
    for sd in SEEDS:
        r = M.run_sim(M.POLICIES[name], V=V, seed=sd)
        for k in a:
            a[k].append(r[k])
    TOT = M.N_SAT * M.HORIZON
    return (np.mean(a["blackout_slots"]) / TOT * 100, np.mean(a["min_fill"]),
            np.std(a["min_fill"]), np.mean(a["split_tasks"]), np.mean(a["total_quality"]))

if __name__ == "__main__":
    for h in [0.25, 0.60, 0.69]:
        M.HANDOFF_S = h
        extra = 0 if (M.CONFIGS["7B"]["T"] / 2.0 + h) <= M.SPLIT["N_front"] * M.TAU else 1
        print(f"=== HANDOFF {h:.2f}s (front {1.38 + h:.2f}s vs 2.0s window -> extra slot = {extra}) ===")
        print("%-18s %8s %14s %8s %8s" % ("policy", "down%", "min-fill", "split", "totQ"))
        for name, V in [("Lyapunov (ours)", 10), ("Greedy-Q", 100), ("Greedy-E", 100),
                        ("Random", 100), ("Static", 100)]:
            d, mf, mfs, sp, tq = stat(name, V)
            lab = "Proposed (V=10)" if "Lyap" in name else name
            print("%-18s %7.1f%% %.3f±%.3f %8.0f %8.0f" % (lab, d, mf, mfs, sp, tq))
    print("done")
