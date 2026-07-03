#!/usr/bin/env python3
"""P0-3 step 2: run the LOCKED sim (src/04, imported) on the TLE-driven traces
from src/10 (data/tle_traces.npz), overriding exactly three things:
  sunlit_indicator, isl_connected, eclipse_slots_to_sunrise (the reserve guard's
  deterministic forecast, recomputed from the trace so the guard forecasts the
  REAL ephemeris, consistent with "computable offline from TLE data").
Everything else (configs, energy model, policies, seeds) is untouched.

Output: the TLE-version core 4-sat comparison (10 seeds). These numbers become
the new sealed baseline for the TMC revision; shifts vs the synthetic schedule
are EXPECTED (real trace has ~0.639 sunlit and SIMULTANEOUS eclipse across the
chain, an energy-harder structure than the synthetic staggered phases).
"""
import importlib.util, os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("splitpipe", os.path.join(HERE, "04_split_pipeline.py"))
M = importlib.util.module_from_spec(spec); sys.modules["splitpipe"] = M; spec.loader.exec_module(M)

npz = np.load(os.path.join(HERE, "..", "data", "tle_traces.npz"))
SUNLIT = npz["sunlit"]          # [T, S] bool
ISL = npz["isl"]                # [T, S, S] bool
META = json.loads(str(npz["meta"]))
T, S = SUNLIT.shape
assert S == M.N_SAT and T == M.HORIZON, (SUNLIT.shape, M.N_SAT, M.HORIZON)
print("TLE traces:", META["sats"], "| beta=%.1f | sunlit=%s | gate=%s"
      % (META["beta"], np.round(META["sunlit_frac"], 3), META["gate"]))
assert META["gate"] == "PASS", "physical gate failed -- refuse to run"

# ---- eclipse forecast from the trace (periodic extension for the seam) ----
S2 = np.concatenate([SUNLIT, SUNLIT], axis=0)
FORECAST = np.zeros((T, S), dtype=np.int32)
for s in range(S):
    col = S2[:, s]
    for t in range(T):
        if col[t]:
            # next eclipse window length
            j = t
            while j < 2 * T and col[j]:
                j += 1
            k = j
            while k < 2 * T and not col[k]:
                k += 1
            FORECAST[t, s] = k - j
        else:
            j = t
            while j < 2 * T and not col[j]:
                j += 1
            FORECAST[t, s] = j - t

M.sunlit_indicator = lambda t, sat: 1 if SUNLIT[t % T, sat] else 0
M.isl_connected = lambda t, s1, s2: bool(ISL[min(t, T - 1), s1, s2]) if s1 != s2 else False
M.eclipse_slots_to_sunrise = lambda t, s: int(FORECAST[t % T, s])

TOT = M.N_SAT * M.HORIZON

def row(name, V):
    a = {k: [] for k in ["pcap_violations", "blackout_slots", "service_rate",
                         "split_tasks", "min_fill", "total_quality"]}
    for sd in range(7, 17):
        r = M.run_sim(M.POLICIES[name], V=V, seed=sd)
        for k in a:
            a[k].append(r[k])
    m = {k: np.mean(v) for k, v in a.items()}
    sdv = {k: np.std(v) for k, v in a.items()}
    print("%-18s %4.0f %6.1f±%-4.1f %.3f±%.3f %5.0f±%-4.0f %.3f±%.3f %6.0f±%-4.0f" % (
        name if V == 100 or "Lyap" not in name else f"Proposed (V={V})",
        m["pcap_violations"], m["blackout_slots"] / TOT * 100, sdv["blackout_slots"] / TOT * 100,
        m["service_rate"], sdv["service_rate"], m["split_tasks"], sdv["split_tasks"],
        m["min_fill"], sdv["min_fill"], m["total_quality"], sdv["total_quality"]))

if __name__ == "__main__":
    print("=== TLE-driven core comparison (locked config, 10 seeds) ===")
    print("%-18s %4s %11s %11s %10s %13s %11s" % ("policy", "pkv", "down%", "service", "split", "min-fill", "totQ"))
    for V in [1, 10, 100]:
        row("Lyapunov (ours)", V)
    for n in ["Greedy-Q", "Greedy-E", "Random", "Static"]:
        row(n, 100)
