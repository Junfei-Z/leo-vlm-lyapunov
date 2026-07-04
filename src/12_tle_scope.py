#!/usr/bin/env python3
"""P0-3 step 3 ((a)+sweep, advisor-mandated pairing): on the REAL TLE trace,
(1) re-derive the nominal sizing by the PRE-DECLARED rules
      panel  = sustainability floor x 1.114   (13 / 11.67 on the synthetic trace)
      battery = eclipse platform need x 1.128 (18000 / 15960 on the synthetic trace)
    applied to the trace's measured sunlit fraction and eclipse length, and run
    the core comparison there (the rules give the default point its legitimacy);
(2) run FULL B_max and P_solar sweeps under the TLE trace (the sweep gives the
    whole picture its honesty: where the deployability partition lives, where it
    dissolves, who wins outside). Doing (1) without (2) would be repackaged tuning.
Outputs: data/tle_scope_battery.csv, data/tle_scope_panel.csv + printed core table.
"""
import importlib.util, os, sys, json, csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("splitpipe", os.path.join(HERE, "04_split_pipeline.py"))
M = importlib.util.module_from_spec(spec); sys.modules["splitpipe"] = M; spec.loader.exec_module(M)

npz = np.load(os.path.join(HERE, "..", "data", "tle_traces.npz"))
SUNLIT = npz["sunlit"]; ISL = npz["isl"]; META = json.loads(str(npz["meta"]))
T, S = SUNLIT.shape
assert META["gate"] == "PASS"

S2 = np.concatenate([SUNLIT, SUNLIT], axis=0)
FORECAST = np.zeros((T, S), dtype=np.int32)
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
            FORECAST[t, s] = k - j
        else:
            j = t
            while j < 2 * T and not col[j]:
                j += 1
            FORECAST[t, s] = j - t

M.sunlit_indicator = lambda t, sat: 1 if SUNLIT[t % T, sat] else 0
M.isl_connected = lambda t, s1, s2: bool(ISL[min(t, T - 1), s1, s2]) if s1 != s2 else False
M.eclipse_slots_to_sunrise = lambda t, s: int(FORECAST[t % T, s])

# ---- (1) pre-declared sizing rules applied to the real trace ----
sunlit_frac = float(SUNLIT.mean())
floor_w = M.P_BASE / sunlit_frac
P_SOLAR_RULE = round(floor_w * (13.0 / (7.0 / 0.60)), 1)          # x1.114
ecl_need = float(FORECAST[:, 0][SUNLIT[:, 0]].max()) * M.P_BASE    # longest eclipse x P_base
B_RULE = round(ecl_need * (18000.0 / 15960.0), -1)                 # x1.128
print(f"declared-rule sizing on the real trace: sunlit={sunlit_frac:.3f} -> floor={floor_w:.2f} W")
print(f"  P_solar = {P_SOLAR_RULE} W   (rule: floor x 1.114)")
print(f"  B_max   = {B_RULE:.0f} J  (rule: eclipse need {ecl_need:.0f} J x 1.128)")

SEEDS = list(range(7, 17))
POLS = ["Lyapunov (ours)", "Greedy-Q", "Greedy-E", "Random", "Static"]

def set_sizes(psolar, bmax):
    M.P_SOLAR = float(psolar); M.B_MAX = float(bmax); M.B_INIT = 0.6 * float(bmax)

def stat(name, V):
    a = {k: [] for k in ["blackout_slots", "min_fill", "split_tasks", "total_quality", "service_rate"]}
    for sd in SEEDS:
        r = M.run_sim(M.POLICIES[name], V=V, seed=sd)
        for k in a:
            a[k].append(r[k])
    TOT = M.N_SAT * M.HORIZON
    return (np.mean(a["blackout_slots"]) / TOT * 100, np.std(a["blackout_slots"]) / TOT * 100,
            np.mean(a["min_fill"]), np.std(a["min_fill"]),
            np.mean(a["split_tasks"]), np.mean(a["total_quality"]), np.mean(a["service_rate"]))

def core_table():
    print("%-18s %11s %13s %8s %8s %9s" % ("policy", "down%", "min-fill", "split", "totQ", "service"))
    for V in [1, 10, 100]:
        d, ds, mf, mfs, sp, tq, sv = stat("Lyapunov (ours)", V)
        print("Proposed (V=%-3d)   %5.1f±%-4.1f %.3f±%.3f %8.0f %8.0f %9.3f" % (V, d, ds, mf, mfs, sp, tq, sv))
    for n in POLS[1:]:
        d, ds, mf, mfs, sp, tq, sv = stat(n, 100)
        print("%-18s %5.1f±%-4.1f %.3f±%.3f %8.0f %8.0f %9.3f" % (n, d, ds, mf, mfs, sp, tq, sv))

def sweep(tag, values, setter):
    rows = []
    for v in values:
        set_sizes(P_SOLAR_RULE, B_RULE)
        setter(v)
        for n in POLS:
            V = 10 if n == "Lyapunov (ours)" else 100
            d, ds, mf, mfs, sp, tq, sv = stat(n, V)
            rows.append([v, n, d, ds, mf, mfs, sp])
    path = os.path.join(HERE, "..", "data", f"tle_scope_{tag}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["value", "policy", "down_mean", "down_std", "minfill_mean", "minfill_std", "split_mean"])
        w.writerows(rows)
    print(f"wrote tle_scope_{tag}.csv")

if __name__ == "__main__":
    print("=== TLE core at DECLARED-RULE sizes (10 seeds) ===")
    set_sizes(P_SOLAR_RULE, B_RULE)
    core_table()
    print("=== B_max sweep under TLE trace ===")
    sweep("battery", [14000, 15000, 16350, 17500, 19000, 22000],
          lambda v: set_sizes(P_SOLAR_RULE, v))
    print("=== P_solar sweep under TLE trace ===")
    sweep("panel", [11.0, 11.6, 12.2, 13.0, 14.5, 18.0],
          lambda v: set_sizes(v, B_RULE))
    print("done")
