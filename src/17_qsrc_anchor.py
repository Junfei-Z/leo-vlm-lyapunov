#!/usr/bin/env python3
"""Q_SRC anchor chain (paper_staging/QSRC_anchor_prereg.md, pre-registered).

B1 anchor6: TLE mainline with ONLY Q_SRC swapped to measured per-class values
            (same six classes, data/q_src45.json). Single variable.
B2 src45  : all 45 classes, zero selection, raw measured values; per-source
            arrival 0.10*6/45 so total offered load matches the mainline.
            Single variable relative to B1.

Usage: python3 src/17_qsrc_anchor.py {anchor6|src45} {core|battery<J>}
Outputs data/qsrc/{mode}_{what}.csv (+ fill quantiles columns for src45).
"""
import importlib.util, os, sys, json, csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("splitpipe", os.path.join(HERE, "04_split_pipeline.py"))
M = importlib.util.module_from_spec(spec); sys.modules["splitpipe"] = M; spec.loader.exec_module(M)

# ---- TLE overrides, identical to src/14 apply_primary ----
npz = np.load(os.path.join(HERE, "..", "data", "tle_traces.npz"))
SUNLIT = npz["sunlit"]; ISL = npz["isl"]; META = json.loads(str(npz["meta"]))
assert META["gate"] == "PASS"
T, S = SUNLIT.shape
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
BASE_ISL = ISL.copy()

Q45 = json.load(open(os.path.join(HERE, "..", "data", "q_src45.json")))
MAIN6 = ["beach", "intersection", "church", "baseball_diamond", "roundabout", "mountain"]

def apply_primary():
    M.sunlit_indicator = lambda t, sat: 1 if SUNLIT[t % T, sat] else 0
    M.isl_connected = lambda t, s1, s2: bool(BASE_ISL[min(t, T - 1), s1, s2]) if s1 != s2 else False
    M.eclipse_slots_to_sunrise = lambda t, s: int(F[t % T, s])
    M.P_SOLAR = 12.2
    M.B_MAX = 16580.0
    M.B_INIT = 0.6 * 16580.0
    M.P_BASE = 7.0
    M.ARRIVAL_PROB = 0.10
    M.HANDOFF_S = 0.69

def apply_anchor6():
    apply_primary()
    M.SRC_CLASS = list(MAIN6)
    M.Q_SRC = [dict(Q45[c]) for c in MAIN6]        # measured values, same classes
    M.N_SOURCES = 6
    M.VIS = [set(M.ALL_SATS) for _ in range(6)]

def apply_src45():
    apply_primary()
    classes = sorted(Q45)
    M.SRC_CLASS = classes
    M.Q_SRC = [dict(Q45[c]) for c in classes]      # zero selection, raw
    M.N_SOURCES = 45
    M.VIS = [set(M.ALL_SATS) for _ in range(45)]
    M.ARRIVAL_PROB = 0.10 * 6 / 45                 # declared: equal offered load

APPLY = {"anchor6": apply_anchor6, "src45": apply_src45}
SEEDS = list(range(7, 17))
POLS = ["Lyapunov (ours)", "Greedy-Q", "Greedy-E", "Random", "Static"]

def stat(name, V, apply_fn):
    keys = ["pcap_violations", "blackout_slots", "service_rate",
            "split_tasks", "min_fill", "total_quality"]
    a = {k: [] for k in keys}
    fills = []
    for sd in SEEDS:
        apply_fn()
        r = M.run_sim(M.POLICIES[name], V=V, seed=sd)
        for k in keys:
            a[k].append(r[k])
        fills.append(r["fill"])
    fills = np.mean(np.array(fills), axis=0)        # per-source mean fill
    TOT = M.N_SAT * M.HORIZON
    return dict(down=np.mean(a["blackout_slots"]) / TOT * 100,
                down_sd=np.std(a["blackout_slots"]) / TOT * 100,
                mf=np.mean(a["min_fill"]), mf_sd=np.std(a["min_fill"]),
                sp=np.mean(a["split_tasks"]), tq=np.mean(a["total_quality"]),
                sv=np.mean(a["service_rate"]), pkv=np.mean(a["pcap_violations"]),
                fill_q10=float(np.percentile(fills, 10)),
                fill_q25=float(np.percentile(fills, 25)),
                fill_med=float(np.median(fills)),
                n_starved=int((fills < 0.05).sum()))

OUT = os.path.join(HERE, "..", "data", "qsrc")
os.makedirs(OUT, exist_ok=True)
HDR = ["label", "pkv", "down_mean", "down_std", "service", "split",
       "minfill_mean", "minfill_std", "totQ", "fill_q10", "fill_q25",
       "fill_med", "n_starved"]

def row(lab, st):
    return [lab, st["pkv"], st["down"], st["down_sd"], st["sv"], st["sp"],
            st["mf"], st["mf_sd"], st["tq"], st["fill_q10"], st["fill_q25"],
            st["fill_med"], st["n_starved"]]

if __name__ == "__main__":
    mode, what = sys.argv[1], sys.argv[2]
    rows = []
    if what == "core":
        for name, V in [("Lyapunov (ours)", 1), ("Lyapunov (ours)", 10),
                        ("Lyapunov (ours)", 100), ("Greedy-Q", 100),
                        ("Greedy-E", 100), ("Random", 100), ("Static", 100)]:
            st = stat(name, V, APPLY[mode])
            lab = f"Proposed(V={V})" if "Lyap" in name else name
            rows.append(row(lab, st))
            print(f"  {mode} {lab:18s} down {st['down']:.1f}% mf {st['mf']:.3f} "
                  f"sp {st['sp']:.0f} tq {st['tq']:.0f} starved {st['n_starved']}", flush=True)
    elif what.startswith("battery"):
        v = float(what[len("battery"):])
        base_apply = APPLY[mode]
        def setter():
            base_apply()
            M.B_MAX = v; M.B_INIT = 0.6 * v
        for name in POLS:
            V = 10 if name == "Lyapunov (ours)" else 100
            st = stat(name, V, setter)
            rows.append(row(f"{name}@{v:.0f}", st))
            print(f"  {mode} B={v:.0f} {name:18s} down {st['down']:.1f}% mf {st['mf']:.3f}", flush=True)
    with open(os.path.join(OUT, f"{mode}_{what}.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(HDR); w.writerows(rows)
    print(f"wrote data/qsrc/{mode}_{what}.csv", flush=True)
