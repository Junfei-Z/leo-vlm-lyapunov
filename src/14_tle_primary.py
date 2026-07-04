#!/usr/bin/env python3
"""OPTION A, Phase 1: the FULL primary experiment batch on the TLE trace.
Primary config = TLE traces (src/10 gate PASS) + declared-rule sizes
(P_solar=12.2 W, B_max=16580 J) + measured handoff h=0.69 s (chain-geometry
value: 1800 km neighbors -> 6 ms one-way, extrapolated from the measured
round-trip model; geometry-derived, not selected).

Runs and persists (data/tleP_*.csv):
  core        -- Table IV numbers (Proposed V=1/10/100 + 4 baselines, 10 seeds)
  pbase / battery / panel / arrival sweeps (5 policies x 10 seeds each point)
  isl         -- p_ISL degradation sweep on top of the real trace (proposed)
  stability   -- 20-orbit queue traces (proposed V=10)
  idle        -- decomposition at the primary point (mechanism numbers)
nsat sweep needs per-size TLE chains (src/10 extension) -- separate follow-up.
"""
import importlib.util, os, sys, json, csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("splitpipe", os.path.join(HERE, "04_split_pipeline.py"))
M = importlib.util.module_from_spec(spec); sys.modules["splitpipe"] = M; spec.loader.exec_module(M)

# ---------------- TLE overrides ----------------
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

SEEDS = list(range(7, 17))
POLS = ["Lyapunov (ours)", "Greedy-Q", "Greedy-E", "Random", "Static"]

def stat(name, V):
    a = {k: [] for k in ["pcap_violations", "blackout_slots", "service_rate",
                         "split_tasks", "min_fill", "total_quality"]}
    for sd in SEEDS:
        r = M.run_sim(M.POLICIES[name], V=V, seed=sd)
        for k in a:
            a[k].append(r[k])
    TOT = M.N_SAT * M.HORIZON
    return dict(down=np.mean(a["blackout_slots"]) / TOT * 100,
                down_sd=np.std(a["blackout_slots"]) / TOT * 100,
                mf=np.mean(a["min_fill"]), mf_sd=np.std(a["min_fill"]),
                sp=np.mean(a["split_tasks"]), tq=np.mean(a["total_quality"]),
                sv=np.mean(a["service_rate"]), pkv=np.mean(a["pcap_violations"]))

def save(tag, rows, hdr):
    path = os.path.join(HERE, "..", "data", f"tleP_{tag}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(hdr); w.writerows(rows)
    print(f"wrote tleP_{tag}.csv", flush=True)

def core():
    rows = []
    for name, V in [("Lyapunov (ours)", 1), ("Lyapunov (ours)", 10), ("Lyapunov (ours)", 100),
                    ("Greedy-Q", 100), ("Greedy-E", 100), ("Random", 100), ("Static", 100)]:
        apply_primary()
        st = stat(name, V)
        lab = f"Proposed(V={V})" if "Lyap" in name else name
        rows.append([lab, st["pkv"], st["down"], st["down_sd"], st["sv"], st["sp"],
                     st["mf"], st["mf_sd"], st["tq"]])
        print(f"  {lab:ird 18s}" if False else f"  {lab:18s} down {st['down']:.1f}% mf {st['mf']:.3f} sp {st['sp']:.0f} tq {st['tq']:.0f}", flush=True)
    save("core", rows, ["policy", "pkv", "down_mean", "down_std", "service", "split", "minfill_mean", "minfill_std", "totQ"])

def sweep(tag, setter, values):
    rows = []
    for v in values:
        apply_primary()
        setter(v)
        for name in POLS:
            V = 10 if name == "Lyapunov (ours)" else 100
            st = stat(name, V)
            rows.append([v, name, st["down"], st["down_sd"], st["mf"], st["mf_sd"], st["sp"]])
        print(f"  {tag}={v} done", flush=True)
    save(tag, rows, ["value", "policy", "down_mean", "down_std", "minfill_mean", "minfill_std", "split_mean"])

def isl_sweep():
    WINDOW = 300
    rows = []
    for p in [0.3, 0.5, 0.7, 0.85, 1.0]:
        apply_primary()
        def isl(t, s1, s2, _p=p):
            if s1 == s2 or not BASE_ISL[min(t, T - 1), s1, s2]:
                return False
            a, b = (s1, s2) if s1 < s2 else (s2, s1)
            w = t // WINDOW
            h = ((a * 73856093) ^ (b * 19349663) ^ (w * 83492791)) & 0xFFFFFFFF
            return (h / 0xFFFFFFFF) < _p
        M.isl_connected = isl
        st = stat("Lyapunov (ours)", 10)
        rows.append([p, st["sp"], st["mf"], st["mf_sd"], st["tq"], st["down"]])
        print(f"  p_isl={p} done", flush=True)
    save("isl", rows, ["p_isl", "split_mean", "minfill_mean", "minfill_std", "totQ_mean", "down_mean"])

def stability():
    apply_primary()
    H20 = M.N_SLOTS_PER_ORBIT * 20
    r = M.run_sim(M.POLICIES["Lyapunov (ours)"], V=10, horizon=H20, seed=7)
    qe = np.array(r["log"]["QE_mean"]); qu = np.array(r["log"]["QU_mean"])
    b0 = r["log"]["battery"][:, 0]
    np.savez_compressed(os.path.join(HERE, "..", "data", "tleP_stability.npz"),
                        qe=qe, qu=qu, b0=b0)
    print(f"  stability: QE/T={qe[-1]/H20:.2e} QU/T={qu[-1]/H20:.2e} "
          f"QEpeak={qe.max():.1f} QUpeak={qu.max():.1f} "
          f"blackout={r['blackout_slots']/(M.N_SAT*H20)*100:.2f}%", flush=True)

def idle_decomposition():
    apply_primary()
    CNT = {}
    def pol(pending, free, b, omega, QE, QP, QU, V, rng, t):
        CNT["called"] += 1
        acts = M.feasible_actions(pending, free, b, omega, t)
        if not acts:
            CNT["no_feasible"] += 1; return None
        acts2 = [a for a in acts if M._keeps_reserve(a, b, omega, t)]
        if not acts2:
            CNT["reserve_blocked"] += 1; return None
        best = min(acts2, key=lambda a: M._score_dpp(a, b, omega, QE, QP, QU, V))
        if M._score_dpp(best, b, omega, QE, QP, QU, V) < 0:
            CNT["act"] += 1; return best
        CNT["score_idle"] += 1; return None
    agg = dict(called=0, act=0, score_idle=0, no_feasible=0, reserve_blocked=0)
    for sd in [7, 8, 9]:
        CNT.clear(); CNT.update(called=0, act=0, score_idle=0, no_feasible=0, reserve_blocked=0)
        M.run_sim(pol, V=10, seed=sd)
        for k in agg:
            agg[k] += CNT[k]
    print("  idle decomposition (3-seed mean):",
          {k: round(v / 3) for k, v in agg.items()}, flush=True)

if __name__ == "__main__":
    print("=== TLE PRIMARY BATCH (h=0.69, rule sizes) ===", flush=True)
    print("-- core --", flush=True); core()
    print("-- pbase --", flush=True); sweep("pbase", lambda v: setattr(M, "P_BASE", float(v)), [6, 7, 8, 9, 10, 11])
    print("-- battery --", flush=True)
    def setB(v): M.B_MAX = float(v); M.B_INIT = 0.6 * float(v)
    sweep("battery", setB, [14000, 15000, 16580, 17500, 19000, 22000])
    print("-- panel --", flush=True); sweep("panel", lambda v: setattr(M, "P_SOLAR", float(v)), [11.0, 11.6, 12.2, 13.0, 14.5, 18.0])
    print("-- arrival --", flush=True); sweep("arrival", lambda v: setattr(M, "ARRIVAL_PROB", float(v)), [0.05, 0.10, 0.20, 0.40, 0.70])
    print("-- isl --", flush=True); isl_sweep()
    print("-- stability --", flush=True); stability()
    print("-- idle --", flush=True); idle_decomposition()
    print("ALL DONE", flush=True)
