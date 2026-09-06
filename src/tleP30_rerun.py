#!/usr/bin/env python3
"""Re-run the TLE primary batch at the new feasible mainline target RESERVE_FRAC=0.3, U_TGT=0.30.

Same protocol as src/14_tle_primary.py (TLE trace, primary config), but with
M.U_TGT = 0.15.  Writes data/tleP30_{core,stability,panel,battery,pbase,arrival}.csv
so the original U=0.30 results remain untouched.
"""
import importlib.util, os, sys, json, csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exp_common import M, apply_config  # noqa: E402
from exp_common import SUNLIT, ISL, F, T  # noqa: E402

OUT = os.path.join(HERE, "..", "data")

SEEDS = list(range(7, 17))
POLS = ["Lyapunov (ours)", "Greedy-Q", "Greedy-E", "Random", "Static"]
U15 = 0.30


def apply_u15():
    apply_config(dict(RESERVE_FRAC=0.3, U_TGT=U15))


def stat(name, V, horizon=None, extra=None):
    a = {k: [] for k in ["pcap_violations", "blackout_slots", "service_rate",
                         "split_tasks", "min_fill", "total_quality"]}
    for sd in SEEDS:
        cfg = dict(RESERVE_FRAC=0.3, U_TGT=U15)
        if extra:
            cfg.update(extra)
        apply_config(cfg)
        kw = dict(V=V, seed=sd)
        if horizon:
            kw["horizon"] = horizon
        r = M.run_sim(M.POLICIES[name], **kw)
        for k in a:
            a[k].append(r[k])
    TOT = (M.N_SAT * (horizon or M.HORIZON))
    return dict(down=np.mean(a["blackout_slots"]) / TOT * 100,
                down_sd=np.std(a["blackout_slots"]) / TOT * 100,
                mf=np.mean(a["min_fill"]), mf_sd=np.std(a["min_fill"]),
                sp=np.mean(a["split_tasks"]), tq=np.mean(a["total_quality"]),
                sv=np.mean(a["service_rate"]), pkv=np.mean(a["pcap_violations"]))


def save(tag, rows, hdr):
    with open(os.path.join(OUT, f"tleP30_{tag}.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writeheader if False else None
        w.writerow(hdr); w.writerows(rows)
    print(f"wrote tleP30_{tag}.csv", flush=True)


def core():
    rows = []
    for name, V in [("Lyapunov (ours)", 1), ("Lyapunov (ours)", 10),
                    ("Lyapunov (ours)", 100), ("Greedy-Q", 100),
                    ("Greedy-E", 100), ("Random", 100), ("Static", 100)]:
        st = stat(name, V)
        lab = f"Proposed(V={V})" if "Lyap" in name else name
        rows.append([lab, st["pkv"], st["down"], st["down_sd"], st["sv"], st["sp"],
                     st["mf"], st["mf_sd"], st["tq"]])
        print(f"  {lab:18s} down {st['down']:.2f}% mf {st['mf']:.3f} sp {st['sp']:.0f} "
              f"tq {st['tq']:.0f}", flush=True)
    save("core", rows, ["policy", "pkv", "down_mean", "down_std", "service", "split",
                        "minfill_mean", "minfill_std", "totQ"])


def sweep(tag, setter, values):
    rows = []
    for v in values:
        for name in POLS:
            V = 10 if name == "Lyapunov (ours)" else 100
            st = stat(name, V, extra=setter(v))
            rows.append([v, name, st["down"], st["down_sd"], st["mf"], st["mf_sd"], st["sp"]])
        print(f"  {tag}={v} done", flush=True)
    save(tag, rows, ["value", "policy", "down_mean", "down_std", "minfill_mean", "minfill_std", "split_mean"])


def stability():
    apply_u15()
    H20 = M.N_SLOTS_PER_ORBIT * 20
    r = M.run_sim(M.POLICIES["Lyapunov (ours)"], V=10, horizon=H20, seed=7)
    qe = np.array(r["log"]["QE_mean"]); qu = np.array(r["log"]["QU_mean"])
    b0 = r["log"]["battery"][:, 0]
    np.savez_compressed(os.path.join(OUT, "tleP30_stability.npz"), qe=qe, qu=qu, b0=b0)
    print(f"  stability: QE/T={qe[-1]/H20:.2e} QU/T={qu[-1]/H20:.2e} "
          f"QEpeak={qe.max():.1f} QUpeak={qu.max():.1f} "
          f"blackout={r['blackout_slots']/(M.N_SAT*H20)*100:.2f}% "
          f"min_fill={r['min_fill']:.3f}", flush=True)


if __name__ == "__main__":
    print("=== TLE PRIMARY RE-RUN at RESERVE_FRAC=0.3, U_TGT=0.30 ===", flush=True)
    print("-- core --", flush=True); core()
    print("-- stability --", flush=True); stability()
    print("-- panel --", flush=True)
    sweep("panel", lambda v: {"P_SOLAR": float(v)},
          [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0])
    print("-- battery --", flush=True)
    def setB(v): return {"B_MAX": float(v), "B_INIT": 0.6 * float(v)}
    sweep("battery", setB, [14000, 15000, 16000, 17000, 18000, 19000, 20000, 21000, 22000])
    print("-- pbase --", flush=True)
    sweep("pbase", lambda v: {"P_BASE": float(v)}, [6, 7, 8, 9, 10, 11])
    print("-- arrival --", flush=True)
    sweep("arrival", lambda v: {"ARRIVAL_PROB": float(v)},
          [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70])
    print("ALL DONE", flush=True)
