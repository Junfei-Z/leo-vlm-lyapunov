#!/usr/bin/env python3
"""Aggregate the MPC matrix (data/mpc/run_*.csv, solves_*.csv) and adjudicate
the pre-registered criteria of paper_staging/MPC_baseline_design.md verbatim:
  (a) complexity line: MPC mean per-decision wall >= 100x proposed per-slot
  (b) real-time criterion: solve > tau = 1 s -> tier not per-slot real-time
  fix-3 usability: real-time tier with no incumbent or gap > 50% on > 20% of
      re-solves -> "not usable at the real-time budget"
Writes data/mpc_summary.csv; prints the filled reporting sentences.
"""
import os, csv, json, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MPC = os.path.join(HERE, "..", "data", "mpc")

# eclipse-boundary phase tagging: distance (slots) from a solve's t to the
# nearest sunlit/eclipse transition of ANY satellite (trace wraps at T)
npz = np.load(os.path.join(HERE, "..", "data", "tle_traces.npz"))
SUNLIT = npz["sunlit"]
T_TRACE = SUNLIT.shape[0]
BOUNDS = sorted({t for s in range(SUNLIT.shape[1])
                 for t in range(T_TRACE)
                 if SUNLIT[t, s] != SUNLIT[(t - 1) % T_TRACE, s]})
BOUNDS = np.array(BOUNDS)

def dist_to_eclipse_boundary(t):
    tm = t % T_TRACE
    d = np.abs(BOUNDS - tm)
    return int(min(d.min(), T_TRACE - d.max()))

TAU = 1.0   # slot length [s]; pre-registered real-time line

def rows(pattern):
    out = []
    for p in sorted(glob.glob(os.path.join(MPC, pattern))):
        with open(p) as f:
            r = csv.DictReader(f)
            out += list(r)
    return out

runs = rows("run_*.csv")
solves = rows("solves_*.csv")
with open(os.path.join(MPC, "proposed_decision_time.json")) as f:
    prop = json.load(f)

W_ORDER = ["half", "eclipse", "orbit"]
summary = []
print(f"proposed per-slot decision: mean {prop['mean_us']:.1f} us (n={prop['n']})\n")
for wk in W_ORDER:
    for tier in ["realtime", "oracle"]:
        rr = [r for r in runs if r["W"] == wk and r["tier"] == tier]
        ss = [s for s in solves if s["W_s"] and s["tier"] == tier
              and {"half": "1035", "eclipse": "2070", "orbit": "5736"}[wk] == s["W_s"]]
        if not rr:
            continue
        seeds = sorted(int(r["seed"]) for r in rr)
        g = lambda k: np.array([float(r[k]) for r in rr])
        wall = np.array([float(s["wall_s"]) for s in ss])
        tvals = np.array([int(s["t"]) for s in ss])
        noinc = sum(1 for s in ss if s["incumbent"] in ("", "None"))
        gaps = np.array([float(s["gap"]) for s in ss if s["gap"] not in ("", "None")])
        bad = noinc + int((gaps > 0.50).sum())
        frac_bad = bad / max(1, len(ss))
        # solve-time DISTRIBUTION (real-time systems are adjudicated on the tail)
        med, p95, mx = np.median(wall), np.percentile(wall, 95), wall.max()
        frac_tau = float((wall > TAU).mean())
        # eclipse-boundary clustering of hard windows (wall > tau)
        d_all = np.array([dist_to_eclipse_boundary(t) for t in tvals])
        hard = wall > TAU
        d_hard_med = float(np.median(d_all[hard])) if hard.any() else float("nan")
        d_all_med = float(np.median(d_all))
        # pre-registered adjudications
        per_dec_us = wall.mean() * 1e6
        ratio = per_dec_us / prop["mean_us"]
        amort_us = wall.mean() / 60 * 1e6          # amortized per executed slot
        a_verdict = "STANDS" if ratio >= 100 else "WEAKENS"
        b_verdict = (f"tau-line crossed on {frac_tau*100:.1f}% of solves"
                     if frac_tau > 0 else "no solve crossed tau")
        usable = "" if tier != "realtime" else (
            "NOT usable at real-time budget" if frac_bad > 0.20 else "usable at real-time budget")
        summary.append(dict(W=wk, tier=tier, n_seeds=len(seeds),
                            down=g("down_pct").mean(), down_sd=g("down_pct").std(),
                            mf=g("min_fill").mean(), mf_sd=g("min_fill").std(),
                            splits=g("splits").mean(), totQ=g("totQ").mean(),
                            solve_mean_s=round(wall.mean(), 3), solve_med_s=round(med, 3),
                            solve_p95_s=round(p95, 3), solve_max_s=round(mx, 2),
                            frac_over_tau=round(frac_tau, 4),
                            dmed_hard_slots=d_hard_med, dmed_all_slots=d_all_med,
                            no_incumbent=noinc, frac_gap_bad=round(frac_bad, 4),
                            ratio_raw=round(ratio, 1), amort_us=round(amort_us, 1),
                            a=a_verdict, b=b_verdict, usable=usable))
        print(f"W={wk:8s} {tier:8s} seeds={len(seeds):2d} down={g('down_pct').mean():.2f}%"
              f" mf={g('min_fill').mean():.4f}+-{g('min_fill').std():.4f}"
              f" splits={g('splits').mean():.0f} totQ={g('totQ').mean():.0f}")
        print(f"    solves: med {med:.2f}s p95 {p95:.2f}s max {mx:.1f}s"
              f" | >tau {frac_tau*100:.1f}% | hard-window dist-to-eclipse-boundary"
              f" med {d_hard_med:.0f} vs all {d_all_med:.0f} slots"
              f" | no-inc {noinc} gap>50% frac {frac_bad:.3f}")
        print(f"    (a) {ratio:.0f}x raw / {amort_us/prop['mean_us']:.0f}x amortized -> {a_verdict}"
              f" | (b) {b_verdict} {('| ' + usable) if usable else ''}")

with open(os.path.join(HERE, "..", "data", "mpc_summary.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
    w.writeheader(); w.writerows(summary)
print("\nwrote data/mpc_summary.csv")

# cross-seed pattern check (advisor): does MPC consistently spend safety
# margin to buy fairness (downtime > proposed AND min-fill > proposed)?
# Enters the limitation section ONLY if consistent across seeds.
PROP_DOWN, PROP_MF = 1.6, 0.174   # proposed V=10 TLE mainline (tleP_core)
print("\n--- pattern check: 'spends safety margin to buy fairness' ---")
for wk in W_ORDER:
    for tier in ["realtime", "oracle"]:
        rr = [r for r in runs if r["W"] == wk and r["tier"] == tier]
        if not rr:
            continue
        both = sum(1 for r in rr
                   if float(r["down_pct"]) > PROP_DOWN and float(r["min_fill"]) > PROP_MF)
        print(f"  W={wk:8s} {tier:8s}: {both}/{len(rr)} seeds with down>{PROP_DOWN}% AND mf>{PROP_MF}"
              f"  -> {'PATTERN HOLDS' if both == len(rr) else ('mixed' if both else 'absent')}")

# reporting sentence template (pre-registered (c)), filled per tier, oracle
print("\n--- filled reporting sentences (template (c), oracle tier) ---")
MAINLINE_MF = 0.174   # proposed V=10 TLE mainline (tleP_core)
for wk in W_ORDER:
    s = [x for x in summary if x["W"] == wk and x["tier"] == "oracle"]
    rt = [x for x in summary if x["W"] == wk and x["tier"] == "realtime"]
    if not s:
        continue
    s = s[0]
    z = s["ratio_raw"]
    pct = MAINLINE_MF / s["mf"] * 100 if s["mf"] > 0 else float("inf")
    print(f'W={wk}: "MPC with window W attains min-fill {s["mf"]:.3f} and downtime '
          f'{s["down"]:.1f}% at {z:.0f}x the per-decision compute of the proposed '
          f'scheduler; the proposed scheduler attains {pct:.0f}% of MPC\'s min-fill '
          f'at O(1) per-slot cost with no forecast machinery."')
