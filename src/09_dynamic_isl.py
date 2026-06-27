#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EXP-3 : Dynamic ISL topology  (makes the handoff constraint Eq.16 non-trivial, C1)
==================================================================================

In src/04 the inter-satellite link indicator is a placeholder (all pairs always
connected), so the pipeline handoff constraint Eq.16 never binds. Here we replace it
with a deterministic but TIME-VARYING +Grid-style connectivity: each inter-satellite
link is up or down over coherence windows, with a controllable availability fraction
p_isl. A two-stage 8B split can only be committed if the link is up at BOTH the
commit slot and the activation-handoff slot t + N_front (Eq.16).

We sweep p_isl and rerun the split-pipeline scenario under the proposed scheduler.
Honest finding (from the real runs below): as ISL availability drops, fewer splits
are feasible (split count 1100 -> 461), so Eq.16 genuinely binds. The scheduler
reroutes the blocked splits to cheaper standalone configs, so throughput and the
max-min fill do NOT fall (they slightly rise, because the 8B split is resource-
expensive) and blackout stays at zero. The cost of intermittent ISL is therefore
borne as fewer top-tier 8B inferences, not as lost throughput or lost safety.

Reuses the faithful model in src/06_sensitivity.py (same configs, solar, scheduler);
only isl_connected is overridden. At p_isl = 1.0 it reduces to the all-connected
case and reproduces the src/04 split count (~1100) as an internal check.
"""

import os, sys, csv, importlib
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import apply_house_style, savefig_pub, PALETTE
apply_house_style()

sens = importlib.import_module("06_sensitivity")

WINDOW = 300          # topology coherence window [slots]


def make_isl(p_isl):
    """Deterministic time-varying inter-satellite connectivity with duty cycle p_isl.
    Same (pair, window) -> same up/down across runs (reproducible)."""
    def isl(t, s1, s2):
        if s1 == s2:
            return False
        a, b = (s1, s2) if s1 < s2 else (s2, s1)
        w = t // WINDOW
        h = (a * 73856093) ^ (b * 19349663) ^ (w * 83492791)
        h &= 0xFFFFFFFF
        return (h / 0xFFFFFFFF) < p_isl
    return isl


def main():
    print("=" * 78)
    print("  EXP-3  Dynamic ISL topology (handoff constraint Eq.16)")
    print("=" * 78)
    prm = sens.Params()      # default 4-sat split-pipeline scenario
    Ps = [0.3, 0.5, 0.7, 0.85, 1.0]
    rows = []
    print(f"  {'p_isl':>6s} | {'split#':>7s} | {'min_fill':>8s} | {'totQ':>8s} | "
          f"{'blackout':>8s} | {'peakViol':>8s}")
    for p in Ps:
        sens.isl_connected = make_isl(p)          # patch the module global used by choose_lyapunov
        r = sens.run_sim(sens.choose_lyapunov, prm)
        rows.append((p, r["split_tasks"], r["min_fill"], r["total_quality"],
                     r["blackout_slots"], r["pcap_violations"]))
        print(f"  {p:6.2f} | {r['split_tasks']:7d} | {r['min_fill']:8.3f} | "
              f"{r['total_quality']:8.1f} | {r['blackout_slots']:8d} | {r['pcap_violations']:8d}")
    sens.isl_connected = make_isl(1.0)            # restore all-connected

    os.makedirs("data", exist_ok=True)
    with open("data/results_dynamic_isl.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["p_isl", "split_tasks", "min_fill", "total_quality",
                    "blackout_slots", "peak_violations"])
        w.writerows(rows)
    print("  saved data/results_dynamic_isl.csv")

    ps = np.array([r[0] for r in rows])
    splits = np.array([r[1] for r in rows])
    minf = np.array([r[2] for r in rows])
    totq = np.array([r[3] for r in rows])

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].plot(ps, splits, "o-", color=PALETTE["blue_main"], lw=2.5, ms=7)
    ax[0].set_xlabel("ISL availability fraction $p_{\\mathrm{ISL}}$")
    ax[0].set_ylabel("7B split inferences")
    ax[0].set_title("(a)", loc="left")
    axq = ax[1]
    axq.plot(ps, totq, "o-", color=PALETTE["blue_main"], lw=2.5, ms=7, label="total quality")
    axq.set_xlabel("ISL availability fraction $p_{\\mathrm{ISL}}$")
    axq.set_ylabel("cumulative quality")
    axt = axq.twinx()
    axt.plot(ps, minf, "s--", color=PALETTE["red_strong"], lw=2, ms=6, label="min-fill")
    axt.set_ylabel("min-fill (max-min)")
    axt.spines["top"].set_visible(False)
    axq.set_title("(b)", loc="left")
    lines = axq.get_lines() + axt.get_lines()
    axq.legend(lines, [l.get_label() for l in lines], loc="lower right")
    savefig_pub(fig, "dynamic_isl.pdf")
    fig.savefig("dynamic_isl.png", dpi=130)
    print("  saved dynamic_isl.pdf")
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
