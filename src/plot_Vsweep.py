#!/usr/bin/env python3
"""V-knob figure (multi-satellite heterogeneous setting).
Two quantities vs the Lyapunov weight V:
  total delivered quality -- RISES with V (Theorem 2's O(1/V) improvement holds), and
  max-min quality (max-min)      -- FALLS with V (the quality-fairness conflict).
The direction reversal of max-min quality (up with V in the conflict-free single-sat case,
down with V here) is the V-space signature of the conflict; it does not violate
Theorem 2, which bounds total quality, not max-min quality.
Data: locked config (P_base=7, P_solar=13, B=18000, N=4), 10-seed means.
"""
import os, sys
import matplotlib as mpl
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import apply_house_style, apply_paper_style, savefig_pub

# measured Q_SRC values, from data/contrast_frontier.csv (src/18)
import csv as _csv
_rows = [r for r in _csv.DictReader(open(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data",
    "contrast_frontier.csv"))) if r["label"] == "Proposed"]
V     = [int(r["V"]) for r in _rows]
TOTQ  = [float(r["totQ_mean"]) for r in _rows]
MINF  = [float(r["minfill_mean"]) for r in _rows]

def main():
    apply_paper_style()
    fig, ax1 = plt.subplots(figsize=(3.5, 2.9))
    ax2 = ax1.twinx()

    l1 = ax1.plot(V, TOTQ, "-o", color="#0F4D92", lw=1.6, ms=4.5,
                  label="Total quality (rises)")
    l2 = ax2.plot(V, MINF, "-s", color="#B64342", lw=1.6, ms=4.5,
                  label="Min-fill (falls)")

    ax1.set_xscale("log")
    ax1.set_xlabel("Lyapunov weight $V$")
    ax1.set_ylabel(r"total delivered quality $\uparrow$", color="#0F4D92")
    ax2.set_ylabel(r"max-min quality $\uparrow$", color="#B64342")
    ax1.tick_params(axis="y", labelcolor="#0F4D92")
    ax2.tick_params(axis="y", labelcolor="#B64342")
    ax1.grid(True, which="both", alpha=0.3)
    lns = l1 + l2
    # both curves cross mid-plot and occupy all four corners -> legend above the axes
    ax1.legend(lns, [x.get_label() for x in lns], loc="lower center",
               bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False, fontsize=7.5,
               columnspacing=1.2, handlelength=1.6)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "..", "figures", "Vsweep_conflict")
    savefig_pub(fig, out + ".pdf")
    fig.savefig(out + ".png", dpi=200, bbox_inches="tight")
    print("wrote", out + ".pdf/.png")

if __name__ == "__main__":
    main()
