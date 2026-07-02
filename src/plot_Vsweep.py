#!/usr/bin/env python3
"""V-knob figure (multi-satellite heterogeneous setting).
Two quantities vs the Lyapunov weight V:
  total delivered quality -- RISES with V (Theorem 2's O(1/V) improvement holds), and
  min-fill (max-min)      -- FALLS with V (the quality-fairness conflict).
The direction reversal of min-fill (up with V in the conflict-free single-sat case,
down with V here) is the V-space signature of the conflict; it does not violate
Theorem 2, which bounds total quality, not min-fill.
Data: locked config (P_base=7, P_solar=13, B=18000, N=4), 10-seed means.
"""
import os, sys
import matplotlib as mpl
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import apply_house_style, savefig_pub

V     = [1, 3, 10, 30, 100, 300]
TOTQ  = [3254, 3273, 3311, 3394, 3636, 3850]
MINF  = [0.175, 0.173, 0.169, 0.158, 0.124, 0.118]

def main():
    apply_house_style()
    mpl.rcParams.update({"axes.labelsize": 10, "xtick.labelsize": 8.5,
                         "ytick.labelsize": 8.5, "legend.fontsize": 7.5})
    fig, ax1 = plt.subplots(figsize=(3.6, 2.8))
    ax2 = ax1.twinx()

    l1 = ax1.plot(V, TOTQ, "-o", color="#0F4D92", lw=1.6, ms=4.5,
                  label="Total quality (rises: Thm.~2)")
    l2 = ax2.plot(V, MINF, "-s", color="#B64342", lw=1.6, ms=4.5,
                  label="Min-fill (falls: conflict)")

    ax1.set_xscale("log")
    ax1.set_xlabel("Lyapunov weight $V$")
    ax1.set_ylabel("Total delivered quality", color="#0F4D92")
    ax2.set_ylabel("Min-fill (max-min)", color="#B64342")
    ax1.tick_params(axis="y", labelcolor="#0F4D92")
    ax2.tick_params(axis="y", labelcolor="#B64342")
    ax1.grid(True, which="both", alpha=0.3)
    lns = l1 + l2
    ax1.legend(lns, [x.get_label() for x in lns], loc="center right", fontsize=7)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "..", "figures", "Vsweep_conflict")
    savefig_pub(fig, out + ".pdf")
    fig.savefig(out + ".png", dpi=200, bbox_inches="tight")
    print("wrote", out + ".pdf/.png")

if __name__ == "__main__":
    main()
