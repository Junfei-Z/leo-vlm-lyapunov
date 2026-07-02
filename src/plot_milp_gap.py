#!/usr/bin/env python3
"""Standalone replot of the MILP optimality-gap figure from
data/results_milp_gap.csv (no MILP re-solve). Identical curves to
src/05_milp_gap.py; only the legend placement is fixed:
(a) legend -> upper left (clear of the rising curve + std band),
(b) legend -> lower left (clear of the O(1/V) reference dashes).
"""
import os, sys, csv
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import apply_house_style, savefig_pub, PALETTE

HERE = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.reader(open(os.path.join(HERE, "..", "data", "results_milp_gap.csv"))))
data = np.array([[float(x) for x in r] for r in rows[2:]])
Varr, mean_gap, std_gap, minq, std_q = (data[:, i] for i in range(5))

def main():
    apply_house_style()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].axhline(1.0, color=PALETTE["neutral"], ls="--", lw=2, label=r"offline optimum $U^\star$")
    ax[0].plot(Varr, minq, "o-", color=PALETTE["blue_main"], lw=2.5, ms=7, label="online (ours)")
    ax[0].fill_between(Varr, minq - std_q, np.minimum(1.0, minq + std_q),
                       color=PALETTE["blue_main"], alpha=0.15)
    ax[0].set_xscale("log")
    ax[0].set_xlabel(r"control parameter $V$")
    ax[0].set_ylabel(r"online min-quality $/\ U^\star$")
    ax[0].set_title("(a)", loc="left")
    ax[0].legend(loc="upper left", fontsize=16)

    ax[1].plot(Varr, mean_gap, "o-", color=PALETTE["blue_main"], lw=2.5, ms=7,
               label=r"normalized gap")
    # Anchor the 1/V guide in the decay regime (V=100). Anchoring at argmin picked
    # the V=500 point where the gap is ~0, making c~0 and the guide invisible.
    j0 = int(np.argmin(np.abs(Varr - 100)))
    c = mean_gap[j0] * Varr[j0]
    mask = Varr >= 20
    ax[1].plot(Varr[mask], c / Varr[mask], "--", color=PALETTE["red_strong"], lw=2.5,
               label=r"$O(1/V)$ reference")
    ax[1].set_xscale("log")
    ax[1].set_ylim(0, float(mean_gap.max()) * 1.15)
    ax[1].set_xlabel(r"control parameter $V$")
    ax[1].set_ylabel(r"$(U^\star-$online$)\,/\,U^\star$")
    ax[1].set_title("(b)", loc="left")
    ax[1].legend(loc="lower left", fontsize=16)

    savefig_pub(fig, os.path.join(HERE, "..", "milp_gap.pdf"))
    fig.savefig(os.path.join(HERE, "..", "milp_gap.png"), dpi=130)
    print("wrote milp_gap.pdf/.png")

if __name__ == "__main__":
    main()
