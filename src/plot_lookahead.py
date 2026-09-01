#!/usr/bin/env python3
"""Value and price of lookahead (from paper Table V, real-time regime).
x = planning window W; top = max-min quality; bottom = per-decision compute
(log scale) against the tau = 1 s slot. W = 0 is the proposed per-slot rule."""
import os, sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import apply_house_style, apply_paper_style, savefig_pub

ORBIT_BLUE, MPC_TEAL = "#274C77", "#4F7C82"
LABELS = ["0\n(per-slot)", "1035\n(half-eclipse)", "2070\n(eclipse)", "5736\n(orbit)", "$\\infty$\n(offline MILP)"]
QUAL   = [0.165, 0.153, 0.172, 0.269]
COMP   = [27e-6, 0.11, 0.23, 0.81]     # median per-decision wall clock [s]
P95    = [27e-6, 0.14, 0.71, 1.44]     # p95 per-decision wall clock [s]

def main():
    apply_paper_style()
    x = np.arange(4)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.5, 2.85), sharex=True)
    for a in (a1, a2):
        a.axvspan(3.6, 4.4, color="#D9DEE2", alpha=0.55, lw=0)
    a1.annotate("intractable\nat scale", xy=(4, 0.13), fontsize=6, color="0.35",
                ha="center", va="center")
    a2.annotate("intractable\nat scale", xy=(4, 3e-3), fontsize=6, color="0.35",
                ha="center", va="center")

    colors = [ORBIT_BLUE, MPC_TEAL, MPC_TEAL, MPC_TEAL]
    a1.bar(x, QUAL, width=0.55, color=colors, edgecolor="white", linewidth=0.5)
    a1.set_ylabel(r"max-min quality$\,\uparrow$")
    a1.set_ylim(0, 0.30)
    import matplotlib.patches as mpatches
    a1.legend(handles=[mpatches.Patch(color=ORBIT_BLUE, label="Proposed"),
                       mpatches.Patch(color=MPC_TEAL, label="MPC")],
              loc="upper left", frameon=False, fontsize=7, handlelength=1.2)

    a2.bar(x, COMP, width=0.55, color=colors, edgecolor="white", linewidth=0.5,
           log=True)
    a2.axhline(1.0, color="#525960", ls="--", lw=0.9, alpha=0.75)
    a2.annotate(r"slot length $\tau$", xy=(-0.35, 1.5), fontsize=6.5, color="0.3")
    a2.set_ylim(8e-6, 6)
    a2.set_ylabel(r"$\downarrow\,$compute (s)")
    a2.set_xlabel(r"planning window $W$ (s)")
    a2.set_xticks(np.arange(5))
    a2.set_xticklabels(LABELS, fontsize=7)
    a2.set_xlim(-0.6, 4.5)

    for a in (a1, a2):
        a.grid(True, alpha=0.3, axis="y")
    fig.tight_layout(pad=0.5)
    out = os.path.join(os.path.dirname(__file__), "..", "figures", "lookahead_value")
    savefig_pub(fig, out + ".pdf")
    fig.savefig(out + ".png", dpi=170, bbox_inches="tight")
    print("wrote", out)

if __name__ == "__main__":
    main()
