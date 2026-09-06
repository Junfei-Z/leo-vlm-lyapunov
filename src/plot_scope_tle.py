#!/usr/bin/env python3
"""Scope/sensitivity figure (replaces the old 8-panel pair): 2 rows x 5 columns.
Top row = downtime% (log-ish scale, 5% SLA line) -> WHO IS DEPLOYABLE WHERE.
Bottom row = min-fill -> the fairness picture. Both rows are needed: min-fill
alone would mislead (Random's min-fill exceeds the proposed scheduler's at the
nominal point; it is disqualified by downtime, which only the top row shows).
Honest by construction: the battery and constellation panels SHOW Random
overtaking when resources are generous (the limitation's graphic evidence).
Data: data/scope_{pbase,battery,panel,arrival,nsat}.csv (locked config, 10 seeds).
"""
import os, sys, csv
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import apply_house_style, apply_paper_style, savefig_pub

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
PANELS = [
    ("pbase",   "platform load $P_{base}$ (W)",      None),
    ("battery", "battery $B^{max}$ (kJ)",            1e-3),
    ("panel",   "solar panel $P^{solar}$ (W)",       None),
    ("arrival", "arrival probability",               None),
]
POLS = ["Lyapunov (ours)", "Greedy-Q", "Greedy-E", "Random", "Static"]
STYLE = {
    "Lyapunov (ours)": dict(color="#274C77", marker="o", ls="-",  lab="Proposed"),
    "Greedy-Q":        dict(color="#B65C4A", marker="s", ls=":",  lab="QMax"),
    "Greedy-E":        dict(color="#2A9D8F", marker="^", ls="--", lab="EMin"),
    "Random":          dict(color="#C7952D", marker="D", ls="-.", lab="RFit"),
    "Static":          dict(color="#6C757D", marker="v", ls=(0, (3, 1, 1, 1)), lab="Static"),
}

def load(tag):
    rows = list(csv.DictReader(open(os.path.join(DATA, f"{PREFIX}{tag}.csv"))))
    out = {}
    for p in POLS:
        r = [x for x in rows if x["policy"] == p]
        r.sort(key=lambda x: float(x["value"]))
        out[p] = (np.array([float(x["value"]) for x in r]),
                  np.array([float(x["down_mean"]) for x in r]),
                  np.array([float(x["down_std"]) for x in r]),
                  np.array([float(x["minfill_mean"]) for x in r]),
                  np.array([float(x["minfill_std"]) for x in r]))
    return out

PREFIX = "tleP_"
OUTNAME = "tleP_sensitivity"

def main():
    apply_paper_style()
    fig, axes = plt.subplots(2, 4, figsize=(7.16, 3.15))
    for j, (tag, xlab, xscale) in enumerate(PANELS):
        d = load(tag)
        for p in POLS:
            x, dn, dns, mf, mfs = d[p]
            xs = x * xscale if xscale else x
            st = STYLE[p]
            z = 5 if st["lab"] == "Proposed" else 3
            axes[0, j].errorbar(xs, dn, yerr=dns, color=st["color"], marker=st["marker"],
                                ls=st["ls"], lw=1.1, ms=2.3, mew=0.6, capsize=1.5,
                                elinewidth=0.6, alpha=0.9, zorder=z, label=st["lab"])
            axes[1, j].errorbar(xs, mf, yerr=mfs, color=st["color"], marker=st["marker"],
                                ls=st["ls"], lw=1.1, ms=2.3, mew=0.6, capsize=1.5,
                                elinewidth=0.6, alpha=0.9, zorder=z, label=st["lab"])
        axes[0, j].axhline(5.0, color="#525960", lw=0.9, ls="--", alpha=0.75)
        axes[0, j].set_title("(%s)" % "abcd"[j], loc="left", fontsize=9)
        axes[1, j].set_xlabel(xlab)
        for ax in (axes[0, j], axes[1, j]):
            ax.grid(True, alpha=0.3)
    axes[0, 0].set_ylabel(r"downtime (\%) $\downarrow$")
    axes[1, 0].set_ylabel(r"max-min quality $\uparrow$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    sla = mpl.lines.Line2D([], [], color="#525960", lw=0.9, ls="--", alpha=0.75)
    handles.append(sla); labels.append("5% SLA")
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(HERE, "..", "figures", OUTNAME)
    savefig_pub(fig, out + ".pdf")
    fig.savefig(out + ".png", dpi=170, bbox_inches="tight")
    print("wrote", out + ".pdf/.png")

if __name__ == "__main__":
    main()
