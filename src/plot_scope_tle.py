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
from figstyle import apply_house_style, savefig_pub

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
    "Lyapunov (ours)": dict(color="#0F4D92", marker="o", ls="-",  lab="Proposed"),
    "Greedy-Q":        dict(color="#B64342", marker="s", ls=":",  lab="Greedy-Q"),
    "Greedy-E":        dict(color="#2E7D32", marker="^", ls="--", lab="Greedy-E"),
    "Random":          dict(color="#E58C00", marker="D", ls="-.", lab="Random"),
    "Static":          dict(color="#555555", marker="v", ls=(0, (3, 1, 1, 1)), lab="Static"),
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
    apply_house_style()
    mpl.rcParams.update({"axes.labelsize": 8.5, "axes.titlesize": 9,
                         "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                         "legend.fontsize": 8, "lines.linewidth": 1.4,
                         "lines.markersize": 3.6})
    fig, axes = plt.subplots(2, 4, figsize=(10.5, 4.6))
    for j, (tag, xlab, xscale) in enumerate(PANELS):
        d = load(tag)
        for p in POLS:
            x, dn, dns, mf, mfs = d[p]
            xs = x * xscale if xscale else x
            st = STYLE[p]
            axes[0, j].errorbar(xs, dn, yerr=dns, color=st["color"], marker=st["marker"],
                                ls=st["ls"], capsize=2, elinewidth=0.8, label=st["lab"])
            axes[1, j].errorbar(xs, mf, yerr=mfs, color=st["color"], marker=st["marker"],
                                ls=st["ls"], capsize=2, elinewidth=0.8, label=st["lab"])
        axes[0, j].axhline(5.0, color="k", lw=0.9, ls="--", alpha=0.6)
        if j == 0:
            axes[0, j].annotate("5% SLA", xy=(0.03, 0.62), xycoords="axes fraction", fontsize=7)
        axes[0, j].set_title("(%s)" % "abcd"[j], loc="left", fontsize=9)
        axes[1, j].set_xlabel(xlab)
        if tag == "nsat":
            for ax in (axes[0, j], axes[1, j]):
                ax.set_xscale("log", base=2)
                ax.set_xticks([2, 4, 8, 16]); ax.set_xticklabels(["2", "4", "8", "16"])
                ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
        for ax in (axes[0, j], axes[1, j]):
            ax.grid(True, alpha=0.3)
    axes[0, 0].set_ylabel("downtime (%)")
    axes[1, 0].set_ylabel("min-fill (max-min)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(HERE, "..", "figures", OUTNAME)
    savefig_pub(fig, out + ".pdf")
    fig.savefig(out + ".png", dpi=170, bbox_inches="tight")
    print("wrote", out + ".pdf/.png")

if __name__ == "__main__":
    main()
