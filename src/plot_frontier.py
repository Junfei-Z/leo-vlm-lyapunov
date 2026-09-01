#!/usr/bin/env python3
"""Publication figure for the max-min quality and mean-quality frontier."""
import os, sys
import matplotlib as mpl
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import apply_house_style, apply_paper_style, savefig_pub

# measured Q_SRC values, read from data/contrast_frontier.csv (src/18)
import csv as _csv
_rows = list(_csv.DictReader(open(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "contrast_frontier.csv"))))
_NORM = 6 * 0.10 * 17100.0   # sources x arrival prob x contrast horizon slots
PROP = [(float(r["totQ_mean"]) / _NORM, float(r["minfill_mean"]), r["V"])
        for r in _rows if r["label"] == "Proposed"]
SLA = 5.0
DISPLAY = {"Random": "RFit", "Greedy-Q": "QMax", "Greedy-E": "EMin"}
BASE = [(r["label"], float(r["totQ_mean"]) / _NORM, float(r["minfill_mean"]),
         float(r["down_mean"]), float(r["down_mean"]) < SLA)
        for r in _rows if r["label"] != "Proposed"]

def main():
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.65))

    navy = "#244A73"
    slate = "#667784"
    rust = "#A65A46"
    grid = "#D9DEE3"
    styles = {
        "Greedy-Q": ("s", "QMax", rust, (-5, -12), "right"),
        "Greedy-E": ("^", "EMin", slate, (5, -4), "left"),
        "Random":   ("D", "RFit", rust, (-7, 5), "right"),
        "Static":   ("v", "Static", slate, (5, -4), "left"),
    }

    xs = [p[0] for p in PROP]
    ys = [p[1] for p in PROP]
    ax.plot(xs, ys, color=navy, lw=1.8, zorder=3)
    ax.scatter(xs, ys, s=24, color=navy, edgecolor="white",
               linewidth=0.55, zorder=4)

    v_index = {p[2]: i for i, p in enumerate(PROP)}
    for value, offset, align in [
        ("1", (-5, 7), "left"),
        ("30", (4, 5), "left"),
        ("300", (2, -12), "center"),
    ]:
        i = v_index[value]
        ax.annotate(fr"$V={value}$", (xs[i], ys[i]),
                    xytext=offset, textcoords="offset points",
                    fontsize=6.8, color=navy, ha=align)

    for name, x, y, dt, deployable in BASE:
        marker, label, color, offset, align = styles[name]
        face = color if deployable else "white"
        ax.scatter(x, y, marker=marker, s=38, facecolor=face,
                   edgecolor=color, linewidth=1.25, zorder=5)
        ax.annotate(label, (x, y), xytext=offset,
                    textcoords="offset points", fontsize=6.8,
                    color=color, ha=align)

    orbit_handle = mpl.lines.Line2D([], [], color=navy, marker="o", lw=1.8,
                                    ms=4, label="ORBIT, $V$ sweep")
    feasible_handle = mpl.lines.Line2D([], [], marker="o", mfc=slate,
                                       mec=slate, ls="none", ms=4.5,
                                       label="SLA compliant")
    violation_handle = mpl.lines.Line2D([], [], marker="o", mfc="white",
                                        mec=rust, ls="none", ms=4.5,
                                        mew=1.2, label="SLA violation")
    ax.legend(handles=[orbit_handle, feasible_handle, violation_handle],
              loc="upper left", ncol=1, frameon=False,
              borderaxespad=0.25, handlelength=1.6,
              handletextpad=0.45, labelspacing=0.35)

    ax.set_xlabel(r"Mean quality $\uparrow$")
    ax.set_ylabel(r"Max-min quality $\uparrow$")
    ax.set_xlim(0.30, 0.635)
    ax.set_ylim(0.13, 0.375)
    ax.grid(True, color=grid, linewidth=0.55)
    ax.set_axisbelow(True)
    ax.spines["left"].set_color("#3F4850")
    ax.spines["bottom"].set_color("#3F4850")
    fig.tight_layout(pad=0.6)
    out = os.path.join(os.path.dirname(__file__), "..", "figures", "tradeoff_frontier")
    savefig_pub(fig, out + ".pdf")
    fig.savefig(out + ".png", dpi=200, bbox_inches="tight")
    print("wrote", out + ".pdf/.png")

if __name__ == "__main__":
    main()
