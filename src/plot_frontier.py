#!/usr/bin/env python3
"""Quality-fairness tradeoff frontier (main comparison figure).
X = total delivered quality, Y = min-fill (max-min objective); both higher=better.
Proposed = a curve swept by the Lyapunov knob V (NO single V is marked -- the whole
curve is the claim). Baselines are fixed points. Deployable (downtime within the
SLA band) = filled; blackout policies that violate the SLA = hollow, labelled with
their downtime. The top-right (high fairness AND high throughput) is reachable only
by the blackout policies, i.e. only by being undeployable -- the empty deployable
top-right is the visual signature of the quality-fairness conflict.
Numbers: locked config (P_base=7, P_solar=13, B=18000, N=4), 10-seed means.
"""
import os, sys
import matplotlib as mpl
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import apply_house_style, savefig_pub

# measured Q_SRC values, read from data/contrast_frontier.csv (src/18)
import csv as _csv
_rows = list(_csv.DictReader(open(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "contrast_frontier.csv"))))
PROP = [(float(r["totQ_mean"]), float(r["minfill_mean"]), r["V"])
        for r in _rows if r["label"] == "Proposed"]
SLA = 5.0
BASE = [(r["label"], float(r["totQ_mean"]), float(r["minfill_mean"]),
         float(r["down_mean"]), float(r["down_mean"]) < SLA)
        for r in _rows if r["label"] != "Proposed"]

def main():
    apply_house_style()
    mpl.rcParams.update({"axes.labelsize": 10, "xtick.labelsize": 8.5,
                         "ytick.labelsize": 8.5, "legend.fontsize": 7})
    fig, ax = plt.subplots(figsize=(4.3, 3.3))

    RED, GREEN, BLUE = "#B64342", "#2E7D32", "#0F4D92"
    xs = [p[0] for p in PROP]; ys = [p[1] for p in PROP]
    ax.plot(xs, ys, "-o", color=BLUE, lw=1.6, ms=4.5, zorder=4,
            label="Proposed (swept by $V$)")
    # V=1/3/10 cluster is unreadable at full scale -> labelled inside the zoom inset;
    # only the spread-out V=100/300 keep labels on the main axes
    for x, y, v in PROP:
        if v == "100":
            ax.annotate("$V$=100", (x, y), textcoords="offset points",
                        xytext=(-8, -13), fontsize=6.3, color=BLUE)
        elif v == "300":
            ax.annotate("$V$=300", (x, y), textcoords="offset points",
                        xytext=(7, -8), fontsize=6.3, color=BLUE)
    ax.set_xlim(3050, 6750)

    for name, x, y, dt, dep in BASE:
        if dep:
            ax.scatter([x], [y], marker="s", s=42, c=GREEN, zorder=5,
                       label=name if name == "Static" else None)
            off, ha = ((8, -3), "left") if name == "Greedy-E" else ((7, 2), "left")
            ax.annotate(name, (x, y), textcoords="offset points", xytext=off,
                        fontsize=7, ha=ha)
        else:
            # undeployable: hollow circle STRUCK THROUGH with an x, so the top-right
            # position cannot read as "best" -- the marker itself says excluded
            ax.scatter([x], [y], marker="o", s=64, facecolors="none",
                       edgecolors=RED, linewidths=1.4, zorder=5)
            ax.scatter([x], [y], marker="x", s=44, c=RED, linewidths=1.6, zorder=6)
            off, ha = ((-8, -16), "right") if name == "Greedy-Q" else ((2, -22), "center")
            ax.annotate("%s: %.1f%% downtime\n(SLA violated, undeployable)" % (name, dt),
                        (x, y), textcoords="offset points", xytext=off, fontsize=6.0,
                        color=RED, ha=ha)

    ax.annotate("high fairness + high throughput\nreachable only by blacking out",
                xy=(5350, 0.128), fontsize=6.3, color=RED, ha="center", style="italic")

    # ---- zoom inset: the fairness end (V=1/3/10 + Greedy-E), where the 5-sigma
    # min-fill gap over Greedy-E lives but the points overlap at full scale
    axins = ax.inset_axes([0.075, 0.56, 0.34, 0.40])
    axins.set_facecolor("white"); axins.patch.set_alpha(1.0)
    axins.plot(xs, ys, "-o", color=BLUE, lw=1.4, ms=4.5, zorder=4)
    for (x, y, v), off in zip(PROP[:4], [(-2, 6), (5, 3), (4, -4), (-16, -9)]):
        axins.annotate("$V$=" + v, (x, y), textcoords="offset points",
                       xytext=off, fontsize=5.8, color=BLUE)
    axins.set_xlim(3520, 3740); axins.set_ylim(0.219, 0.252)
    axins.set_xticks([]); axins.set_yticks([])
    for sp in axins.spines.values():
        sp.set_visible(True); sp.set_color("0.4"); sp.set_linewidth(0.8)
    ax.indicate_inset_zoom(axins, edgecolor="0.4", lw=0.8)

    # legend above the axes (frees the upper-left for the inset); the struck-through
    # marker gets its own explicit entry
    h_circ = mpl.lines.Line2D([], [], marker="o", mfc="none", mec=RED, ls="none", ms=7, mew=1.4)
    h_x = mpl.lines.Line2D([], [], marker="x", color=RED, ls="none", ms=6, mew=1.6)
    handles, labels = ax.get_legend_handles_labels()
    handles += [(h_circ, h_x)]
    labels += ["undeployable (SLA violated)"]
    ax.legend(handles, labels, loc="lower left", bbox_to_anchor=(-0.02, 1.01),
              ncol=2, frameon=False, handletextpad=0.4, columnspacing=1.0,
              borderaxespad=0,
              handler_map={tuple: mpl.legend_handler.HandlerTuple(ndivide=1)})

    ax.set_xlabel("Total delivered quality")
    ax.set_ylabel("Min-fill (max-min)")
    ax.set_ylim(0.075, 0.44)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "..", "figures", "tradeoff_frontier")
    savefig_pub(fig, out + ".pdf")
    fig.savefig(out + ".png", dpi=200, bbox_inches="tight")
    print("wrote", out + ".pdf/.png")

if __name__ == "__main__":
    main()
