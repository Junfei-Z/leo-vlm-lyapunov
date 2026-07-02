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

# Proposed swept by V (V=1..300): (total_quality, min_fill)
PROP = [(3254, 0.175, "1"), (3273, 0.173, "3"), (3311, 0.169, "10"),
        (3394, 0.158, "30"), (3636, 0.124, "100"), (3850, 0.118, "300")]
# Baselines: name, totQ, min_fill, downtime%, deployable
BASE = [
    ("Greedy-E", 3193, 0.156, 0.8, True),
    ("Static",   5102, 0.119, 0.8, True),
    ("Greedy-Q", 6083, 0.398, 7.0, False),
    ("Random",   5022, 0.354, 5.3, False),
]

def main():
    apply_house_style()
    mpl.rcParams.update({"axes.labelsize": 10, "xtick.labelsize": 8.5,
                         "ytick.labelsize": 8.5, "legend.fontsize": 7.5})
    fig, ax = plt.subplots(figsize=(4.3, 3.1))

    xs = [p[0] for p in PROP]; ys = [p[1] for p in PROP]
    ax.plot(xs, ys, "-o", color="#0F4D92", lw=1.6, ms=4.5, zorder=4,
            label="Proposed (swept by $V$)")
    # label only V=1,10,100,300; uniform, none emphasised; staggered to avoid overlap
    VOFF = {"1": (-26, -3), "10": (3, 8), "100": (-8, -13), "300": (7, -8)}
    ax.set_xlim(2930, 6400)   # left pad so the V=1 label is not clipped
    for x, y, v in PROP:
        if v in VOFF:
            ax.annotate("$V$=" + v, (x, y), textcoords="offset points",
                        xytext=VOFF[v], fontsize=6.3, color="#0F4D92")

    for name, x, y, dt, dep in BASE:
        if dep:
            ax.scatter([x], [y], marker="s", s=42, c="#2E7D32", zorder=5, label=name)
            # Greedy-E sits just below/left of Proposed(V=1); label below the square,
            # clear of the descending proposed curve
            if name == "Greedy-E":
                ax.annotate(name, (x, y), textcoords="offset points", xytext=(0, -15),
                            fontsize=7, ha="center")
            else:
                ax.annotate(name, (x, y), textcoords="offset points", xytext=(7, 2), fontsize=7)
        else:                          # blackout -> hollow, SLA-violation labelled
            ax.scatter([x], [y], marker="o", s=48, facecolors="none",
                       edgecolors="#B64342", linewidths=1.4, zorder=5)
            # Greedy-Q label extends left-above; Random label goes below-right so it
            # cannot collide with the legend in the upper-left
            off, ha = ((-4, 7), "right") if name == "Greedy-Q" else ((7, -11), "left")
            ax.annotate("%s (%.1f%% down, $\\times$)" % (name, dt), (x, y),
                        textcoords="offset points", xytext=off, fontsize=6.3,
                        color="#B64342", ha=ha)

    # the deployable top region is empty -> the conflict (place in the clear mid area)
    ax.annotate("high fairness + high throughput\nreachable only by blacking out",
                xy=(4700, 0.255), fontsize=6.3, color="#B64342", ha="center", style="italic")

    ax.set_xlabel("Total delivered quality")
    ax.set_ylabel("Min-fill (max-min)")
    ax.set_ylim(0.075, 0.44)
    ax.legend(loc="upper left", frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "..", "figures", "tradeoff_frontier")
    savefig_pub(fig, out + ".pdf")
    fig.savefig(out + ".png", dpi=200, bbox_inches="tight")
    print("wrote", out + ".pdf/.png")

if __name__ == "__main__":
    main()
