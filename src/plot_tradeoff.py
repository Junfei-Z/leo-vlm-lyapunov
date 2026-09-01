#!/usr/bin/env python3
"""Quality-energy trade-off (Pareto) figure for the EuroSAT calibration.

Standalone: consumes ONLY data/jetson_eurosat/calibration_all.csv (produced by
build_calibration.py) so the figure can be hand-tuned and re-rendered in seconds
without re-running any benchmark. House style via figstyle.

Outputs: figures/quality_energy_tradeoff.{pdf,png}
"""
import os, sys
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import apply_house_style, apply_paper_style, savefig_pub, PALETTE

# This Pareto scatter is a COMPACT single-column figure, not a big line plot, so we
# override the house-style large fonts with publication sizes proportioned to the
# figure — otherwise the 24pt labels dwarf the panel once it is shrunk to column width.
def _compact_style():
    apply_paper_style()

HERE = os.path.dirname(os.path.abspath(__file__))
# Dataset dir name via arg 1 (default RESISC45 if present, else EuroSAT). Output name carries it.
_dsname = sys.argv[1] if len(sys.argv) > 1 else (
    "jetson_resisc45" if os.path.isdir(os.path.join(HERE, "..", "data", "jetson_resisc45")) else "jetson_eurosat")
CSV = os.path.join(HERE, "..", "data", _dsname, "calibration_all.csv")
_suffix = "_RESISC45" if "resisc" in _dsname else "_EuroSAT"
OUT = os.path.join(HERE, "..", "figures", "quality_energy_tradeoff" + _suffix)

# short labels for plotting
SHORT = {
    "Qwen2-VL-2B-Q4": "Qwen2-VL-2B", "Qwen2.5-VL-3B-Q4": "Qwen2.5-VL-3B",
    "SmolVLM2-2.2B-Q4": "SmolVLM2-2.2B", "gemma-3-4b-Q4": "Gemma3-4B",
    "Qwen2.5-VL-7B-Q4": "Qwen2.5-VL-7B", "InternVL3-8B-Q4": "InternVL3-8B",
    "InternVL2.5-4B-Q4": "InternVL2.5-4B",
    "InternVL3-1B-Q8": "InternVL3-1B", "InternVL3-1B-Q8-224": "InternVL3-1B(224)",
}

# per-model label offsets (pts) + alignment to de-clutter the ~0.55-accuracy cluster
LOFF = {
    "Qwen2-VL-2B-Q4": (5, -19), "Qwen2.5-VL-3B-Q4": (7, -17),
    "Qwen2.5-VL-7B-Q4": (-8, 8), "InternVL2.5-4B-Q4": (5, -12),
    "InternVL3-8B-Q4": (-2, 8), "gemma-3-4b-Q4": (-2, 9), "SmolVLM2-2.2B-Q4": (7, -2),
}
LHA = {"Qwen2.5-VL-7B-Q4": "right", "InternVL3-8B-Q4": "center", "gemma-3-4b-Q4": "right"}

def main():
    apply_house_style()
    _compact_style()
    df = pd.read_csv(CSV)
    # 8B is not deployable on the Jetson-class satellite and is excluded
    df = df[df["model"] != "InternVL3-8B-Q4"]
    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    # Pareto frontier as a staircase; shade the region it dominates
    pf = df[df["pareto"]].sort_values("E_J")
    xmax = float(df["E_J"].max()) * 1.6
    xs = list(pf["E_J"]) + [xmax]
    ys = list(pf["Q"]) + [float(pf["Q"].iloc[-1])]
    ax.plot(xs, ys, drawstyle="steps-post", color=PALETTE["blue_main"], lw=1.6,
            zorder=1, label="Pareto frontier")
    ax.fill_between(xs, ys, 0.0, step="post", color=PALETTE["blue_main"],
                    alpha=0.06, zorder=0)

    # frontier = filled dots; dominated = HOLLOW squares (so a dominated point that
    # nearly coincides with a frontier point, e.g. InternVL2.5-4B vs Qwen-7B, still
    # reads as two distinct entities)
    groups = [
        ("pareto",   lambda d: d["pareto"],                     dict(c=PALETTE["blue_main"], marker="o", s=48, edgecolors="white", linewidths=0.6), "Pareto-optimal"),
        ("dominated",lambda d: (~d["pareto"]) & (~d["failed"]), dict(facecolors="none", marker="s", s=52, edgecolors=PALETTE["red_strong"], linewidths=1.3), "dominated"),
    ]
    for _, mask, style, lab in groups:
        sub = df[mask(df)]
        if sub.empty:
            continue
        ax.scatter(sub["E_J"], sub["Q"], zorder=3, label=lab, **style)
        for _, r in sub.iterrows():
            name = SHORT.get(r["model"], r["model"])
            if r["pareto"]:
                name += "\n%.2f s, %.2f W peak" % (r["T_s"], r["Ppeak_W"])
            ax.annotate(name,
                        (r["E_J"], r["Q"]), textcoords="offset points",
                        xytext=LOFF.get(r["model"], (6, 4)), fontsize=6.5,
                        ha=LHA.get(r["model"], "left"))

    ax.set_xlabel(r"Energy per inference  $\mathbb{E}[E_{im}]$  (J)  $\downarrow$")
    ax.set_ylabel(r"Accuracy  $Q_{im}$  $\uparrow$")
    ax.set_xscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_ylim(0.27, 0.62)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.01), ncol=3, frameon=False,
              fontsize=6.3, handletextpad=0.3, columnspacing=0.8, borderaxespad=0)

    fig.tight_layout()
    savefig_pub(fig, OUT + ".pdf")
    fig.savefig(OUT + ".png", dpi=200, bbox_inches="tight")
    print("wrote", OUT + ".pdf / .png")

if __name__ == "__main__":
    main()
