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
from figstyle import apply_house_style, savefig_pub, PALETTE

# This Pareto scatter is a COMPACT single-column figure, not a big line plot, so we
# override the house-style large fonts with publication sizes proportioned to the
# figure — otherwise the 24pt labels dwarf the panel once it is shrunk to column width.
def _compact_style():
    mpl.rcParams.update({
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 7.5,
        "lines.linewidth": 1.4,
    })

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
    "Qwen2-VL-2B-Q4": (5, -12), "Qwen2.5-VL-3B-Q4": (6, -3),
    "Qwen2.5-VL-7B-Q4": (-6, -13), "InternVL2.5-4B-Q4": (2, 7),
    "InternVL3-8B-Q4": (-2, 8), "gemma-3-4b-Q4": (-4, 7), "SmolVLM2-2.2B-Q4": (7, -2),
}
LHA = {"Qwen2.5-VL-7B-Q4": "right", "InternVL3-8B-Q4": "center", "gemma-3-4b-Q4": "right"}

def main():
    apply_house_style()
    _compact_style()
    df = pd.read_csv(CSV)
    fig, ax = plt.subplots(figsize=(3.5, 2.7))

    # Pareto frontier line (sorted by energy)
    pf = df[df["pareto"]].sort_values("E_J")
    ax.plot(pf["E_J"], pf["Q"], "-", color=PALETTE["blue_main"], lw=1.6, zorder=1,
            label="Pareto frontier")

    # three classes of points
    groups = [
        ("pareto",   lambda d: d["pareto"],                       PALETTE["blue_main"],   "o", "Pareto-optimal"),
        ("dominated",lambda d: (~d["pareto"]) & (~d["failed"]),   PALETTE["red_strong"],  "s", "dominated"),
        ("failed",   lambda d: d["failed"],                       PALETTE["neutral"],     "x", "failed (chance)"),
    ]
    for _, mask, color, marker, lab in groups:
        sub = df[mask(df)]
        if sub.empty:            # e.g. no "failed (chance)" model in the 1300-image set
            continue
        ax.scatter(sub["E_J"], sub["Q"], c=color, marker=marker, s=55,
                   zorder=3, label=lab, edgecolors="white", linewidths=0.6)
        for _, r in sub.iterrows():
            ax.annotate(SHORT.get(r["model"], r["model"]),
                        (r["E_J"], r["Q"]), textcoords="offset points",
                        xytext=LOFF.get(r["model"], (6, 4)), fontsize=6.5,
                        ha=LHA.get(r["model"], "left"))

    ax.set_xlabel("Energy per inference  $\\mathbb{E}[E_{im}]$  (J)")
    ax.set_ylabel("Accuracy  $Q_{im}$")
    ax.set_xscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_ylim(0.27, 0.60)
    ax.legend(loc="lower right", fontsize=6.5, framealpha=0.9)
    fig.tight_layout()
    savefig_pub(fig, OUT + ".pdf")
    fig.savefig(OUT + ".png", dpi=200, bbox_inches="tight")
    print("wrote", OUT + ".pdf / .png")

if __name__ == "__main__":
    main()
