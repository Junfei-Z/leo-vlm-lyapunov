#!/usr/bin/env python3
"""Quality-energy trade-off (Pareto) figure for the EuroSAT calibration.

Standalone: consumes ONLY data/jetson_eurosat/calibration_all.csv (produced by
build_calibration.py) so the figure can be hand-tuned and re-rendered in seconds
without re-running any benchmark. House style via figstyle.

Outputs: figures/quality_energy_tradeoff.{pdf,png}
"""
import os, sys
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import apply_house_style, savefig_pub, PALETTE

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "..", "data", "jetson_eurosat", "calibration_all.csv")
OUT = os.path.join(HERE, "..", "figures", "quality_energy_tradeoff")

# short labels for plotting
SHORT = {
    "Qwen2-VL-2B-Q4": "Qwen2-VL-2B", "Qwen2.5-VL-3B-Q4": "Qwen2.5-VL-3B",
    "SmolVLM2-2.2B-Q4": "SmolVLM2-2.2B", "gemma-3-4b-Q4": "Gemma3-4B",
    "Qwen2.5-VL-7B-Q4": "Qwen2.5-VL-7B", "InternVL3-8B-Q4": "InternVL3-8B",
    "InternVL3-1B-Q8": "InternVL3-1B", "InternVL3-1B-Q8-224": "InternVL3-1B(224)",
}

def main():
    apply_house_style()
    df = pd.read_csv(CSV)
    fig, ax = plt.subplots(figsize=(5.0, 3.6))

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
        ax.scatter(sub["E_J"], sub["Q"], c=color, marker=marker, s=55,
                   zorder=3, label=lab, edgecolors="white", linewidths=0.6)
        for _, r in sub.iterrows():
            ax.annotate(SHORT.get(r["model"], r["model"]),
                        (r["E_J"], r["Q"]), textcoords="offset points",
                        xytext=(6, 4), fontsize=7)

    ax.set_xlabel("Energy per inference  $\\mathbb{E}[E_{im}]$  (J)")
    ax.set_ylabel("Accuracy  $Q_{im}$")
    ax.set_xscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=7, framealpha=0.9)
    fig.tight_layout()
    savefig_pub(fig, OUT + ".pdf")
    fig.savefig(OUT + ".png", dpi=200, bbox_inches="tight")
    print("wrote", OUT + ".pdf / .png")

if __name__ == "__main__":
    main()
