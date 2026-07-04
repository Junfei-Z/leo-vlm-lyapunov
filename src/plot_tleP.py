#!/usr/bin/env python3
"""Standalone plotters for the two TLE-mainline figures (measured Q_SRC):
  tleP_isl.pdf       from data/tleP_isl.csv       (splits + min-fill vs p_ISL)
  tleP_stability.pdf from data/tleP_stability.npz (20-orbit battery + queues)
Previously generated ad hoc; this script closes the figure-reproducibility
gap (every figure = data CSV/npz + standalone script)."""
import os, sys, csv
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import apply_house_style, savefig_pub

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
FIGS = os.path.join(HERE, "..", "figures")
apply_house_style()
mpl.rcParams.update({"axes.labelsize": 10, "axes.titlesize": 10,
                     "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
                     "legend.fontsize": 8, "lines.linewidth": 1.6,
                     "lines.markersize": 4.5})
BLUE, RED = "#0F4D92", "#B64342"

# ---------------- tleP_isl ----------------
rows = list(csv.DictReader(open(os.path.join(DATA, "tleP_isl.csv"))))
p = [float(r["p_isl"]) for r in rows]
sp = [float(r["split_mean"]) for r in rows]
mf = [float(r["minfill_mean"]) for r in rows]
mfs = [float(r["minfill_std"]) for r in rows]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(4.6, 2.2))
a1.plot(p, sp, "-o", color=BLUE)
a1.set_xlabel(r"ISL availability $p_{\mathrm{ISL}}$")
a1.set_ylabel("7B splits")
a1.set_title("(a)", fontsize=9)
a2.errorbar(p, mf, yerr=mfs, fmt="-s", color=RED, capsize=2)
a2.set_xlabel(r"ISL availability $p_{\mathrm{ISL}}$")
a2.set_ylabel("Min-fill")
a2.set_title("(b)", fontsize=9)
for a in (a1, a2):
    a.grid(True, alpha=0.3)
fig.tight_layout(pad=0.5)
out = os.path.join(FIGS, "tleP_isl")
savefig_pub(fig, out + ".pdf")
fig.savefig(out + ".png", dpi=170, bbox_inches="tight")
print("wrote", out)

# ---------------- tleP_stability ----------------
d = np.load(os.path.join(DATA, "tleP_stability.npz"))
qe, qu, b0 = d["qe"], d["qu"], d["b0"]
t_h = np.arange(len(qe)) / 3600.0

fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.6, 2.7), sharex=True)
a1.plot(t_h, b0 / 1000.0, color=BLUE, lw=1.0)
a1.set_ylabel("Batt. [kJ]")
a2.plot(t_h, qe, color=BLUE, lw=1.2)
a2.plot(t_h, qu, color=RED, lw=1.2)
a2.set_ylabel("Virtual queues")
a2.set_xlabel("Time [h] (20 orbits)")
# inline labels next to the curves (the compact panel leaves no clear legend box)
a2.annotate(r"$\bar Q^U$", xy=(20, qu[int(20*3600)]), xytext=(17.5, 660),
            fontsize=9, color=RED)
a2.annotate(r"$\bar Q^E$", xy=(26, 320), xytext=(25.2, 380),
            fontsize=9, color=BLUE)
for a in (a1, a2):
    a.grid(True, alpha=0.3)
fig.tight_layout(pad=0.5)
out = os.path.join(FIGS, "tleP_stability")
savefig_pub(fig, out + ".pdf")
fig.savefig(out + ".png", dpi=170, bbox_inches="tight")
print("wrote", out)
