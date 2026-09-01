#!/usr/bin/env python3
"""Regenerate figures/exp1_utgt.pdf from data/exp1_utgt.csv (mainline U=0.15)."""
import os, sys, csv
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import apply_house_style, savefig_pub, PALETTE  # noqa: E402

import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "..", "data", "exp1_utgt.csv")
OUT = os.path.join(HERE, "..", "figures", "exp1_utgt.pdf")
FRAC = 0.9
MAINLINE = 0.15

rows = [r for r in csv.DictReader(open(CSV)) if float(r["u"]) >= 0]
uu = np.array([float(r["u"]) for r in rows])
mf = np.array([float(r["min_fill"]) for r in rows])
mf_sd = np.array([float(r["min_fill_sd"]) for r in rows])
qu = np.array([float(r["qu_end"]) for r in rows])
fr = np.array([float(r["feasible_frac"]) for r in rows])
bd = float([r for r in csv.DictReader(open(CSV)) if r.get("note") == "bisection_boundary"][0]["min_fill"])

apply_house_style()
fig, ax = plt.subplots(figsize=(8.6, 6.2))
feas = fr >= FRAC
ax.fill_between(uu, 0, 0.5, where=feas, color=PALETTE["green_3"], alpha=0.35,
                label="empirically feasible")
ax.plot(uu, mf, "o-", color=PALETTE["blue_main"], lw=2.5, ms=7, label="max-min quality")
ax.fill_between(uu, mf - mf_sd, mf + mf_sd, color=PALETTE["blue_main"], alpha=0.15)
ax.axvline(MAINLINE, color=PALETTE["neutral"], ls="--", lw=2,
           label=fr"mainline $U^{{\mathrm{{tgt}}}}={MAINLINE}$")
ax.axvline(bd, color=PALETTE["red_strong"], ls=":", lw=2.2,
           label=f"empirical boundary ({bd:.2f})")
ax.axhline(0.0, color="k", lw=0.8)
ax.set_xlabel(r"quality target $U^{\mathrm{tgt}}$")
ax.set_ylabel("max-min quality")
ax.grid(alpha=0.3)
ax.legend(loc="upper left", fontsize=14)
ax2 = ax.twinx()
ax2.plot(uu, qu, "s--", color=PALETTE["highlight"], ms=6, lw=2,
         label=r"max $Q_i^U(T)/H$")
ax2.set_ylabel(r"normalized quality-queue endpoint $\max_i Q_i^U(T)/H$")
ax2.grid(False)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="center right", fontsize=13)
savefig_pub(fig, OUT)
print(f"wrote {OUT}  (boundary {bd:.3f})")
