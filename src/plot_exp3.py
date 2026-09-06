#!/usr/bin/env python3
"""Regenerate figures/exp3_cycle.pdf from data/exp3_cycle.csv + exp3_raw.npz.

(a) nominal envelope: seed-averaged orbit-boundary L trajectory + fitted bound
(b) phase diagram at the reserve slice rho=0.8: battery survival (down=0) vs
    queue boundedness (kappa < 1).
"""
import os, sys, csv
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import apply_house_style, savefig_pub, PALETTE  # noqa: E402

import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "..", "data", "exp3_cycle.csv")
RAW = os.path.join(HERE, "..", "data", "exp3_raw.npz")
OUT = os.path.join(HERE, "..", "figures", "exp3_cycle.pdf")

rows = list(csv.DictReader(open(CSV)))
nom = [r for r in rows if (float(r["panel"]), float(r["battery"]), float(r["reserve"])) == (12.2, 16.6, 0.3)][0]
raw = np.load(RAW)
Ls = raw["L_12.2_16.6_0.3"]
Lbar = Ls.mean(0)

apply_house_style()
fig, ax = plt.subplots(1, 2, figsize=(15, 6.2), gridspec_kw=dict(width_ratios=[1, 1.15]))
# (a) nominal envelope
ax[0].plot(Lbar[:-1], Lbar[1:], "o-", ms=5, color=PALETTE["blue_main"], lw=1.8,
           label=r"seed-averaged $E[L(t_k)]$ trajectory")
xs = np.linspace(0, Lbar.max() * 1.05, 60)
ax[0].plot(xs, float(nom["kappa"]) * xs + float(nom["D"]), "-", color=PALETTE["red_strong"], lw=2.5,
           label=fr"fitted envelope $\kappa={float(nom['kappa']):.2f},\ D={float(nom['D']):.0f}$")
ax[0].plot(xs, xs, "--", color=PALETTE["neutral"], lw=1.8, label="$L_{k+1}=L_k$")
ax[0].set_xlabel(r"$\mathbb{E}[L(Q(t_k))]$"); ax[0].set_ylabel(r"$\mathbb{E}[L(Q(t_{k+1}))]$")
ax[0].grid(alpha=0.3); ax[0].legend(loc="upper left", fontsize=13)
ax[0].set_title("(a) nominal cycle envelope (expectation)", loc="left")
# (b) phase diagram at rho=0.8: battery survival vs queue boundedness
sub = [r for r in rows if float(r["reserve"]) == 0.3]
P = np.array([float(r["panel"]) for r in sub]); B = np.array([float(r["battery"]) for r in sub])
batt = np.array([r["battery_ok"] == "True" for r in sub])
bd = np.array([float(r["kappa"]) < 0.999 for r in sub])
both = batt & bd
bonly = batt & ~bd
ronly = ~batt & bd
none = ~batt & ~bd
ax[1].scatter(P[both], B[both], s=220, marker="o", color=PALETTE["green_3"], edgecolor="k",
              label="battery + bounded queues")
ax[1].scatter(P[bonly], B[bonly], s=220, marker="^", color=PALETTE["highlight"], edgecolor="k",
              label="battery only")
ax[1].scatter(P[ronly], B[ronly], s=220, marker="P", color=PALETTE["blue_secondary"], edgecolor="k",
              label="bounded only")
ax[1].scatter(P[none], B[none], s=220, marker="x", color=PALETTE["red_strong"], label="neither")
ax[1].axvline(12.2, color=PALETTE["neutral"], ls=":", lw=1.8)
ax[1].axhline(16.6, color=PALETTE["neutral"], ls=":", lw=1.8)
ax[1].set_xlabel(r"panel power $P^{\mathrm{solar}}$ (W)")
ax[1].set_ylabel(r"battery $B^{\max}$ (kJ)")
ax[1].grid(alpha=0.3); ax[1].legend(loc="upper left", fontsize=12)
ax[1].set_title(r"(b) phase diagram at $\rho=0.3$", loc="left")
savefig_pub(fig, OUT)
print(f"wrote {OUT}   (rho=0.3: both={both.sum()} bounded={bd.sum()} battery={batt.sum()} of {len(sub)})")
