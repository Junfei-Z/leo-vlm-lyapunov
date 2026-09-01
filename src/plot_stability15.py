#!/usr/bin/env python3
"""Regenerate figures/tleP_stability.pdf (paper Fig. 5) from the U=0.15
20-orbit run (data/tleP15_stability.npz).

Three curves in three distinct colors (battery / energy queue / quality
queue) with a single legend centered in the gap between the two subplots;
no inline annotations.
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import apply_paper_style, savefig_pub  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
FIGS = os.path.join(HERE, "..", "figures")

apply_paper_style()
BATT = "#333333"      # battery
QE = "#0F4D92"        # energy virtual queue (blue)
QU = "#B64342"        # quality virtual queue (red)

d = np.load(os.path.join(DATA, "tleP15_stability.npz"))
qe, qu, b0 = d["qe"], d["qu"], d["b0"]
t_h = np.arange(len(qe)) / 3600.0

# eclipse shading from the TLE trace (satellite 0), tiled over the 20-orbit run
_tr = np.load(os.path.join(DATA, "tle_traces.npz"))
_sun = _tr["sunlit"][:, 0].astype(bool)
_ecl = ~_sun[np.arange(len(qe)) % len(_sun)]
_edges = np.flatnonzero(np.diff(_ecl.astype(int)))
_starts = list(_edges[np.diff(_ecl.astype(int))[_edges] == 1] + 1)
_ends = list(_edges[np.diff(_ecl.astype(int))[_edges] == -1] + 1)
if _ecl[0]:
    _starts = [0] + _starts
if _ecl[-1]:
    _ends = _ends + [len(_ecl)]

fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.5, 2.9), sharex=True)
fig.subplots_adjust(hspace=0.42)          # gap for the shared legend
for a in (a1, a2):
    for x0, x1 in zip(_starts, _ends):
        a.axvspan(x0 / 3600.0, x1 / 3600.0, color="0.55", alpha=0.16, lw=0)

a1.plot(t_h, b0 / 1000.0, color=BATT, lw=1.0, label="battery")
a1.set_ylabel("Battery [kJ]")

a2.plot(t_h, qe, color=QE, lw=1.2, label=r"energy queue $\bar Q^E$")
a2.plot(t_h, qu, color=QU, lw=1.2, label=r"quality queue $\bar Q^U$")
a2.set_ylabel("Virtual queues")
a2.set_xlabel("Time [h] (20 orbits)")

for a in (a1, a2):
    a.grid(True, alpha=0.3)

# single legend centered between the two subplots
handles, labels = [], []
for a in (a1, a2):
    h, l = a.get_legend_handles_labels()
    handles += h
    labels += l
fig.legend(handles, labels, loc="center", bbox_to_anchor=(0.5, 0.53),
           ncol=3, frameon=False, fontsize=7.5, columnspacing=1.2)

out = os.path.join(FIGS, "tleP_stability")
savefig_pub(fig, out + ".pdf")
fig.savefig(out + ".png", dpi=170, bbox_inches="tight")
print("wrote", out)
