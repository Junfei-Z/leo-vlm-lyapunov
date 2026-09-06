#!/usr/bin/env python3
"""Two locked-config figures:
(1) ISL availability (from data/scope_isl.csv): splits fall as p_ISL falls; the
    loss lands on min-fill (the 7B-dependent source), total quality unchanged.
(2) Stability: 20-orbit battery + virtual-queue traces from the locked src/04
    model (proposed, V=10) — the empirical signature of Theorems (bounded,
    mean-rate stable queues through periodic eclipse).
"""
import os, sys, csv
import importlib.util
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import apply_house_style, savefig_pub

HERE = os.path.dirname(os.path.abspath(__file__))
apply_house_style()
mpl.rcParams.update({"axes.labelsize": 9, "axes.titlesize": 9.5,
                     "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "legend.fontsize": 8, "lines.linewidth": 1.6,
                     "lines.markersize": 4.5})

# ---------- (1) ISL ----------
rows = list(csv.DictReader(open(os.path.join(HERE, "..", "data", "scope_isl.csv"))))
p   = np.array([float(r["p_isl"]) for r in rows])
sp  = np.array([float(r["split_mean"]) for r in rows]); sps = np.array([float(r["split_std"]) for r in rows])
mf  = np.array([float(r["minfill_mean"]) for r in rows]); mfs = np.array([float(r["minfill_std"]) for r in rows])
tq  = np.array([float(r["totQ_mean"]) for r in rows])

fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
ax[0].errorbar(p, sp, yerr=sps, fmt="-o", color="#0F4D92", capsize=2)
ax[0].set_xlabel("ISL availability $p_{\\mathrm{ISL}}$"); ax[0].set_ylabel("7B split inferences")
ax[0].set_title("(a)", loc="left")
ax[1].errorbar(p, mf, yerr=mfs, fmt="-o", color="#B64342", capsize=2, label="min-fill")
ax[1].set_xlabel("ISL availability $p_{\\mathrm{ISL}}$"); ax[1].set_ylabel("min-fill (max-min)", color="#B64342")
ax[1].tick_params(axis="y", labelcolor="#B64342")
ax2 = ax[1].twinx()
ax2.plot(p, tq, "-s", color="#555555", label="total quality")
ax2.set_ylabel("total quality", color="#555555"); ax2.tick_params(axis="y", labelcolor="#555555")
ax2.set_ylim(3200, 3420)
ax[1].set_title("(b)", loc="left")
for a in ax: a.grid(True, alpha=0.3)
fig.tight_layout()
out = os.path.join(HERE, "..", "figures", "isl_locked")
savefig_pub(fig, out + ".pdf"); fig.savefig(out + ".png", dpi=170, bbox_inches="tight")
print("wrote", out)

# ---------- (2) Stability: 20-orbit run on the locked model ----------
s = importlib.util.spec_from_file_location("m", os.path.join(HERE, "04_split_pipeline.py"))
M = importlib.util.module_from_spec(s); sys.modules["m"] = M; s.loader.exec_module(M)
H20 = M.N_SLOTS_PER_ORBIT * 20
r = M.run_sim(M.POLICIES["Lyapunov (ours)"], V=10, horizon=H20, seed=7)
t_min = np.arange(H20) / 60.0
b0 = r["log"]["battery"][:, 0] / 1000.0
qe = np.array(r["log"]["QE_mean"]); qu = np.array(r["log"]["QU_mean"])

mpl.rcParams.update({"font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
                     "xtick.labelsize": 8, "ytick.labelsize": 8})
fig, ax = plt.subplots(3, 1, figsize=(7.0, 5.4), sharex=True)
ax[0].plot(t_min, b0, color="#0F4D92", lw=0.8)
ax[0].set_ylabel("battery (kJ)"); ax[0].set_title("(a) satellite-0 battery", loc="left")
ax[1].plot(t_min, qe, color="#B64342", lw=0.8)
ax[1].set_ylabel("$\\bar{Q}^E(t)$"); ax[1].set_title("(b) energy virtual queue (mean)", loc="left")
ax[2].plot(t_min, qu, color="#2E7D32", lw=0.8)
ax[2].set_ylabel("$\\bar{Q}^U(t)$"); ax[2].set_title("(c) quality-deficit queue (mean)", loc="left")
ax[2].set_xlabel("time (min), 20 orbits")
for a in ax: a.grid(True, alpha=0.3)
fig.tight_layout(pad=0.6)
out = os.path.join(HERE, "..", "figures", "stability_20orbit")
fig.savefig(out + ".pdf", dpi=300, bbox_inches="tight")
fig.savefig(out + ".png", dpi=170, bbox_inches="tight")
print("wrote", out)
print("QE(T)/T=%.2e QU(T)/T=%.2e QEpeak=%.1f QUpeak=%.1f blackout=%.2f%%" % (
    qe[-1]/H20, qu[-1]/H20, qe.max(), qu.max(), r["blackout_slots"]/(M.N_SAT*H20)*100))
