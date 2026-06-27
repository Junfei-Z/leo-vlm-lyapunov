#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EXP-7  Eclipse-fraction stability stress  (supports Theorem 3, the ISS condition).

Faithful port of src/01_single_sat_lyapunov.py. The ONLY thing varied here is the
sunlit fraction (eclipse fraction = 1 - sunlit). Every other element is identical:
the same Jetson-calibrated CONFIGS, the same solar/eclipse model, the same three
virtual queues (energy Q^E / power Q^P / quality Q^U), the same drift-plus-penalty
per-slot decision, the same battery dynamics and hard peak-power handling.

We sweep sunlit_fraction in [0.80 ... 0.40], run several orbits each, and record
the time-average Lyapunov function L = 0.5*(Q^E^2 + Q^P^2 + Q^U^2), its late-window
(lim-sup-style) mean, mean energy backlog Q^E, peak-power violations, and battery
blackout slots. Honest expectation: bounded while sunlit dominates, sharp growth as
the eclipse fraction crosses the stability boundary.
"""

import os, sys, csv
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import apply_house_style, savefig_pub, PALETTE
apply_house_style()

# =============================================================================
# Model constants -- copied verbatim from 01_single_sat_lyapunov.py
# =============================================================================
TAU = 1.0
ORBIT_PERIOD_S = 95 * 60
N_SLOTS_PER_ORBIT = int(ORBIT_PERIOD_S / TAU)          # 5700 slots

CONFIGS = {
    "2B": dict(T=1.24, E=3.93, Ppeak=4.15, Q=0.398),
    "3B": dict(T=1.74, E=5.78, Ppeak=4.51, Q=0.464),
    "7B": dict(T=2.76, E=9.13, Ppeak=5.57, Q=0.573),
}
for m in CONFIGS.values():
    m["N"] = int(np.ceil(m["T"] / TAU))
CONFIG_NAMES = list(CONFIGS.keys())

P_CAP   = 15.0
P_SOLAR = 3.0      # W: small panel, energy-balanced for the single-sat 3-source load
B_MAX   = 30000.0
B_INIT  = 0.6 * B_MAX

N_SOURCES = 3
ARRIVAL_PROB = 0.02
U_TGT = 0.30
V_DEFAULT = 50.0
RNG_SEED = 7

# EXP-7 specifics
N_ORBITS = 6                                   # >=5 orbits for steady behaviour
SUNLIT_SWEEP = [0.80, 0.70, 0.60, 0.55, 0.50, 0.45, 0.40]
LATE_WINDOW_ORBITS = 2                          # last 2 orbits = lim-sup window


def run_sim_eclipse(sunlit_fraction, V=V_DEFAULT, n_orbits=N_ORBITS, seed=RNG_SEED):
    """Single-satellite proposed scheduler -- identical model to 01, with the
    sunlit fraction parameterised."""
    rng = np.random.default_rng(seed)

    n_sunlit = int(N_SLOTS_PER_ORBIT * sunlit_fraction)
    horizon = n_orbits * N_SLOTS_PER_ORBIT

    def sunlit_indicator(t):
        phase = t % N_SLOTS_PER_ORBIT
        return 1 if phase < n_sunlit else 0

    # --- state ---
    b = B_INIT
    QE = 0.0
    QP = 0.0
    QU = np.zeros(N_SOURCES)

    busy_until = -1
    running_E_per_slot = 0.0
    running_P = 0.0

    pcap_violations = 0
    blackout_slots = 0

    L_series = np.empty(horizon)
    QE_series = np.empty(horizon)

    for t in range(horizon):
        delta = sunlit_indicator(t)

        Et_solar = P_SOLAR * TAU * delta
        omega = min(Et_solar, B_MAX - b)

        arrivals = (rng.random(N_SOURCES) < ARRIVAL_PROB).astype(int)

        e_t = 0.0
        p_t = 0.0
        q_t = np.zeros(N_SOURCES)

        if t > busy_until:
            pending = [i for i in range(N_SOURCES) if arrivals[i] == 1]
            best_score = None
            best_choice = None
            for i in pending:
                for mname in CONFIG_NAMES:
                    m = CONFIGS[mname]
                    e_per_slot = m["E"] / m["N"]
                    p_peak = m["Ppeak"]
                    if p_peak > P_CAP:
                        continue
                    if e_per_slot > b + omega:
                        continue
                    q_im = m["Q"]
                    score = (QE * e_per_slot
                             + QP * (p_peak)
                             - (QU[i] + V) * q_im)
                    if (best_score is None) or (score < best_score):
                        best_score = score
                        best_choice = (i, mname)
            if (best_choice is not None) and (best_score < 0):
                i, mname = best_choice
                m = CONFIGS[mname]
                busy_until = t + m["N"] - 1
                running_E_per_slot = m["E"] / m["N"]
                running_P = m["Ppeak"]
                q_t[i] = m["Q"]

        if t <= busy_until:
            e_t = running_E_per_slot
            p_t = running_P

        if p_t > P_CAP + 1e-9:
            pcap_violations += 1

        # battery update; detect blackout (battery hits empty floor while a task
        # is drawing energy this slot)
        b_new = min(max(b + omega - e_t, 0.0), B_MAX)
        if b_new <= 0.0 + 1e-9 and e_t > 0.0:
            blackout_slots += 1
        b = b_new

        QE = max(QE + e_t - omega, 0.0)
        QP = max(QP + p_t - P_CAP, 0.0)
        QU = np.maximum(QU + arrivals * U_TGT - q_t, 0.0)

        # Lyapunov function: aggregate quality queue as sum over sources (consistent
        # with the multi-source state vector). Use sum to capture total backlog.
        QU_agg = QU.sum()
        L_series[t] = 0.5 * (QE * QE + QP * QP + QU_agg * QU_agg)
        QE_series[t] = QE

    late_start = (n_orbits - LATE_WINDOW_ORBITS) * N_SLOTS_PER_ORBIT
    return dict(
        mean_L=float(L_series.mean()),
        late_L=float(L_series[late_start:].mean()),
        mean_QE=float(QE_series.mean()),
        late_QE=float(QE_series[late_start:].mean()),
        peak_violations=int(pcap_violations),
        blackout_slots=int(blackout_slots),
    )


def main():
    print("=" * 70)
    print("  EXP-7  Eclipse-fraction stability stress")
    print("=" * 70)
    rows = []
    for sf in SUNLIT_SWEEP:
        ef = 1.0 - sf
        r = run_sim_eclipse(sf)
        rows.append((sf, ef, r))
        print(f"  sunlit={sf:.2f} eclipse={ef:.2f} | "
              f"mean_L={r['mean_L']:.3e} late_L={r['late_L']:.3e} "
              f"mean_QE={r['mean_QE']:.2f} pcap_viol={r['peak_violations']} "
              f"blackout={r['blackout_slots']}")

    # --- CSV ---
    os.makedirs("data", exist_ok=True)
    csv_path = "data/sens_eclipse_stress.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sunlit_fraction", "eclipse_fraction", "mean_L", "late_L",
                    "mean_QE", "peak_violations", "blackout_slots"])
        for sf, ef, r in rows:
            w.writerow([f"{sf:.4f}", f"{ef:.4f}", f"{r['mean_L']:.6e}",
                        f"{r['late_L']:.6e}", f"{r['mean_QE']:.6f}",
                        r["peak_violations"], r["blackout_slots"]])
    print(f"  saved {csv_path}")

    # --- plot ---
    ef_arr   = np.array([1.0 - sf for sf, _, _ in rows])
    meanL    = np.array([r["mean_L"] for _, _, r in rows])
    lateL    = np.array([r["late_L"] for _, _, r in rows])
    meanQE   = np.array([r["mean_QE"] for _, _, r in rows])

    fig, ax1 = plt.subplots(figsize=(8, 5.2))
    ax1.plot(ef_arr, meanL, "o-", color=PALETTE["blue_main"], lw=2.2,
             ms=7, label=r"time-average $\overline{L}$")
    ax1.plot(ef_arr, lateL, "s--", color=PALETTE["blue_secondary"], lw=2.2,
             ms=7, label=r"late-window $L$ (lim-sup)")
    ax1.set_xlabel("Eclipse fraction  $1-\\delta$")
    ax1.set_ylabel(r"Lyapunov function $L=\frac{1}{2}(Q_E^2+Q_P^2+Q_U^2)$",
                   color=PALETTE["blue_main"])
    ax1.tick_params(axis="y", labelcolor=PALETTE["blue_main"])

    ax2 = ax1.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.plot(ef_arr, meanQE, "^-", color=PALETTE["red_strong"], lw=2.2,
             ms=7, label=r"mean $Q^E$ backlog")
    ax2.set_ylabel(r"mean energy backlog $\overline{Q^E}$",
                   color=PALETTE["red_strong"])
    ax2.tick_params(axis="y", labelcolor=PALETTE["red_strong"])

    l1, lab1 = ax1.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2, loc="upper left", fontsize=12)
    ax1.set_title("Eclipse-fraction stress: backlog grows past the stability boundary")
    ax1.grid(alpha=0.3)

    savefig_pub(fig, "eclipse_stress.pdf")
    fig.savefig("eclipse_stress.png", dpi=200, bbox_inches="tight")
    print("  saved eclipse_stress.pdf / eclipse_stress.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
