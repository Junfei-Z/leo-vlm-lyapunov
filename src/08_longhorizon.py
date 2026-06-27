#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EXP-8  Long-horizon mean-rate stability  (supports Theorem 1).

Faithful port of src/01_single_sat_lyapunov.py at the DEFAULT eclipse fraction
(SUNLIT_FRACTION = 0.60). We run the proposed scheduler over a long horizon
(20 orbits) and track, for each virtual queue, the running quantity Q(T)/T, i.e.
the instantaneous backlog divided by the elapsed time. Theorem 1 (mean-rate
stability) predicts E[Q(T)]/T -> 0. We plot the three curves vs T (slots) and
report the final values at the largest T.

The model -- CONFIGS, solar/eclipse, three queues (energy/power/quality),
drift-plus-penalty decision, battery and hard peak-power handling -- is identical
to 01; only the horizon length is extended.
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
SUNLIT_FRACTION = 0.60                                  # default eclipse fraction
N_SLOTS_PER_ORBIT = int(ORBIT_PERIOD_S / TAU)          # 5700 slots
N_SUNLIT = int(N_SLOTS_PER_ORBIT * SUNLIT_FRACTION)

CONFIGS = {
    "2B": dict(T=1.24, E=3.93, Ppeak=4.15, Q=0.398),
    "3B": dict(T=1.74, E=5.78, Ppeak=4.51, Q=0.464),
    "7B": dict(T=2.76, E=9.13, Ppeak=5.57, Q=0.573),
}
for m in CONFIGS.values():
    m["N"] = int(np.ceil(m["T"] / TAU))
CONFIG_NAMES = list(CONFIGS.keys())

P_CAP   = 15.0
P_BASE  = 9.0      # platform baseline draw (sensing+comms+ADCS); shares the bus
P_SOLAR = 3.0      # W: small panel, energy-balanced for the single-sat 3-source load
B_MAX   = 30000.0
B_INIT  = 0.6 * B_MAX

N_SOURCES = 3
ARRIVAL_PROB = 0.02
U_TGT = 0.30
V_DEFAULT = 50.0
RNG_SEED = 7

# EXP-8 specifics
N_ORBITS = 20
HORIZON = N_ORBITS * N_SLOTS_PER_ORBIT
SAMPLE_EVERY = 500                                      # sample interval [slots]


def sunlit_indicator(t):
    phase = t % N_SLOTS_PER_ORBIT
    return 1 if phase < N_SUNLIT else 0


def run_longhorizon(V=V_DEFAULT, horizon=HORIZON, seed=RNG_SEED):
    rng = np.random.default_rng(seed)

    b = B_INIT
    QE = 0.0
    QP = 0.0
    QU = np.zeros(N_SOURCES)

    busy_until = -1
    running_E_per_slot = 0.0
    running_P = 0.0
    pcap_violations = 0

    samples = []   # (t, QE_over_t, QP_over_t, QU_over_t)

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
                    if P_BASE + p_peak > P_CAP:
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

        if P_BASE + p_t > P_CAP + 1e-9:
            pcap_violations += 1

        b = min(max(b + omega - e_t, 0.0), B_MAX)

        QE = max(QE + e_t - omega, 0.0)
        QP = max(QP + (P_BASE + p_t) - P_CAP, 0.0)
        QU = np.maximum(QU + arrivals * U_TGT - q_t, 0.0)

        # sample Q(T)/T (use instantaneous backlog over elapsed time T = t+1)
        if ((t + 1) % SAMPLE_EVERY == 0) or (t == horizon - 1):
            Tnow = t + 1
            samples.append((Tnow,
                            QE / Tnow,
                            QP / Tnow,
                            QU.sum() / Tnow))

    return samples, pcap_violations


def main():
    print("=" * 70)
    print("  EXP-8  Long-horizon mean-rate stability")
    print("=" * 70)
    print(f"  horizon = {HORIZON} slots ({N_ORBITS} orbits), "
          f"sunlit_fraction = {SUNLIT_FRACTION}")

    samples, pcap_violations = run_longhorizon()
    print(f"  peak-power violations over horizon: {pcap_violations}")

    # --- CSV ---
    os.makedirs("data", exist_ok=True)
    csv_path = "data/longhorizon_stability.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "QE_over_t", "QP_over_t", "QU_over_t"])
        for Tnow, qe, qp, qu in samples:
            w.writerow([Tnow, f"{qe:.8e}", f"{qp:.8e}", f"{qu:.8e}"])
    print(f"  saved {csv_path}")

    arr = np.array(samples)
    Ts  = arr[:, 0]
    qe  = arr[:, 1]
    qp  = arr[:, 2]
    qu  = arr[:, 3]

    print(f"  final T = {int(Ts[-1])} slots")
    print(f"    QE(T)/T = {qe[-1]:.6e}")
    print(f"    QP(T)/T = {qp[-1]:.6e}")
    print(f"    QU(T)/T = {qu[-1]:.6e}")

    # --- plot ---
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.plot(Ts, qe, "-",  color=PALETTE["blue_main"],     lw=2.2,
            label=r"$Q^E(T)/T$")
    ax.plot(Ts, qp, "-",  color=PALETTE["red_strong"],    lw=2.2,
            label=r"$Q^P(T)/T$")
    ax.plot(Ts, qu, "-",  color=PALETTE["green_3"],       lw=2.6,
            label=r"$Q^U(T)/T$")
    ax.axhline(0.0, color=PALETTE["neutral"], lw=1.0, ls=":")
    ax.set_xlabel("Horizon $T$ [slots]")
    ax.set_ylabel(r"running backlog rate $Q(T)/T$")
    ax.set_title(r"$Q(T)/T \to 0$")
    ax.legend(loc="upper right", fontsize=13)
    ax.grid(alpha=0.3)

    savefig_pub(fig, "longhorizon_stability.pdf")
    fig.savefig("longhorizon_stability.png", dpi=200, bbox_inches="tight")
    print("  saved longhorizon_stability.pdf / longhorizon_stability.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
