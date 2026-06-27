#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal Robust-Lyapunov Simulator for Energy-Aware Collaborative VLM Inference
over a Solar-Powered LEO Satellite.

Goal of this MINIMAL version:
  - Single computing satellite, a few sensing sources.
  - Standalone configurations only (no front/rear split yet).
  - Periodic eclipse (delta^t flips on the orbital period).
  - Implements: battery dynamics, 3 virtual queues (energy / power / quality),
    per-slot drift-plus-penalty decision, hard peak-power & battery constraints.
  - Verifies: (1) queue stability, (2) eclipse robustness (queues rise in eclipse,
    recover in sunlight), (3) ZERO peak-power violations, (4) min-quality vs V.

Calibration values (T_im, E_im, Ppeak_im, Q_im) come from REAL Jetson Orin NX
measurements on EuroSAT (224x224), Q4 quantization:

    config        T_im(s)   E_im(J)   Ppeak(W)   acc(Q)
    Qwen2.5-VL-3B   1.32      5.4        6.6       0.30
    Qwen2.5-VL-7B   3.89     27.9       10.7       0.47
    InternVL3-8B    8.66     62.8       11.0       0.51

Author: (your project)
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import apply_house_style, savefig_pub, PALETTE
apply_house_style()

# =============================================================================
# 1. TIME SCALES  (the three-layer separation: tau << T_topo << eclipse)
# =============================================================================
TAU = 1.0                      # scheduling slot duration [s]
ORBIT_PERIOD_S = 95 * 60       # ~95 min LEO orbit  -> 5700 s
SUNLIT_FRACTION = 0.60         # 60% sunlit / 40% eclipse
N_SLOTS_PER_ORBIT = int(ORBIT_PERIOD_S / TAU)          # 5700 slots
N_SUNLIT = int(N_SLOTS_PER_ORBIT * SUNLIT_FRACTION)    # ~3420 slots
N_ECLIPSE = N_SLOTS_PER_ORBIT - N_SUNLIT               # ~2280 slots

N_ORBITS = 3                                           # run 3 full orbits
HORIZON = N_ORBITS * N_SLOTS_PER_ORBIT                 # total slots |T|

# =============================================================================
# 2. VLM CONFIGURATION SET  M  (from real Jetson measurements, all Q4, 224px)
# =============================================================================
# Each config m: T_im (-> N_im = ceil(T/tau)), E_im (J), Ppeak (W), Q (quality)
CONFIGS = {
    "2B": dict(T=1.24, E=3.93, Ppeak=4.15, Q=0.398),
    "3B": dict(T=1.74, E=5.78, Ppeak=4.51, Q=0.464),
    "7B": dict(T=2.76, E=9.13, Ppeak=5.57, Q=0.573),
}
for m in CONFIGS.values():
    m["N"] = int(np.ceil(m["T"] / TAU))   # N_im : number of slots per inference

CONFIG_NAMES = list(CONFIGS.keys())

# =============================================================================
# 3. SYSTEM / ENERGY PARAMETERS
#    Chosen so that eclipse creates real pressure but the system survives.
# =============================================================================
P_CAP   = 15.0     # instantaneous power cap [W]  (above 8B's 11W -> binding-ish)
P_SOLAR = 9.0      # solar charging power when sunlit [W]  (covers avg, not peak)
B_MAX   = 30000.0  # battery capacity [J]  (enough to ride through an eclipse)
B_INIT  = 0.6 * B_MAX

# sensing sources (max-min fairness across these)
# NOTE: arrival rate must be < satellite service capacity, otherwise the quality
# deficit queue grows without bound (system overloaded). A standalone task occupies
# N_im slots (2-9), so the sat can start a new task only every few slots. We keep
# the aggregate arrival rate well below 1 task / (avg N_im) slots.
N_SOURCES = 3
ARRIVAL_PROB = 0.02   # per-slot Bernoulli arrival prob per source (lambda_i)

U_TGT = 0.30          # quality target per source (<= achievable quality; tunes max-min)

# Lyapunov control parameter V (the [O(1/V), O(V)] knob)
V_DEFAULT = 50.0

RNG_SEED = 7

# =============================================================================
# 4. SIMULATOR
# =============================================================================
def sunlit_indicator(t):
    """delta^t : 1 if sunlit, 0 if eclipse. Periodic on the orbit."""
    phase = t % N_SLOTS_PER_ORBIT
    return 1 if phase < N_SUNLIT else 0


def run_sim(V=V_DEFAULT, horizon=HORIZON, seed=RNG_SEED, verbose=False):
    rng = np.random.default_rng(seed)

    # --- state ---
    b = B_INIT                      # battery level [J]
    QE = 0.0                        # energy virtual queue
    QP = 0.0                        # power  virtual queue
    QU = np.zeros(N_SOURCES)        # quality deficit queue per source

    # --- busy bookkeeping: when does the satellite become free, and what is it running ---
    busy_until = -1                 # slot index until which the sat is occupied (inclusive)
    running_E_per_slot = 0.0        # energy/slot of the in-flight task
    running_P = 0.0                 # peak power of the in-flight task

    # --- logs ---
    log = dict(
        t=[], battery=[], QE=[], QP=[], QU_mean=[], delta=[],
        power=[], energy=[], quality=[], chosen=[], pcap_violation=[]
    )
    delivered_quality = np.zeros(N_SOURCES)  # cumulative quality per source
    arrivals_count = np.zeros(N_SOURCES)
    pcap_violations = 0

    for t in range(horizon):
        delta = sunlit_indicator(t)

        # ---- solar harvest ----
        Et_solar = P_SOLAR * TAU * delta            # energy arriving this slot [J]
        # store into battery (omega^t) bounded by capacity
        omega = min(Et_solar, B_MAX - b)
        # (battery updated after consumption below)

        # ---- task arrivals (Bernoulli per source) ----
        arrivals = (rng.random(N_SOURCES) < ARRIVAL_PROB).astype(int)
        arrivals_count += arrivals

        # ---- update quality deficit queues with arrivals (the U_tgt input) ----
        # q^t_i credit is added only when a task COMPLETES (handled below). Here we
        # add the arrival-driven target demand.
        # (We apply the queue update at end of slot, after deciding q^t.)

        # ---------------------------------------------------------------
        # PER-SLOT DECISION (drift-plus-penalty), only if satellite is FREE
        # ---------------------------------------------------------------
        chosen = None
        e_t = 0.0          # energy consumed this slot
        p_t = 0.0          # power drawn this slot
        q_t = np.zeros(N_SOURCES)
        pcap_viol = 0

        if t > busy_until:
            # satellite is free -> may start a new standalone task for some source
            # candidate sources = those with a pending arrival this slot
            pending = [i for i in range(N_SOURCES) if arrivals[i] == 1]

            best_score = None
            best_choice = None  # (source_i, config_name)

            for i in pending:
                for mname in CONFIG_NAMES:
                    m = CONFIGS[mname]
                    e_per_slot = m["E"] / m["N"]      # energy consumed per slot
                    p_peak = m["Ppeak"]

                    # ---- HARD constraints (must hold every slot) ----
                    # peak power cap:
                    if p_peak > P_CAP:
                        continue
                    # battery feasibility (this slot's energy <= available):
                    if e_per_slot > b + omega:
                        continue

                    # ---- drift-plus-penalty objective (minimize) ----
                    # min  QE*e_t + QP*(p - Pcap)  - (QU_i + V)*q_i
                    # here q_i = Q_im delivered when this task eventually completes;
                    # we credit it at decision time as the per-task quality (myopic).
                    q_im = m["Q"]
                    score = (QE * e_per_slot
                             + QP * (p_peak)        # power back-pressure
                             - (QU[i] + V) * q_im)

                    if (best_score is None) or (score < best_score):
                        best_score = score
                        best_choice = (i, mname)

            # also consider doing NOTHING (idle) -> score 0
            if (best_choice is not None) and (best_score < 0):
                i, mname = best_choice
                m = CONFIGS[mname]
                chosen = (i, mname)
                # commit: occupy satellite for N_im slots
                busy_until = t + m["N"] - 1
                running_E_per_slot = m["E"] / m["N"]
                running_P = m["Ppeak"]
                # quality credited at START (rear-stage start == result available)
                q_t[i] = m["Q"]
                delivered_quality[i] += m["Q"]

        # ---- if busy (or just started), this slot consumes energy/power ----
        if t <= busy_until:
            e_t = running_E_per_slot
            p_t = running_P

        # ---- HARD peak-power check (safety; should never trigger) ----
        if p_t > P_CAP + 1e-9:
            pcap_viol = 1
            pcap_violations += 1

        # ---- battery update (eq. 20-style):  b' = min(max(b + omega - e, 0), Bmax) ----
        b = min(max(b + omega - e_t, 0.0), B_MAX)

        # ---- virtual queue updates ----
        # energy queue: Q^E(t+1) = max(Q^E + e - omega, 0)
        QE = max(QE + e_t - omega, 0.0)
        # power queue: Q^P(t+1) = max(Q^P + p - Pcap, 0)
        QP = max(QP + p_t - P_CAP, 0.0)
        # quality queue per source: Q^U_i = max(Q^U_i + a_i*U_tgt - q_i, 0)
        QU = np.maximum(QU + arrivals * U_TGT - q_t, 0.0)

        # ---- log ----
        log["t"].append(t)
        log["battery"].append(b)
        log["QE"].append(QE)
        log["QP"].append(QP)
        log["QU_mean"].append(QU.mean())
        log["delta"].append(delta)
        log["power"].append(p_t)
        log["energy"].append(e_t)
        log["quality"].append(q_t.sum())
        log["chosen"].append(chosen)
        log["pcap_violation"].append(pcap_viol)

    # ---- summary ----
    T_eff = horizon * TAU
    avg_quality_per_source = delivered_quality / max(1, horizon) * N_SLOTS_PER_ORBIT  # rough rate
    result = dict(
        log=log,
        pcap_violations=pcap_violations,
        delivered_quality=delivered_quality,
        arrivals_count=arrivals_count,
        min_quality_rate=(delivered_quality / np.maximum(1, arrivals_count)).min(),
        mean_quality_rate=(delivered_quality / np.maximum(1, arrivals_count)).mean(),
        final_QE=QE, final_QP=QP, final_QU=QU.copy(),
        V=V,
    )
    if verbose:
        print(f"V={V:6.1f} | peak-power violations={pcap_violations} | "
              f"min per-source quality-fill={result['min_quality_rate']:.3f} | "
              f"final QE={QE:8.1f} QP={QP:6.1f} QU_mean={QU.mean():6.2f}")
    return result


# =============================================================================
# 5. PLOTS  (the evidence for queue stability + eclipse robustness)
# =============================================================================
def plot_timeseries(res, fname="sim_timeseries.png"):
    log = res["log"]
    t = np.array(log["t"]) * TAU / 60.0   # minutes
    delta = np.array(log["delta"])

    fig, ax = plt.subplots(4, 1, figsize=(11, 11), sharex=True)

    # shade eclipse regions
    def shade(a):
        in_ecl = False
        start = 0
        for k in range(len(delta)):
            if delta[k] == 0 and not in_ecl:
                in_ecl = True; start = t[k]
            elif delta[k] == 1 and in_ecl:
                in_ecl = False; a.axvspan(start, t[k], color="0.85", zorder=0)
        if in_ecl:
            a.axvspan(start, t[-1], color="0.85", zorder=0)

    # (a) battery
    shade(ax[0])
    ax[0].plot(t, np.array(log["battery"]) / 1000.0, color="tab:green")
    ax[0].set_ylabel("Battery [kJ]")
    ax[0].set_title("(a) Battery level  (shaded = eclipse)")

    # (b) energy virtual queue
    shade(ax[1])
    ax[1].plot(t, log["QE"], color="tab:red")
    ax[1].set_ylabel(r"$Q^E(t)$")
    ax[1].set_title("(b) Energy virtual queue  (rises in eclipse, recovers in sunlight)")

    # (c) power virtual queue + peak power
    shade(ax[2])
    ax[2].plot(t, log["QP"], color="tab:purple", label=r"$Q^P(t)$")
    ax[2].set_ylabel(r"$Q^P(t)$")
    ax[2].set_title("(c) Power virtual queue  (stays ~0 -> peak cap respected)")

    # (d) quality deficit queue (mean over sources)
    shade(ax[3])
    ax[3].plot(t, log["QU_mean"], color="tab:blue")
    ax[3].set_ylabel(r"mean $Q^U_i(t)$")
    ax[3].set_title("(d) Quality deficit queue (mean over sources)")
    ax[3].set_xlabel("Time [min]")

    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fname, dpi=130)
    savefig_pub(fig, os.path.splitext(fname)[0] + ".pdf")
    print(f"saved {fname}")
    plt.close(fig)


def plot_V_tradeoff(Vs, fname="sim_V_tradeoff.png"):
    minq, qe = [], []
    for V in Vs:
        r = run_sim(V=V, verbose=True)
        minq.append(r["min_quality_rate"])
        qe.append(np.mean(r["log"]["QE"]))
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(Vs, minq, "o-", color="tab:blue", label="min per-source quality fill")
    ax1.set_xlabel("Lyapunov parameter V")
    ax1.set_ylabel("min-quality fill ratio", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(Vs, qe, "s--", color="tab:red", label="avg energy queue backlog")
    ax2.set_ylabel("avg $Q^E$ backlog", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax1.set_title("[O(1/V), O(V)] trade-off:  larger V -> more quality, larger backlog")
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fname, dpi=130)
    savefig_pub(fig, os.path.splitext(fname)[0] + ".pdf")
    print(f"saved {fname}")
    plt.close(fig)


# =============================================================================
# 6. MAIN
# =============================================================================
if __name__ == "__main__":
    print("=" * 64)
    print("  Minimal Robust-Lyapunov VLM Scheduling Simulator")
    print("=" * 64)
    print(f"  tau={TAU}s | orbit={N_SLOTS_PER_ORBIT} slots "
          f"(sunlit {N_SUNLIT}, eclipse {N_ECLIPSE}) | horizon={HORIZON} slots "
          f"({N_ORBITS} orbits)")
    print(f"  N_im per config: " + ", ".join(f"{k}={v['N']}" for k, v in CONFIGS.items()))
    print(f"  P_cap={P_CAP}W  P_solar={P_SOLAR}W  B_max={B_MAX/1000:.0f}kJ")
    print("-" * 64)

    # main run at default V
    res = run_sim(V=V_DEFAULT, verbose=True)
    print("-" * 64)
    print(f"  PEAK-POWER VIOLATIONS over {HORIZON} slots : {res['pcap_violations']}")
    print(f"  per-source delivered quality (cumulative)  : "
          + ", ".join(f"{q:.0f}" for q in res["delivered_quality"]))
    print(f"  min/mean per-source quality-fill ratio     : "
          f"{res['min_quality_rate']:.3f} / {res['mean_quality_rate']:.3f}")
    print("-" * 64)

    plot_timeseries(res)
    plot_V_tradeoff([5, 10, 20, 50, 100, 200])
    print("Done.")
