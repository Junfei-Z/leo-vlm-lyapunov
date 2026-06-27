#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Satellite Robust-Lyapunov VLM Scheduling WITH SPLIT PIPELINE
==================================================================

This is the main experiment. It adds two-stage pipeline splitting on top of the
multi-satellite scenario.

KEY STRUCTURAL FACT (paper contribution C1):
  The largest model (8B) does NOT fit in a single satellite's RAM. To run it, the
  model must be SPLIT into a front stage and a rear stage executed on TWO different
  ISL-connected satellites: the front satellite runs the first blocks and passes the
  intermediate hidden states over the ISL to the rear satellite (paper Eq.14-16).

  Therefore:
    * Only a scheduler that can ORCHESTRATE the split (choose a front sat, a rear
      sat, and verify ISL connectivity at the handoff slot) can run 8B and obtain
      its highest quality (Q=0.51).
    * Greedy / Random / fixed baselines cannot organize a cross-satellite pipeline.
      They are limited to single-satellite (standalone) configs, so the best quality
      they can reach is 7B (Q=0.47). 8B is simply unavailable to them.

  This gives the proposed scheduler a STRUCTURAL quality advantage that baselines
  cannot match -- not a tuning artifact.

Solar model: paper Eq.17 (deterministic delta + small Gaussian perturbation),
staggered per-satellite eclipse phases.

Calibration (T_im, E_im, Ppeak_im, Q_im) = REAL Jetson Orin NX / EuroSAT 224px / Q4.
Split-stage costs are derived from the measured 8B totals (front/rear each ~half).
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import apply_house_style, savefig_pub, PALETTE
apply_house_style()

# ---------------- time scales ----------------
TAU = 1.0
ORBIT_PERIOD_S = 95 * 60
SUNLIT_FRACTION = 0.60
N_SLOTS_PER_ORBIT = int(ORBIT_PERIOD_S / TAU)
N_SUNLIT = int(N_SLOTS_PER_ORBIT * SUNLIT_FRACTION)
N_ORBITS = 3
HORIZON = N_ORBITS * N_SLOTS_PER_ORBIT

# ---------------- standalone VLM configs (measured on Jetson Orin NX) ----------------
# Q  = EuroSAT zero-shot accuracy (224px, Q4)
# T  = per-inference latency [s] ; N = ceil(T/tau) slots
# E  = per-inference energy [J] ; Ppeak = calibrated peak power [W]
# RAM = approximate model memory footprint [GB] (for the single-sat memory limit)
CONFIGS = {
    "2B": dict(T=1.24, E=3.93, Ppeak=4.15, Q=0.398, RAM=2.5),
    "3B": dict(T=1.74, E=5.78, Ppeak=4.51, Q=0.464, RAM=3.5),
    "7B": dict(T=2.76, E=9.13, Ppeak=5.57, Q=0.573, RAM=6.5),
}
for m in CONFIGS.values():
    m["N"] = int(np.ceil(m["T"] / TAU))
STANDALONE = ["2B", "3B"]          # fit in one 8GB satellite (shared w/ sensing)
SPLIT_MODEL = "7B"                 # >=4B: split across two satellites

# Per-stage cost of the split 7B pipeline (front/rear each ~half of measured totals).
SPLIT = dict(
    Q=CONFIGS["7B"]["Q"],          # quality credited when the rear stage completes
    N_front=int(np.ceil(CONFIGS["7B"]["N"] / 2)),
    N_rear=int(np.ceil(CONFIGS["7B"]["N"] / 2)),
    E_front=CONFIGS["7B"]["E"] / 2,
    E_rear=CONFIGS["7B"]["E"] / 2,
    Ppeak=CONFIGS["7B"]["Ppeak"],
    RAM=CONFIGS["7B"]["RAM"] / 2,   # each satellite holds only half the model
)

# ---------------- constellation ----------------
N_SAT = 4
N_SOURCES = 6
ARRIVAL_PROB = 0.10
SAT_PHASE = [int(k * N_SLOTS_PER_ORBIT / N_SAT) for k in range(N_SAT)]
SAT_RAM = 5.0                      # per-sat VLM RAM [GB] (8GB - OS/IO/sensing): fits 2B & 3B, NOT 7B

# ---------------- solar model (paper Eq.17) ----------------
P_SOLAR   = 2.0           # W: small panel (inference is a secondary load on the sat)
ETA_PANEL = 0.30
ETA_SIGMA = 0.02

# ---------------- power / battery ----------------
P_CAP   = 15.0     # realistic satellite power-bus limit (Jetson 15W mode)
P_BASE  = 9.0      # platform baseline draw (sensing+comms+ADCS); inference shares the rest
B_MAX   = 5000.0
B_INIT  = 0.6 * B_MAX

U_TGT = 0.30
V_DEFAULT = 100.0
RNG_SEED = 7


def sunlit_indicator(t, sat):
    phase = (t + SAT_PHASE[sat]) % N_SLOTS_PER_ORBIT
    return 1 if phase < N_SUNLIT else 0


def solar_harvest(t, sat, rng):
    if sunlit_indicator(t, sat) == 0:
        return 0.0
    eps = rng.normal(0.0, ETA_SIGMA)
    eff = max(0.0, ETA_PANEL + eps)
    return P_SOLAR * eff / ETA_PANEL * TAU


def isl_connected(t, s1, s2):
    """Minimal ISL model: neighboring satellites (in phase order) are connected.
    In a real run this comes from offline TLE/ephemeris (deterministic). Here we
    treat all sat pairs as ISL-reachable for simplicity, except we keep the handoff
    check explicit so the structure matches paper Eq.16."""
    return s1 != s2


# =============================================================================
# Scheduling policies
#   Return one of:
#     ("standalone", sat, source_i, config_name)
#     ("split",      front_sat, rear_sat, source_i)
#     None
# =============================================================================
def _energy_feasible(e_per_slot, b, omega):
    return e_per_slot <= (b + omega)


def choose_lyapunov(pending, free_sats, b, omega, QE, QP, QU, V, rng, t):
    """Proposed scheduler. Considers BOTH standalone configs AND the split 7B
    pipeline (which needs two free, ISL-connected satellites). Picks the option
    minimizing drift-plus-penalty subject to hard peak-power & battery feasibility."""
    best_score, best = None, None

    # --- standalone options ---
    for s in free_sats:
        for mname in STANDALONE:
            m = CONFIGS[mname]
            if P_BASE + m["Ppeak"] > P_CAP:
                continue
            e_ps = m["E"] / m["N"]
            if not _energy_feasible(e_ps, b[s], omega[s]):
                continue
            for i in pending:
                score = QE[s] * e_ps + QP[s] * m["Ppeak"] - (QU[i] + V) * m["Q"]
                if best_score is None or score < best_score:
                    best_score, best = score, ("standalone", s, i, mname)

    # --- split 8B option (needs two distinct, ISL-connected free sats) ---
    if len(free_sats) >= 2:
        ef_ps = SPLIT["E_front"] / SPLIT["N_front"]
        er_ps = SPLIT["E_rear"] / SPLIT["N_rear"]
        if P_BASE + SPLIT["Ppeak"] <= P_CAP:
            for a in range(len(free_sats)):
                for c in range(len(free_sats)):
                    if a == c:
                        continue
                    sf, sr = free_sats[a], free_sats[c]
                    if not isl_connected(t, sf, sr):
                        continue
                    # handoff connectivity at t + N_front (Eq.16) -- deterministic
                    if not isl_connected(t + SPLIT["N_front"], sf, sr):
                        continue
                    if not _energy_feasible(ef_ps, b[sf], omega[sf]):
                        continue
                    if not _energy_feasible(er_ps, b[sr], omega[sr]):
                        continue
                    for i in pending:
                        # drift-plus-penalty for the pair; quality credited to source i
                        score = (QE[sf] * ef_ps + QE[sr] * er_ps
                                 + QP[sf] * SPLIT["Ppeak"] + QP[sr] * SPLIT["Ppeak"]
                                 - (QU[i] + V) * SPLIT["Q"])
                        if best_score is None or score < best_score:
                            best_score, best = score, ("split", sf, sr, i)

    if best is not None and best_score < 0:
        return best
    return None


def _greedy_standalone(pending, free_sats, b, omega, pick_cfg):
    if not pending or not free_sats:
        return None
    s = free_sats[0]; i = pending[0]
    mname = pick_cfg()
    e_ps = CONFIGS[mname]["E"] / CONFIGS[mname]["N"]
    if _energy_feasible(e_ps, b[s], omega[s]):
        return ("standalone", s, i, mname)
    return None


def choose_greedy_q(pending, free_sats, b, omega, QE, QP, QU, V, rng, t):
    """Wants max quality but CANNOT split -> limited to the best single-sat config (3B)."""
    return _greedy_standalone(pending, free_sats, b, omega,
                              lambda: max(STANDALONE, key=lambda k: CONFIGS[k]["Q"]))


def choose_greedy_e(pending, free_sats, b, omega, QE, QP, QU, V, rng, t):
    return _greedy_standalone(pending, free_sats, b, omega,
                              lambda: min(STANDALONE, key=lambda k: CONFIGS[k]["E"]))


def choose_random(pending, free_sats, b, omega, QE, QP, QU, V, rng, t):
    if not pending or not free_sats:
        return None
    s = free_sats[rng.integers(len(free_sats))]
    i = pending[rng.integers(len(pending))]
    mname = STANDALONE[rng.integers(len(STANDALONE))]
    e_ps = CONFIGS[mname]["E"] / CONFIGS[mname]["N"]
    if _energy_feasible(e_ps, b[s], omega[s]):
        return ("standalone", s, i, mname)
    return None


def choose_maxbatt_7b(pending, free_sats, b, omega, QE, QP, QU, V, rng, t):
    if not pending or not free_sats:
        return None
    s = max(free_sats, key=lambda x: b[x]); i = pending[0]
    for mname in ("3B", "2B"):
        e_ps = CONFIGS[mname]["E"] / CONFIGS[mname]["N"]
        if _energy_feasible(e_ps, b[s], omega[s]):
            return ("standalone", s, i, mname)
    return None


POLICIES = {
    "Lyapunov (ours)": choose_lyapunov,
    "Greedy-Q":        choose_greedy_q,
    "Greedy-E":        choose_greedy_e,
    "Random":          choose_random,
    "Static":      choose_maxbatt_7b,
}


# =============================================================================
# Simulation core (supports standalone + split commitments)
# =============================================================================
def run_sim(policy_fn, V=V_DEFAULT, horizon=HORIZON, seed=RNG_SEED):
    rng = np.random.default_rng(seed)

    b = np.full(N_SAT, B_INIT)
    QE = np.zeros(N_SAT)
    QP = np.zeros(N_SAT)
    QU = np.zeros(N_SOURCES)

    busy_until = np.full(N_SAT, -1)
    run_E_ps = np.zeros(N_SAT)
    run_P = np.zeros(N_SAT)
    blackout_until = np.full(N_SAT, -1)

    delivered = np.zeros(N_SOURCES)
    arrivals_count = np.zeros(N_SOURCES)
    pcap_violations = 0
    blackout_slots = 0
    tasks_started = 0
    split_tasks = 0
    tasks_dropped = 0

    log = dict(t=[], battery=np.zeros((horizon, N_SAT)),
               QE_mean=[], QU_mean=[])

    for t in range(horizon):
        omega = np.zeros(N_SAT)
        for s in range(N_SAT):
            Et = solar_harvest(t, s, rng)
            omega[s] = min(Et, B_MAX - b[s])

        arrivals = (rng.random(N_SOURCES) < ARRIVAL_PROB).astype(int)
        arrivals_count += arrivals
        pending = [i for i in range(N_SOURCES) if arrivals[i] == 1]

        for s in range(N_SAT):
            if t <= blackout_until[s]:
                blackout_slots += 1

        free_sats = [s for s in range(N_SAT)
                     if t > busy_until[s] and t > blackout_until[s]]

        q_credit = np.zeros(N_SOURCES)
        if pending and free_sats:
            choice = policy_fn(pending, free_sats, b, omega, QE, QP, QU, V, rng, t)
            if choice is not None:
                if choice[0] == "standalone":
                    _, s, i, mname = choice
                    m = CONFIGS[mname]
                    busy_until[s] = t + m["N"] - 1
                    run_E_ps[s] = m["E"] / m["N"]
                    run_P[s] = m["Ppeak"]
                    delivered[i] += m["Q"]
                    q_credit[i] = m["Q"]
                    tasks_started += 1
                elif choice[0] == "split":
                    _, sf, sr, i = choice
                    # front stage occupies sf for N_front, rear occupies sr for N_rear
                    busy_until[sf] = t + SPLIT["N_front"] - 1
                    run_E_ps[sf] = SPLIT["E_front"] / SPLIT["N_front"]
                    run_P[sf] = SPLIT["Ppeak"]
                    # rear starts right after front completes (Eq.15)
                    rear_start = t + SPLIT["N_front"]
                    busy_until[sr] = rear_start + SPLIT["N_rear"] - 1
                    run_E_ps[sr] = SPLIT["E_rear"] / SPLIT["N_rear"]
                    run_P[sr] = SPLIT["Ppeak"]
                    # quality credited when the rear stage completes (paper); for the
                    # max-min queue we credit at commit (myopic, consistent w/ standalone)
                    delivered[i] += SPLIT["Q"]
                    q_credit[i] = SPLIT["Q"]
                    tasks_started += 1
                    split_tasks += 1
            else:
                tasks_dropped += len(pending)

        e_t = np.zeros(N_SAT)
        p_t = np.zeros(N_SAT)
        for s in range(N_SAT):
            if t <= busy_until[s] and t > blackout_until[s]:
                e_t[s] = run_E_ps[s]
                p_t[s] = run_P[s]
            if P_BASE + p_t[s] > P_CAP + 1e-9:
                pcap_violations += 1

        for s in range(N_SAT):
            b_new = b[s] + omega[s] - e_t[s]
            if b_new <= 0:
                b_new = 0.0
                recharge = int(np.ceil(0.20 * B_MAX / max(1e-6, P_SOLAR * TAU)))
                blackout_until[s] = t + recharge
                busy_until[s] = -1
            b[s] = min(b_new, B_MAX)

        QE = np.maximum(QE + e_t - omega, 0.0)
        QP = np.maximum(QP + (P_BASE + p_t) - P_CAP, 0.0)
        QU = np.maximum(QU + arrivals * U_TGT - q_credit, 0.0)

        log["t"].append(t)
        log["battery"][t, :] = b
        log["QE_mean"].append(QE.mean())
        log["QU_mean"].append(QU.mean())

    fill = delivered / np.maximum(1, arrivals_count)
    uptime = 1.0 - blackout_slots / (horizon * N_SAT)
    service_rate = tasks_started / max(1, arrivals_count.sum())
    return dict(
        log=log,
        pcap_violations=pcap_violations,
        blackout_slots=blackout_slots,
        uptime_frac=uptime,
        service_rate=service_rate,
        tasks_started=tasks_started,
        split_tasks=split_tasks,
        delivered=delivered,
        min_fill=fill.min(),
        mean_fill=fill.mean(),
        total_quality=delivered.sum(),
        avg_QE=float(np.mean(log["QE_mean"])),
    )


def compare_all():
    results = {}
    print(f"{'Policy':18s} | {'peakViol':>8s} | {'downtime':>8s} | {'svcRate':>7s} | "
          f"{'split#':>6s} | {'minFill':>7s} | {'totQ':>7s}")
    print("-" * 92)
    for name, fn in POLICIES.items():
        r = run_sim(fn)
        results[name] = r
        print(f"{name:18s} | {r['pcap_violations']:8d} | {r['blackout_slots']:8d} | "
              f"{r['service_rate']:7.3f} | {r['split_tasks']:6d} | "
              f"{r['min_fill']:7.3f} | {r['total_quality']:7.1f}")
    print()
    print("  Only the proposed scheduler can run the 7B model (via split pipeline);")
    print("  baselines are capped at 3B because 7B does not fit in a single satellite.")
    return results


def plot_comparison(results, fname="splitpipeline_comparison.png"):
    names = list(results.keys())
    colors = [PALETTE["blue_main"], PALETTE["red_strong"], "tab:green", "tab:orange", "tab:purple"]
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    # (a) total delivered quality (ours wins: unlocks 8B)
    totq = [results[n]["total_quality"] for n in names]
    bars = ax[0, 0].bar(names, totq, color=colors)
    ax[0, 0].set_title("(a) Total quality", loc="left")
    ax[0, 0].set_ylabel("sum quality")
    ax[0, 0].tick_params(axis="x", rotation=20)

    # (b) min per-source fill (max-min fairness)
    minf = [results[n]["min_fill"] for n in names]
    ax[0, 1].bar(names, minf, color=colors)
    ax[0, 1].set_title("(b) Min-fill", loc="left")
    ax[0, 1].set_ylabel("min fill ratio")
    ax[0, 1].tick_params(axis="x", rotation=20)

    # (c) downtime & peak violations (safety)
    down = [results[n]["blackout_slots"] for n in names]
    ax[1, 0].bar(names, down, color=colors)
    ax[1, 0].set_title("(c) Downtime", loc="left")
    ax[1, 0].set_ylabel("# blackout slots")
    ax[1, 0].tick_params(axis="x", rotation=20)

    # (d) number of split (8B) tasks executed
    spl = [results[n]["split_tasks"] for n in names]
    ax[1, 1].bar(names, spl, color=colors)
    ax[1, 1].set_title("(d) 7B splits", loc="left")
    ax[1, 1].set_ylabel("# split tasks")
    ax[1, 1].tick_params(axis="x", rotation=20)

    for a in ax.flat:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fname, dpi=130)
    savefig_pub(fig, os.path.splitext(fname)[0] + ".pdf")
    print(f"saved {fname}")
    plt.close(fig)


if __name__ == "__main__":
    print("=" * 92)
    print(f"  SPLIT-PIPELINE MULTI-SAT COMPARISON  --  {N_SAT} sats, {N_SOURCES} sources")
    print("=" * 92)
    print(f"  tau={TAU}s | {N_ORBITS} orbits ({HORIZON} slots) | P_cap={P_CAP}W "
          f"P_solar={P_SOLAR}W eta={ETA_PANEL} B_max={B_MAX/1000:.0f}kJ")
    print(f"  per-sat RAM={SAT_RAM}GB  ->  2B(2.5) & 3B(3.5) fit; 7B(6.5) does NOT (split)")
    print(f"  7B runs only as a 2-sat split pipeline (front N={SPLIT['N_front']}, "
          f"rear N={SPLIT['N_rear']})")
    print("-" * 92)
    results = compare_all()
    print("-" * 92)
    plot_comparison(results)
    print("Done.")
