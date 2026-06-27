#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Satellite Robust-Lyapunov VLM Scheduling  --  BASELINE COMPARISON

Multiple solar-powered LEO computing satellites with STAGGERED eclipse phases.
Sensing sources generate tasks that can be offloaded to any ISL-reachable
computing satellite. The scheduler decides BOTH which satellite and which VLM
configuration to use.

This is where the proposed scheduler structurally wins: it can route a task to a
satellite that is still SUNLIT / has battery headroom, while greedy baselines keep
hammering whatever satellite/config maximizes instantaneous quality, draining
eclipse-side satellites into blackout.

Solar model (paper Eq. 17, NOT a terrestrial weather HMM):
    E^t_s = P_solar * (eta_s + eps^t_s) * tau * delta^t_s
  - delta^t_s : DETERMINISTIC sunlit indicator from orbital mechanics (0/1),
                with a per-satellite phase offset (staggered eclipse).
  - eta_s     : panel efficiency (space-grade GaAs ~0.30).
  - eps^t_s   : small Gaussian perturbation (attitude/aging), ~ N(0, sigma^2).

Calibration (T_im, E_im, Ppeak_im, Q_im) = REAL Jetson Orin NX / EuroSAT 224px / Q4.
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

# ---------------- VLM configs (measured) ----------------
CONFIGS = {
    "2B": dict(T=1.24, E=3.93, Ppeak=4.15, Q=0.398),
    "3B": dict(T=1.74, E=5.78, Ppeak=4.51, Q=0.464),
    "7B": dict(T=2.76, E=9.13, Ppeak=5.57, Q=0.573),
}
for m in CONFIGS.values():
    m["N"] = int(np.ceil(m["T"] / TAU))
CONFIG_NAMES = list(CONFIGS.keys())

# ---------------- constellation ----------------
N_SAT = 4                 # computing satellites
N_SOURCES = 6             # more sources -> higher aggregate load
ARRIVAL_PROB = 0.12       # high load: system is genuinely resource-constrained

# each satellite has a phase offset so eclipses are STAGGERED
SAT_PHASE = [int(k * N_SLOTS_PER_ORBIT / N_SAT) for k in range(N_SAT)]

# ---------------- solar model (paper Eq.17) ----------------
P_SOLAR   = 2.0           # W: small panel (energy-balanced ~1.6W demand + margin)
ETA_PANEL = 0.30          # space-grade GaAs efficiency ~30%
ETA_SIGMA = 0.02          # small Gaussian perturbation on efficiency (attitude/aging)

# ---------------- power / battery ----------------
P_CAP   = 15.0
P_BASE  = 9.0      # platform baseline draw (sensing+comms+ADCS); shares the bus
B_MAX   = 4000.0          # smaller battery -> eclipse is genuinely hard
B_INIT  = 0.6 * B_MAX

U_TGT = 0.30
V_DEFAULT = 100.0
RNG_SEED = 7


def sunlit_indicator(t, sat):
    """Deterministic delta^t_s with per-satellite phase offset (staggered eclipse)."""
    phase = (t + SAT_PHASE[sat]) % N_SLOTS_PER_ORBIT
    return 1 if phase < N_SUNLIT else 0


def solar_harvest(t, sat, rng):
    """Paper Eq.17: P_solar * (eta + eps) * tau * delta, eps~N(0,sigma)."""
    delta = sunlit_indicator(t, sat)
    if delta == 0:
        return 0.0
    eps = rng.normal(0.0, ETA_SIGMA)
    eff = max(0.0, ETA_PANEL + eps)
    return P_SOLAR * eff / ETA_PANEL * TAU   # normalized so full-sun ~ P_SOLAR*tau


# =============================================================================
# Scheduling policies.
# Each returns (sat, source_i, config_name) or None.
# Inputs: list of pending sources, per-sat state arrays.
# =============================================================================
def _energy_feasible(mname, b, omega):
    return (CONFIGS[mname]["E"] / CONFIGS[mname]["N"]) <= (b + omega)


def choose_lyapunov(pending, free_sats, b, omega, QE, QP, QU, V, rng):
    """Proposed: jointly choose (satellite, source, config) minimizing drift-plus-penalty,
    subject to HARD peak-power and battery feasibility per satellite.
    Routes work to sunlit / high-battery satellites because their Q^E backlog is lower."""
    best_score, best = None, None
    for s in free_sats:
        for i in pending:
            for mname in CONFIG_NAMES:
                m = CONFIGS[mname]
                if P_BASE + m["Ppeak"] > P_CAP:                       # HARD peak-power
                    continue
                if not _energy_feasible(mname, b[s], omega[s]):  # HARD battery
                    continue
                e_ps = m["E"] / m["N"]
                score = QE[s] * e_ps + QP[s] * m["Ppeak"] - (QU[i] + V) * m["Q"]
                if best_score is None or score < best_score:
                    best_score, best = score, (s, i, mname)
    if best is not None and best_score < 0:
        return best
    return None


def choose_greedy_q(pending, free_sats, b, omega, QE, QP, QU, V, rng):
    """Highest quality, first free sat. No eclipse/energy awareness."""
    if not pending or not free_sats:
        return None
    s = free_sats[0]; i = pending[0]
    mname = max(CONFIG_NAMES, key=lambda k: CONFIGS[k]["Q"])
    if _energy_feasible(mname, b[s], omega[s]):
        return (s, i, mname)
    return None


def choose_greedy_e(pending, free_sats, b, omega, QE, QP, QU, V, rng):
    """Cheapest config, first free sat."""
    if not pending or not free_sats:
        return None
    s = free_sats[0]; i = pending[0]
    mname = min(CONFIG_NAMES, key=lambda k: CONFIGS[k]["E"])
    if _energy_feasible(mname, b[s], omega[s]):
        return (s, i, mname)
    return None


def choose_random(pending, free_sats, b, omega, QE, QP, QU, V, rng):
    if not pending or not free_sats:
        return None
    s = free_sats[rng.integers(len(free_sats))]
    i = pending[rng.integers(len(pending))]
    mname = CONFIG_NAMES[rng.integers(len(CONFIG_NAMES))]
    if _energy_feasible(mname, b[s], omega[s]):
        return (s, i, mname)
    return None


def choose_maxbatt_fixed(pending, free_sats, b, omega, QE, QP, QU, V, rng):
    """Energy-aware-ish baseline: route to the highest-battery sat, fixed 7B config.
    Smarter than greedy but still ignores quality fairness and peak/config tradeoff."""
    if not pending or not free_sats:
        return None
    s = max(free_sats, key=lambda x: b[x])
    i = pending[0]
    mname = "3B"
    if _energy_feasible(mname, b[s], omega[s]):
        return (s, i, mname)
    # fall back to 2B if 3B not feasible
    if _energy_feasible("2B", b[s], omega[s]):
        return (s, i, "2B")
    return None


POLICIES = {
    "Lyapunov (ours)": choose_lyapunov,
    "Greedy-Q":        choose_greedy_q,
    "Greedy-E":        choose_greedy_e,
    "Random":          choose_random,
    "Static":      choose_maxbatt_fixed,
}


# =============================================================================
# Multi-satellite simulation core
# =============================================================================
def run_sim(policy_fn, V=V_DEFAULT, horizon=HORIZON, seed=RNG_SEED):
    rng = np.random.default_rng(seed)

    b = np.full(N_SAT, B_INIT)             # battery per sat
    QE = np.zeros(N_SAT)                   # energy queue per sat
    QP = np.zeros(N_SAT)                   # power queue per sat
    QU = np.zeros(N_SOURCES)               # quality deficit per source

    busy_until = np.full(N_SAT, -1)        # per-sat busy
    run_E_ps = np.zeros(N_SAT)
    run_P = np.zeros(N_SAT)
    blackout_until = np.full(N_SAT, -1)

    delivered = np.zeros(N_SOURCES)
    arrivals_count = np.zeros(N_SOURCES)
    pcap_violations = 0
    blackout_slots = 0
    tasks_started = 0
    tasks_dropped = 0

    log = dict(t=[], battery=np.zeros((horizon, N_SAT)),
               QE_mean=[], QU_mean=[], any_sunlit=[])

    for t in range(horizon):
        # solar per sat
        omega = np.zeros(N_SAT)
        deltas = np.zeros(N_SAT)
        for s in range(N_SAT):
            Et = solar_harvest(t, s, rng)
            omega[s] = min(Et, B_MAX - b[s])
            deltas[s] = sunlit_indicator(t, s)

        # arrivals
        arrivals = (rng.random(N_SOURCES) < ARRIVAL_PROB).astype(int)
        arrivals_count += arrivals
        pending = [i for i in range(N_SOURCES) if arrivals[i] == 1]

        # blackout bookkeeping
        for s in range(N_SAT):
            if t <= blackout_until[s]:
                blackout_slots += 1

        # which sats are free AND not in blackout
        free_sats = [s for s in range(N_SAT)
                     if t > busy_until[s] and t > blackout_until[s]]

        # one scheduling decision per slot (could be extended to multiple)
        q_credit = np.zeros(N_SOURCES)   # quality delivered to each source this slot
        if pending and free_sats:
            choice = policy_fn(pending, free_sats, b, omega, QE, QP, QU, V, rng)
            if choice is not None:
                s, i, mname = choice
                m = CONFIGS[mname]
                busy_until[s] = t + m["N"] - 1
                run_E_ps[s] = m["E"] / m["N"]
                run_P[s] = m["Ppeak"]
                delivered[i] += m["Q"]
                q_credit[i] = m["Q"]               # credit applied in queue update below
                tasks_started += 1
            else:
                tasks_dropped += len(pending)

        # per-sat energy/power consumption this slot
        e_t = np.zeros(N_SAT)
        p_t = np.zeros(N_SAT)
        for s in range(N_SAT):
            if t <= busy_until[s] and t > blackout_until[s]:
                e_t[s] = run_E_ps[s]
                p_t[s] = run_P[s]
            if P_BASE + p_t[s] > P_CAP + 1e-9:
                pcap_violations += 1

        # battery update + blackout trigger
        for s in range(N_SAT):
            b_new = b[s] + omega[s] - e_t[s]
            if b_new <= 0:
                b_new = 0.0
                safe_level = 0.20 * B_MAX
                recharge = int(np.ceil(safe_level / max(1e-6, P_SOLAR * TAU)))
                blackout_until[s] = t + recharge
                busy_until[s] = -1
            b[s] = min(b_new, B_MAX)

        # queue updates
        QE = np.maximum(QE + e_t - omega, 0.0)
        QP = np.maximum(QP + (P_BASE + p_t) - P_CAP, 0.0)
        # quality deficit queue (paper Eq.30): demand from arrivals minus delivered credit
        QU = np.maximum(QU + arrivals * U_TGT - q_credit, 0.0)

        log["t"].append(t)
        log["battery"][t, :] = b
        log["QE_mean"].append(QE.mean())
        log["QU_mean"].append(QU.mean())
        log["any_sunlit"].append(int(deltas.sum() > 0))

    fill = delivered / np.maximum(1, arrivals_count)
    uptime = 1.0 - blackout_slots / (horizon * N_SAT)
    # service rate = fraction of arrivals actually served (dropped during blackout/overload hurt)
    total_arrivals = arrivals_count.sum()
    service_rate = tasks_started / max(1, total_arrivals)
    return dict(
        log=log,
        pcap_violations=pcap_violations,
        blackout_slots=blackout_slots,
        uptime_frac=uptime,
        tasks_started=tasks_started,
        tasks_dropped=tasks_dropped,
        service_rate=service_rate,
        delivered=delivered,
        min_fill=fill.min(),
        mean_fill=fill.mean(),
        total_quality=delivered.sum(),
        avg_QE=float(np.mean(log["QE_mean"])),
    )


def compare_all():
    results = {}
    print(f"{'Policy':18s} | {'peakViol':>8s} | {'downtime':>8s} | {'svcRate':>7s} | "
          f"{'minFill':>7s} | {'totQ':>7s}")
    print("-" * 82)
    for name, fn in POLICIES.items():
        r = run_sim(fn)
        results[name] = r
        print(f"{name:18s} | {r['pcap_violations']:8d} | {r['blackout_slots']:8d} | "
              f"{r['service_rate']:7.3f} | {r['min_fill']:7.3f} | "
              f"{r['total_quality']:7.1f}")
    return results


def plot_comparison(results, fname="multisat_comparison.png"):
    names = list(results.keys())
    colors = [PALETTE["blue_main"], PALETTE["red_strong"], "tab:green", "tab:orange", "tab:purple"]
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    # (a) downtime
    down = [results[n]["blackout_slots"] for n in names]
    ax[0, 0].bar(names, down, color=colors)
    ax[0, 0].set_title("(a) Total satellite downtime  (lower better)")
    ax[0, 0].set_ylabel("# blackout slots (all sats)")
    ax[0, 0].tick_params(axis="x", rotation=20)

    # (b) min per-source quality fill (max-min fairness)
    minf = [results[n]["min_fill"] for n in names]
    ax[0, 1].bar(names, minf, color=colors)
    ax[0, 1].set_title("(b) Min per-source quality fill  (higher better, max-min)")
    ax[0, 1].set_ylabel("min fill ratio")
    ax[0, 1].tick_params(axis="x", rotation=20)

    # (c) total quality
    totq = [results[n]["total_quality"] for n in names]
    ax[1, 0].bar(names, totq, color=colors)
    ax[1, 0].set_title("(c) Total delivered quality")
    ax[1, 0].set_ylabel("sum quality")
    ax[1, 0].tick_params(axis="x", rotation=20)

    # (d) mean battery across sats over time
    for n, c in zip(names, colors):
        bt = results[n]["log"]["battery"].mean(axis=1) / 1000.0
        tt = np.array(results[n]["log"]["t"]) * TAU / 60.0
        ax[1, 1].plot(tt, bt, color=c, label=n, lw=1.2)
    ax[1, 1].set_title("(d) Mean battery across satellites")
    ax[1, 1].set_xlabel("Time [min]"); ax[1, 1].set_ylabel("Mean battery [kJ]")
    ax[1, 1].legend(fontsize=8, loc="lower right")

    for a in ax.flat:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fname, dpi=130)
    savefig_pub(fig, os.path.splitext(fname)[0] + ".pdf")
    print(f"saved {fname}")
    plt.close(fig)


if __name__ == "__main__":
    print("=" * 92)
    print(f"  MULTI-SAT BASELINE COMPARISON  --  {N_SAT} computing sats (staggered eclipse), "
          f"{N_SOURCES} sources")
    print("=" * 92)
    print(f"  tau={TAU}s | {N_ORBITS} orbits ({HORIZON} slots) | "
          f"P_cap={P_CAP}W P_solar={P_SOLAR}W eta={ETA_PANEL} B_max={B_MAX/1000:.0f}kJ")
    print(f"  sat eclipse phase offsets (slots): {SAT_PHASE}")
    print("-" * 92)
    results = compare_all()
    print("-" * 92)
    plot_comparison(results)
    print("Done.")
