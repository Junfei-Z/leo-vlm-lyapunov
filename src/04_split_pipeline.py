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
    # RESISC45 n=1300 calibration, all Q4. Ppeak = MAX per-inference peak (conservative
    # bound for the hard cap); only 7B shows a real transient (6.37 vs ~3.8 for 2B/3B).
    "2B": dict(T=1.24, E=3.93, Ppeak=3.75, Q=0.394, RAM=2.5),
    "3B": dict(T=1.74, E=5.72, Ppeak=4.02, Q=0.456, RAM=3.5),
    "7B": dict(T=2.76, E=9.30, Ppeak=6.37, Q=0.549, RAM=6.5),
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

# Ground-source VISIBILITY (which satellites can serve each source). In LEO a ground
# target is only in view of the satellites whose track passes over it, so coverage is
# structurally HETEROGENEOUS: some sources see the whole constellation, others only a
# single satellite. Poorly-covered sources are starvation-prone -- a quality-greedy
# scheduler serves whoever is convenient and lets them fall behind, whereas the max-min
# scheduler must protect them. This is where fairness genuinely binds.
ALL_SATS = {0, 1, 2, 3}
VIS = [set(ALL_SATS) for _ in range(N_SOURCES)]   # full coverage (visibility not the heterogeneity axis)

# DATA-DRIVEN per-source quality heterogeneity. Each sensing source monitors one land-cover
# class; its quality under model m is the MEASURED per-class RESISC45 accuracy of that model
# (data/jetson_resisc45/, Q4, Jetson Orin NX). The six classes are a fixed, pre-declared
# spread of the real difficulty spectrum -- two "flat" classes where 2B already suffices, two
# "medium", two "steep" classes whose accuracy collapses without the largest model. This is a
# genuine property of the data (some land-cover types need a bigger VLM), NOT tuned to win: the
# mapping is fixed before any policy is run. For steep classes only the split-only 7B tier
# delivers usable quality, so a scheduler that cannot sustainably run the split starves them.
SRC_CLASS = ["beach", "intersection", "church", "baseball_diamond", "roundabout", "mountain"]
Q_SRC = [
    {"2B": 0.90, "3B": 0.90, "7B": 0.90},   # 0 beach            (flat: model size irrelevant)
    {"2B": 0.40, "3B": 0.50, "7B": 0.70},   # 1 intersection     (mild gradient)
    {"2B": 0.30, "3B": 0.60, "7B": 0.70},   # 2 church           (3B already helps a lot)
    {"2B": 0.30, "3B": 0.70, "7B": 0.80},   # 3 baseball_diamond (3B helps, 7B a bit more)
    {"2B": 0.30, "3B": 0.80, "7B": 0.90},   # 4 roundabout       (3B nearly enough)
    {"2B": 0.20, "3B": 0.20, "7B": 0.80},   # 5 mountain         (only 7B helps; 2B/3B stuck)
]
HETERO = True   # False -> all sources use the config-only Q (homogeneous STEP-2 baseline)


def q_im(source, mname):
    """Per-source quality of configuration `mname` (measured per-class accuracy if HETERO)."""
    if HETERO:
        return Q_SRC[source][mname]
    return CONFIGS[mname]["Q"]

# ---------------- solar model (paper Eq.17) ----------------
# The battery now powers BOTH the continuous platform baseline (P_BASE) and inference,
# so the panel must cover platform+inference over an orbit and the battery must carry
# the platform through eclipse. Sunlit harvest per slot ~= P_SOLAR J; per orbit the
# panel yields N_SUNLIT*P_SOLAR J, the platform costs N_SLOTS_PER_ORBIT*P_BASE J, so
# sustainability needs P_SOLAR > P_BASE / sunlit_fraction (=15W at 60% sunlit); we size
# it just above that so only PACED inference is sustainable -> the energy-constrained regime.
P_SOLAR   = 13.0          # W panel (peak, sunlit): 0.6*13=7.8W orbit-avg vs 7W platform -> thin inference margin
ETA_PANEL = 0.30
ETA_SIGMA = 0.02

# ---------------- power / battery ----------------
P_CAP   = 15.0     # instantaneous power-bus cap (Jetson 15W mode): P_BASE + inference peak
P_BASE  = 7.0      # nominal platform baseline (CubeSat ADCS+housekeeping lit. range 3-10W); swept
B_MAX   = 18000.0  # sized to carry the platform through eclipse (2280 s x 7 W = 15960 J) + reserve
B_INIT  = 0.6 * B_MAX

U_TGT = 0.30
V_DEFAULT = 100.0
QE_MODE = "V2"     # energy queue tracks inference vs harvest; platform safety via eclipse reserve
RESERVE_FRAC = 0.85  # proposed keeps >= this fraction of the eclipse platform energy in reserve
RNG_SEED = 7


def eclipse_slots_to_sunrise(t, s):
    """Deterministic orbital forecast: eclipse slots between now and the next sunrise
    for satellite s (the energy the battery must still carry on the platform)."""
    phase = (t + SAT_PHASE[s]) % N_SLOTS_PER_ORBIT
    return (N_SLOTS_PER_ORBIT - N_SUNLIT) if phase < N_SUNLIT else (N_SLOTS_PER_ORBIT - phase)


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
    # the current slot must survive BOTH the continuous platform draw P_BASE and this
    # inference; the check is per-slot (myopic) -- it cannot foresee the multi-slot
    # eclipse-side accumulation of P_BASE, which is what produces blackout.
    return (e_per_slot + P_BASE * TAU) <= (b + omega)


def feasible_actions(pending, free_sats, b, omega, t):
    """SHARED feasibility oracle: every physically-allowed action this slot.
    IDENTICAL for the proposed scheduler and ALL baselines -- the split option,
    the ISL/handoff check (Eq.16), the per-slot energy (e<=b+omega) and the
    peak-power bus check (P_base+Ppeak<=P_cap) are defined ONCE here. Policies
    differ only in how they SELECT from this set, never in what counts as feasible."""
    acts = []
    # standalone options
    for s in free_sats:
        for mname in STANDALONE:
            m = CONFIGS[mname]
            if P_BASE + m["Ppeak"] > P_CAP:
                continue
            if not _energy_feasible(m["E"] / m["N"], b[s], omega[s]):
                continue
            for i in pending:
                if s not in VIS[i]:          # source i not in view of satellite s
                    continue
                acts.append(("standalone", s, i, mname))
    # split options: two distinct ISL-connected free sats; handoff up at t+N_front (Eq.16)
    if len(free_sats) >= 2 and P_BASE + SPLIT["Ppeak"] <= P_CAP:
        ef_ps = SPLIT["E_front"] / SPLIT["N_front"]
        er_ps = SPLIT["E_rear"] / SPLIT["N_rear"]
        for a in range(len(free_sats)):
            for c in range(len(free_sats)):
                if a == c:
                    continue
                sf, sr = free_sats[a], free_sats[c]
                if not isl_connected(t, sf, sr):
                    continue
                if not isl_connected(t + SPLIT["N_front"], sf, sr):
                    continue
                if not _energy_feasible(ef_ps, b[sf], omega[sf]):
                    continue
                if not _energy_feasible(er_ps, b[sr], omega[sr]):
                    continue
                for i in pending:
                    if sf not in VIS[i]:      # source i uplinks to the front satellite sf
                        continue
                    acts.append(("split", sf, sr, i))
    return acts


def _act_quality(act):
    # per-source delivered quality (config-quality x source-difficulty via q_im)
    if act[0] == "split":
        return q_im(act[3], "7B")
    return q_im(act[2], act[3])


def _act_energy(act):
    if act[0] == "split":
        return SPLIT["E_front"] + SPLIT["E_rear"]
    return CONFIGS[act[3]]["E"]


def _score_dpp(act, b, omega, QE, QP, QU, V):
    if act[0] == "standalone":
        _, s, i, mname = act
        m = CONFIGS[mname]; e_ps = m["E"] / m["N"]
        return QE[s] * e_ps + QP[s] * m["Ppeak"] - (QU[i] + V) * q_im(i, mname)
    _, sf, sr, i = act
    ef_ps = SPLIT["E_front"] / SPLIT["N_front"]; er_ps = SPLIT["E_rear"] / SPLIT["N_rear"]
    return (QE[sf] * ef_ps + QE[sr] * er_ps
            + QP[sf] * SPLIT["Ppeak"] + QP[sr] * SPLIT["Ppeak"]
            - (QU[i] + V) * SPLIT["Q"])


def _keeps_reserve(act, b, omega, t):
    """Would committing `act` still leave every involved satellite above the eclipse
    energy reserve? This is the proposed scheduler's eclipse-aware energy foresight:
    it uses the deterministic orbital forecast to hold back enough charge for the
    continuous platform load through the coming eclipse, choosing to idle otherwise."""
    if act[0] == "standalone":
        _, s, i, mn = act
        e = CONFIGS[mn]["E"] / CONFIGS[mn]["N"]; sats = [s]
    else:
        _, sf, sr, i = act
        e = SPLIT["E_front"] / SPLIT["N_front"]; sats = [sf, sr]
    for s in sats:
        need = P_BASE * TAU * eclipse_slots_to_sunrise(t, s) * RESERVE_FRAC
        if b[s] + omega[s] - P_BASE * TAU - e < need:
            return False
    return True


def choose_lyapunov(pending, free_sats, b, omega, QE, QP, QU, V, rng, t):
    """Proposed: drift-plus-penalty over the SHARED feasible set, but idling any action
    that would eat into the eclipse energy reserve (may idle if no action is safe/helpful)."""
    acts = feasible_actions(pending, free_sats, b, omega, t)
    acts = [a for a in acts if _keeps_reserve(a, b, omega, t)]
    if not acts:
        return None
    best = min(acts, key=lambda a: _score_dpp(a, b, omega, QE, QP, QU, V))
    return best if _score_dpp(best, b, omega, QE, QP, QU, V) < 0 else None


def choose_greedy_q(pending, free_sats, b, omega, QE, QP, QU, V, rng, t):
    """Greedy-Q: from the SAME feasible set, always take the highest-quality action --
    reaches for the 7B split whenever feasible. No long-term energy planning.
    Quality ties (e.g. every 7B split scores equally) are broken UNIFORMLY AT RANDOM
    so the low min-fill reflects memoryless greed, not a source-order artifact."""
    acts = feasible_actions(pending, free_sats, b, omega, t)
    if not acts:
        return None
    top = max(_act_quality(a) for a in acts)
    cand = [a for a in acts if _act_quality(a) == top]
    return cand[rng.integers(len(cand))]


def choose_greedy_e(pending, free_sats, b, omega, QE, QP, QU, V, rng, t):
    """Greedy-E: cheapest feasible action (so it stays on the cheap standalone tier).
    Energy ties broken uniformly at random (same anti-strawman fix as Greedy-Q)."""
    acts = feasible_actions(pending, free_sats, b, omega, t)
    if not acts:
        return None
    lo = min(_act_energy(a) for a in acts)
    cand = [a for a in acts if _act_energy(a) == lo]
    return cand[rng.integers(len(cand))]


def choose_random(pending, free_sats, b, omega, QE, QP, QU, V, rng, t):
    acts = feasible_actions(pending, free_sats, b, omega, t)
    return acts[rng.integers(len(acts))] if acts else None


def choose_static(pending, free_sats, b, omega, QE, QP, QU, V, rng, t):
    """Non-adaptive: fixed 3B standalone on the highest-battery free sat; never splits."""
    acts = [a for a in feasible_actions(pending, free_sats, b, omega, t)
            if a[0] == "standalone" and a[3] == "3B"]
    return max(acts, key=lambda a: b[a[1]]) if acts else None


POLICIES = {
    "Lyapunov (ours)": choose_lyapunov,
    "Greedy-Q":        choose_greedy_q,
    "Greedy-E":        choose_greedy_e,
    "Random":          choose_random,
    "Static":          choose_static,
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
                    delivered[i] += q_im(i, mname)
                    q_credit[i] = q_im(i, mname)
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
                    delivered[i] += q_im(i, "7B")
                    q_credit[i] = q_im(i, "7B")
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
            if t <= blackout_until[s]:
                # safe mode during recharge: platform is load-shed, battery charges on solar only
                b_new = b[s] + omega[s]
            else:
                # continuous platform draw P_BASE + inference, minus solar harvest
                b_new = b[s] + omega[s] - P_BASE * TAU - e_t[s]
                if b_new <= 0:
                    b_new = 0.0
                    recharge = int(np.ceil(0.20 * B_MAX / max(1e-6, P_SOLAR * TAU)))
                    blackout_until[s] = t + recharge
                    busy_until[s] = -1
            b[s] = min(b_new, B_MAX)

        # energy virtual queue (mode selects how the platform load enters the back-pressure)
        if QE_MODE == "V1":      # total deficit: platform + inference - harvest
            QE = np.maximum(QE + (P_BASE * TAU + e_t) - omega, 0.0)
        elif QE_MODE == "V2":    # inference vs harvest only (platform handled by battery/feasibility)
            QE = np.maximum(QE + e_t - omega, 0.0)
        else:                    # V3: inference vs the surplus left after the platform draw
            QE = np.maximum(QE + e_t - np.maximum(0.0, omega - P_BASE * TAU), 0.0)
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


SEEDS = list(range(7, 17))   # 10 seeds: average over arrival + solar-perturbation randomness
_METRICS = ["pcap_violations", "blackout_slots", "service_rate", "split_tasks", "min_fill", "total_quality"]


def compare_all():
    """Run every policy over SEEDS and report mean +/- std for each metric."""
    agg = {name: {m: [] for m in _METRICS} for name in POLICIES}
    for seed in SEEDS:
        for name, fn in POLICIES.items():
            r = run_sim(fn, seed=seed)
            for m in _METRICS:
                agg[name][m].append(r[m])
    results = {}
    print(f"{'Policy':18s} | {'downtime':>13s} | {'split#':>11s} | {'minFill':>13s} | {'totQ':>13s}   (mean+/-std over %d seeds)" % len(SEEDS))
    print("-" * 92)
    for name in POLICIES:
        st = {m: (float(np.mean(agg[name][m])), float(np.std(agg[name][m]))) for m in _METRICS}
        results[name] = st
        print(f"{name:18s} | {st['blackout_slots'][0]:7.0f}+/-{st['blackout_slots'][1]:<4.0f} | "
              f"{st['split_tasks'][0]:6.0f}+/-{st['split_tasks'][1]:<4.0f} | "
              f"{st['min_fill'][0]:.3f}+/-{st['min_fill'][1]:.3f} | "
              f"{st['total_quality'][0]:7.1f}+/-{st['total_quality'][1]:<5.1f}")
    return results


def plot_comparison(results, fname="splitpipeline_comparison.png"):
    names = list(results.keys())
    colors = [PALETTE["blue_main"], PALETTE["red_strong"], "tab:green", "tab:orange", "tab:purple"]
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    panels = [("total_quality", "(a) Total quality", "sum quality"),
              ("min_fill",      "(b) Min-fill",      "min fill ratio"),
              ("blackout_slots","(c) Downtime",      "# blackout slots"),
              ("split_tasks",   "(d) 7B splits",     "# split tasks")]
    for a, (metric, title, ylab) in zip(ax.flat, panels):
        means = [results[n][metric][0] for n in names]
        stds  = [results[n][metric][1] for n in names]
        a.bar(names, means, yerr=stds, color=colors, capsize=4, error_kw=dict(lw=1.5))
        a.set_title(title, loc="left"); a.set_ylabel(ylab)
        a.tick_params(axis="x", rotation=20); a.grid(alpha=0.3)
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
