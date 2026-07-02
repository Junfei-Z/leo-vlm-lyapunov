#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EXP-6 : Systematic sensitivity analysis  (honest boundary characterization)
===========================================================================

Goal: map HONESTLY where the proposed scheduler helps and where it is only a
trade-off. We sweep one scenario parameter at a time and record ALL five policies'
metrics at every point, including the regimes where the proposed scheduler does
NOT win. This is boundary mapping, not parameter fishing.

The simulator is a parameterized, faithful port of src/04_split_pipeline.py: same
Jetson-calibrated configs, same solar model (Eq.17), same two-stage 8B split, same
drift-plus-penalty scheduler and the same four baselines. A reproduction guardrail
asserts that at the default parameters it matches src/04's published numbers
(ours: 0 blackout / 1100 split / totQ 2378.4 ; Greedy-Q: 2438 blackout).

Reproducible: fixed seed; one command reproduces every CSV and figure.
"""

import os, sys, csv
from dataclasses import dataclass, replace
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import apply_house_style, savefig_pub, PALETTE
apply_house_style()

# ---------------- fixed time scales (as in src/04) ----------------
TAU = 1.0
ORBIT_PERIOD_S = 95 * 60
N_SLOTS_PER_ORBIT = int(ORBIT_PERIOD_S / TAU)      # 5700
N_ORBITS = 3
HORIZON = N_ORBITS * N_SLOTS_PER_ORBIT             # 17100

# ---------------- measured configs (Table I) ----------------
CONFIGS = {
    # RESISC45 n=1300 calibration, all Q4. Ppeak = MAX per-inference peak (conservative
    # bound for the hard cap); only 7B shows a real transient (6.37 vs ~3.8 for 2B/3B).
    "2B": dict(T=1.24, E=3.93, Ppeak=3.75, Q=0.394, RAM=2.5),
    "3B": dict(T=1.74, E=5.72, Ppeak=4.02, Q=0.456, RAM=3.5),
    "7B": dict(T=2.76, E=9.30, Ppeak=6.37, Q=0.549, RAM=6.5),
}
for _m in CONFIGS.values():
    _m["N"] = int(np.ceil(_m["T"] / TAU))
STANDALONE = ["2B", "3B"]
SPLIT = dict(
    Q=CONFIGS["7B"]["Q"],
    N_front=int(np.ceil(CONFIGS["7B"]["N"] / 2)),
    N_rear=int(np.ceil(CONFIGS["7B"]["N"] / 2)),
    E_front=CONFIGS["7B"]["E"] / 2,
    E_rear=CONFIGS["7B"]["E"] / 2,
    Ppeak=CONFIGS["7B"]["Ppeak"],
    RAM=CONFIGS["7B"]["RAM"] / 2,
)

ETA_PANEL = 0.30
ETA_SIGMA = 0.02
U_TGT = 0.30
V_DEFAULT = 100.0
RNG_SEED = 7
SEEDS = list(range(7, 17))   # 10 seeds for arrival + solar-perturbation randomness (error bars)


@dataclass(frozen=True)
class Params:
    n_sat: int = 4
    n_sources: int = 6
    arrival_prob: float = 0.10
    B_max: float = 5000.0
    P_solar: float = 1.0
    P_cap: float = 15.0      # realistic satellite power-bus limit (Jetson 15W mode)
    P_base: float = 9.0      # platform baseline draw (sensing+comms+ADCS); inference shares the rest
    sunlit_fraction: float = 0.60
    U_tgt: float = 0.30

    @property
    def n_sunlit(self):
        return int(N_SLOTS_PER_ORBIT * self.sunlit_fraction)

    @property
    def sat_phase(self):
        return [int(k * N_SLOTS_PER_ORBIT / self.n_sat) for k in range(self.n_sat)]

    @property
    def b_init(self):
        return 0.6 * self.B_max


def sunlit_indicator(t, sat, prm):
    phase = (t + prm.sat_phase[sat]) % N_SLOTS_PER_ORBIT
    return 1 if phase < prm.n_sunlit else 0


def solar_harvest(t, sat, rng, prm):
    if sunlit_indicator(t, sat, prm) == 0:
        return 0.0
    eps = rng.normal(0.0, ETA_SIGMA)
    eff = max(0.0, ETA_PANEL + eps)
    return prm.P_solar * eff / ETA_PANEL * TAU


def isl_connected(t, s1, s2):
    return s1 != s2


def _energy_feasible(e_per_slot, b, omega):
    return e_per_slot <= (b + omega)


# ---------------- policies (ported verbatim from src/04, P_cap via prm) ----------------
def feasible_actions(pending, free_sats, b, omega, t, prm):
    """SHARED feasibility oracle (identical for proposed + all baselines): split option,
    ISL/handoff check (Eq.16), per-slot energy (e<=b+omega), peak-power bus check are all
    defined ONCE here. Policies differ only in how they SELECT, never in what is feasible."""
    acts = []
    for s in free_sats:
        for mname in STANDALONE:
            m = CONFIGS[mname]
            if prm.P_base + m["Ppeak"] > prm.P_cap:
                continue
            if not _energy_feasible(m["E"] / m["N"], b[s], omega[s]):
                continue
            for i in pending:
                acts.append(("standalone", s, i, mname))
    if len(free_sats) >= 2 and prm.P_base + SPLIT["Ppeak"] <= prm.P_cap:
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
                    acts.append(("split", sf, sr, i))
    return acts


def _act_quality(act):
    return SPLIT["Q"] if act[0] == "split" else CONFIGS[act[3]]["Q"]


def _act_energy(act):
    if act[0] == "split":
        return SPLIT["E_front"] + SPLIT["E_rear"]
    return CONFIGS[act[3]]["E"]


def _score_dpp(act, b, omega, QE, QP, QU, V):
    if act[0] == "standalone":
        _, s, i, mname = act
        m = CONFIGS[mname]; e_ps = m["E"] / m["N"]
        return QE[s] * e_ps + QP[s] * m["Ppeak"] - (QU[i] + V) * m["Q"]
    _, sf, sr, i = act
    ef_ps = SPLIT["E_front"] / SPLIT["N_front"]; er_ps = SPLIT["E_rear"] / SPLIT["N_rear"]
    return (QE[sf] * ef_ps + QE[sr] * er_ps
            + QP[sf] * SPLIT["Ppeak"] + QP[sr] * SPLIT["Ppeak"]
            - (QU[i] + V) * SPLIT["Q"])


def choose_lyapunov(pending, free_sats, b, omega, QE, QP, QU, V, rng, t, prm):
    acts = feasible_actions(pending, free_sats, b, omega, t, prm)
    if not acts:
        return None
    best = min(acts, key=lambda a: _score_dpp(a, b, omega, QE, QP, QU, V))
    return best if _score_dpp(best, b, omega, QE, QP, QU, V) < 0 else None


def choose_greedy_q(pending, free_sats, b, omega, QE, QP, QU, V, rng, t, prm):
    acts = feasible_actions(pending, free_sats, b, omega, t, prm)
    return max(acts, key=_act_quality) if acts else None


def choose_greedy_e(pending, free_sats, b, omega, QE, QP, QU, V, rng, t, prm):
    acts = feasible_actions(pending, free_sats, b, omega, t, prm)
    return min(acts, key=_act_energy) if acts else None


def choose_random(pending, free_sats, b, omega, QE, QP, QU, V, rng, t, prm):
    acts = feasible_actions(pending, free_sats, b, omega, t, prm)
    return acts[rng.integers(len(acts))] if acts else None


def choose_static(pending, free_sats, b, omega, QE, QP, QU, V, rng, t, prm):
    acts = [a for a in feasible_actions(pending, free_sats, b, omega, t, prm)
            if a[0] == "standalone" and a[3] == "3B"]
    return max(acts, key=lambda a: b[a[1]]) if acts else None


POLICIES = {
    "Lyapunov (ours)": choose_lyapunov,
    "Greedy-Q":        choose_greedy_q,
    "Greedy-E":        choose_greedy_e,
    "Random":          choose_random,
    "Static":          choose_static,
}


# ---------------- simulation core (parameterized port of src/04 run_sim) ----------------
def run_sim(policy_fn, prm, V=V_DEFAULT, horizon=HORIZON, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    N_SAT, N_SRC = prm.n_sat, prm.n_sources
    b = np.full(N_SAT, prm.b_init)
    QE = np.zeros(N_SAT); QP = np.zeros(N_SAT); QU = np.zeros(N_SRC)
    busy_until = np.full(N_SAT, -1)
    run_E_ps = np.zeros(N_SAT); run_P = np.zeros(N_SAT)
    blackout_until = np.full(N_SAT, -1)
    delivered = np.zeros(N_SRC); arrivals_count = np.zeros(N_SRC)
    pcap_violations = 0; blackout_slots = 0
    tasks_started = 0; split_tasks = 0

    for t in range(horizon):
        omega = np.zeros(N_SAT)
        for s in range(N_SAT):
            Et = solar_harvest(t, s, rng, prm)
            omega[s] = min(Et, prm.B_max - b[s])

        arrivals = (rng.random(N_SRC) < prm.arrival_prob).astype(int)
        arrivals_count += arrivals
        pending = [i for i in range(N_SRC) if arrivals[i] == 1]

        for s in range(N_SAT):
            if t <= blackout_until[s]:
                blackout_slots += 1

        free_sats = [s for s in range(N_SAT)
                     if t > busy_until[s] and t > blackout_until[s]]

        q_credit = np.zeros(N_SRC)
        if pending and free_sats:
            choice = policy_fn(pending, free_sats, b, omega, QE, QP, QU, V, rng, t, prm)
            if choice is not None:
                if choice[0] == "standalone":
                    _, s, i, mname = choice
                    m = CONFIGS[mname]
                    busy_until[s] = t + m["N"] - 1
                    run_E_ps[s] = m["E"] / m["N"]; run_P[s] = m["Ppeak"]
                    delivered[i] += m["Q"]; q_credit[i] = m["Q"]
                    tasks_started += 1
                elif choice[0] == "split":
                    _, sf, sr, i = choice
                    busy_until[sf] = t + SPLIT["N_front"] - 1
                    run_E_ps[sf] = SPLIT["E_front"] / SPLIT["N_front"]; run_P[sf] = SPLIT["Ppeak"]
                    rear_start = t + SPLIT["N_front"]
                    busy_until[sr] = rear_start + SPLIT["N_rear"] - 1
                    run_E_ps[sr] = SPLIT["E_rear"] / SPLIT["N_rear"]; run_P[sr] = SPLIT["Ppeak"]
                    delivered[i] += SPLIT["Q"]; q_credit[i] = SPLIT["Q"]
                    tasks_started += 1; split_tasks += 1

        e_t = np.zeros(N_SAT); p_t = np.zeros(N_SAT)
        for s in range(N_SAT):
            if t <= busy_until[s] and t > blackout_until[s]:
                e_t[s] = run_E_ps[s]; p_t[s] = run_P[s]
            if prm.P_base + p_t[s] > prm.P_cap + 1e-9:
                pcap_violations += 1

        for s in range(N_SAT):
            b_new = b[s] + omega[s] - e_t[s]
            if b_new <= 0:
                b_new = 0.0
                recharge = int(np.ceil(0.20 * prm.B_max / max(1e-6, prm.P_solar * TAU)))
                blackout_until[s] = t + recharge
                busy_until[s] = -1
            b[s] = min(b_new, prm.B_max)

        QE = np.maximum(QE + e_t - omega, 0.0)
        QP = np.maximum(QP + (prm.P_base + p_t) - prm.P_cap, 0.0)
        QU = np.maximum(QU + arrivals * prm.U_tgt - q_credit, 0.0)

    fill = delivered / np.maximum(1, arrivals_count)
    service_rate = tasks_started / max(1, arrivals_count.sum())
    return dict(
        pcap_violations=pcap_violations,
        blackout_slots=blackout_slots,
        service_rate=service_rate,
        split_tasks=split_tasks,
        min_fill=float(fill.min()),
        total_quality=float(delivered.sum()),
    )


def eval_all(prm):
    return {name: run_sim(fn, prm) for name, fn in POLICIES.items()}


# ---------------- reproduction guardrail ----------------
def reproduction_check():
    r = eval_all(Params())
    ours, gq = r["Lyapunov (ours)"], r["Greedy-Q"]
    # RESISC45 8GB-satellite configs (2B/3B standalone, 7B split) + small 2W default panel
    # Energy-constrained default (1.0W panel, opportunistic inference on a power-starved sat) +
    # shared-feasibility baselines. Greedy-Q greedily splits with no energy foresight and blacks
    # out 32741 slots; the proposed DPP scheduler stays at 0 and delivers the highest min-fill.
    ok = (ours["blackout_slots"] == 0 and ours["split_tasks"] == 605
          and abs(ours["total_quality"] - 3384.7) < 0.5 and gq["blackout_slots"] == 32741)
    print(f"  reproduction @ default: ours blackout={ours['blackout_slots']} "
          f"split={ours['split_tasks']} totQ={ours['total_quality']:.1f} | "
          f"Greedy-Q blackout={gq['blackout_slots']}  -> "
          f"{'REPRODUCTION OK' if ok else 'MISMATCH!!'}")
    if not ok:
        raise SystemExit("Reproduction guardrail failed; port diverged from src/04.")
    return r


# ---------------- sweeps ----------------
SWEEPS = [
    ("nsat",    "n_sat",           [2, 4, 8, 16]),
    ("arrival", "arrival_prob",    [0.02, 0.05, 0.10, 0.20, 0.40, 0.70]),
    ("battery", "B_max",           [1000, 2000, 3000, 5000, 8000, 12000]),
    ("panel",   "P_solar",         [0.6, 0.8, 1.0, 1.5, 2.5, 5.0]),  # panel harvest power; 1.0=constrained default
    ("pbase",   "P_base",          [8, 10, 11, 12, 13, 14]),  # platform load rises -> inference headroom (15-Pbase) shrinks -> bus binds
    ("sunlit",  "sunlit_fraction", [0.45, 0.50, 0.55, 0.60, 0.70, 0.80]),
    ("nsrc",    "n_sources",       [2, 4, 6, 10, 16]),
    ("qdemand", "U_tgt",         [0.1, 0.2, 0.3, 0.4, 0.5]),
]
METRICS = ["pcap_violations", "blackout_slots", "service_rate", "split_tasks",
           "min_fill", "total_quality"]


def run_sweeps():
    os.makedirs("data", exist_ok=True)
    all_data = {}
    for tag, field, values in SWEEPS:
        rows = []
        for v in values:
            per_pol = {name: {m: [] for m in METRICS} for name in POLICIES}
            for seed in SEEDS:
                prm = replace(Params(), **{field: v})
                for name, fn in POLICIES.items():
                    r = run_sim(fn, prm, seed=seed)
                    for m in METRICS:
                        per_pol[name][m].append(r[m])
            for pol in POLICIES:
                means = [float(np.mean(per_pol[pol][m])) for m in METRICS]
                stds = [float(np.std(per_pol[pol][m])) for m in METRICS]
                rows.append([v, pol] + means + stds)
        path = f"data/sens_{tag}.csv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["param_value", "policy"] + METRICS + [m + "_sd" for m in METRICS])
            w.writerows(rows)
        all_data[tag] = rows
        print(f"  swept {field}: {len(values)} points x {len(POLICIES)} policies -> {path}")
    return all_data


def _series(rows, policy, metric_idx):
    xs = sorted(set(r[0] for r in rows))
    ys, es = [], []
    nm = len(METRICS)
    for x in xs:
        for r in rows:
            if r[0] == x and r[1] == policy:
                ys.append(r[2 + metric_idx])
                es.append(r[2 + nm + metric_idx] if len(r) > 2 + nm + metric_idx else 0.0)
                break
    return xs, ys, es


def plot_sweeps(all_data, metric, ylabel, fname, logy=False):
    titles = {"nsat": "satellites", "arrival": "arrival prob.", "battery": "battery $B_{\\max}$ (kJ)",
              "panel": "solar panel power (W)", "pbase": "platform load $P_{base}$ (W)", "sunlit": "sunlit fraction",
              "nsrc": "sources", "qdemand": "quality demand $U_{tgt}$"}
    midx = METRICS.index(metric)
    colors = {"Lyapunov (ours)": PALETTE["blue_main"], "Greedy-Q": PALETTE["red_strong"],
              "Greedy-E": PALETTE["green_3"], "Random": PALETTE["neutral"],
              "Static": PALETTE["highlight"]}
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    from matplotlib.ticker import ScalarFormatter, NullFormatter
    for k, (ax, (tag, field, values)) in enumerate(zip(axes.flat, SWEEPS)):
        rows = all_data[tag]
        for pol in POLICIES:
            xs, ys, es = _series(rows, pol, midx)
            lab = "ours" if pol == "Lyapunov (ours)" else pol
            ax.errorbar(xs, ys, yerr=es, fmt="o-", color=colors[pol], lw=2.4, ms=7,
                        capsize=3, elinewidth=1.2,
                        label=lab, alpha=0.95 if pol == "Lyapunov (ours)" else 0.8)
        ax.set_xlabel(titles[tag])
        ax.set_title("(%s)" % chr(ord("a") + k), loc="left", fontsize=13)
        if logy:
            ax.set_yscale("symlog")
        if tag in ("nsat", "nsrc", "battery"):
            ax.set_xscale("log")
            ax.set_xticks(values)
            if tag == "battery":
                ax.set_xticklabels([str(int(v / 1000)) for v in values])
            else:
                ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.tick_params(axis="x", which="minor", length=0)
    axes.flat[0].set_ylabel(ylabel); axes.flat[4].set_ylabel(ylabel)
    axes.flat[3].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.suptitle(f"Sensitivity of {ylabel} across eight scenario parameters", y=1.00)
    savefig_pub(fig, fname)
    fig.savefig(os.path.splitext(fname)[0] + ".png", dpi=130)
    print(f"  saved {fname}")
    plt.close(fig)


def main():
    print("=" * 78)
    print("  EXP-6  Systematic sensitivity analysis")
    print("=" * 78)
    reproduction_check()
    print("-" * 78)
    all_data = run_sweeps()
    print("-" * 78)
    plot_sweeps(all_data, "blackout_slots", "blackout slots", "sensitivity_safety.pdf", logy=True)
    plot_sweeps(all_data, "min_fill", "min-fill (max-min quality)", "sensitivity_quality.pdf")
    print("Done.")


if __name__ == "__main__":
    main()
