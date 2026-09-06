#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EXP-1 : Offline MILP optimality-gap benchmark  (paper Theorem 2, O(1/V) gap)
============================================================================

Theorem 2 claims the online drift-plus-penalty scheduler is within O(1/V) of the
OFFLINE optimum of the max-min quality problem. Earlier experiments only showed
the *trend* of min-quality vs V; they never computed the actual optimum, so the
gap was unquantified. This script closes that hole.

Procedure (research-loop CHEAPEN: smallest faithful instance):
  - A small constellation (S sats, I sources) over a short, time-scaled orbit so a
    Mixed-Integer Linear Program is tractable and solvable to optimality with CBC.
  - (P_off) : solve the offline max-min total delivered quality with full future
    knowledge -> U_star  (the benchmark optimum).
  - Run the SAME instance (same deterministic solar/eclipse trace, same horizon)
    under the online Lyapunov scheduler for a sweep of V -> realized min-quality.
  - Plot the optimality gap (U_star - online(V)) vs V; expect it to shrink ~ 1/V.

Calibration (Q, E, Ppeak, N) = REAL Jetson Orin NX / EuroSAT 224px / Q4 (Table I).
Standalone configs only (3B, 7B): the gap theorem concerns the online-vs-offline
optimum of the SAME model, which the standalone instance already exercises; the
split-8B option is excluded here purely to keep the MILP tractable.

Reproducible: fixed seed, deterministic solar; one command reproduces every number.
"""

import os, sys, csv
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import apply_house_style, savefig_pub, PALETTE
apply_house_style()

import pulp

# ----------------------------- instance ------------------------------
TAU            = 1.0
ORBIT          = 24          # slots per (scaled) orbit
SUNLIT         = 14          # ~60% sunlit
N_ORBITS       = 2
HORIZON        = N_ORBITS * ORBIT      # 48 slots
S_SATS         = 2
I_SRC          = 2
SAT_PHASE      = [0, ORBIT // 2]       # staggered eclipse
P_SOLAR        = 8.0                   # J harvested per sunlit slot
P_CAP          = 15.0
P_BASE  = 9.0      # platform baseline draw (sensing+comms+ADCS); shares the bus
B_MAX          = 150.0
B_INIT         = 100.0
U_TGT          = 0.50                  # per-slot quality demand (keeps sources hungry)
RNG_SEED       = 7

# measured standalone configs (Table I)
CFG = {
    "2B": dict(Q=0.398, E=3.93, P=4.15, N=2),
    "3B": dict(Q=0.464, E=5.78, P=4.51, N=2),
}
MODELS = list(CFG)
for m in CFG.values():
    m["e_ps"] = m["E"] / m["N"]        # energy per active slot


ETA_PANEL = 0.30
ETA_SIGMA = 0.05
N_INSTANCES = 12             # average the gap over randomized instances (not cherry-picked)


def make_instance(seed):
    """A randomized but reproducible instance: random eclipse phases, initial
    battery, and per-slot solar perturbation (paper Eq.17 noise). The SAME realized
    solar/battery trace is handed to both the offline MILP and the online scheduler,
    so the comparison is apples-to-apples."""
    rng = np.random.default_rng(seed)
    phase = [int(rng.integers(0, ORBIT)) for _ in range(S_SATS)]
    b_init = [float(rng.uniform(0.5, 0.8) * B_MAX) for _ in range(S_SATS)]
    solar = np.zeros((S_SATS, HORIZON))
    for s in range(S_SATS):
        for t in range(HORIZON):
            if (t + phase[s]) % ORBIT < SUNLIT:
                eff = max(0.0, ETA_PANEL + rng.normal(0.0, ETA_SIGMA))
                solar[s, t] = P_SOLAR * eff / ETA_PANEL * TAU
    return dict(solar=solar, b_init=b_init)


# ============================ offline MILP (P_off) ============================
def solve_offline(inst):
    solar, b_init = inst["solar"], inst["b_init"]
    prob = pulp.LpProblem("P_off", pulp.LpMaximize)
    U = pulp.LpVariable("U", lowBound=0)

    # x[s,t,m,i] = sat s starts config m at slot t serving source i (must finish in horizon)
    x = {}
    for s in range(S_SATS):
        for t in range(HORIZON):
            for m in MODELS:
                if t + CFG[m]["N"] - 1 >= HORIZON:
                    continue
                for i in range(I_SRC):
                    x[s, t, m, i] = pulp.LpVariable(f"x_{s}_{t}_{m}_{i}", cat="Binary")

    b = {(s, t): pulp.LpVariable(f"b_{s}_{t}", lowBound=0, upBound=B_MAX)
         for s in range(S_SATS) for t in range(HORIZON + 1)}
    om = {(s, t): pulp.LpVariable(f"om_{s}_{t}", lowBound=0, upBound=float(solar[s, t]))
          for s in range(S_SATS) for t in range(HORIZON)}

    def active_starts(s, t):
        # (m,i,t') such that an exec started at t' is running during slot t
        out = []
        for m in MODELS:
            for tp in range(max(0, t - CFG[m]["N"] + 1), t + 1):
                for i in range(I_SRC):
                    if (s, tp, m, i) in x:
                        out.append((m, i, tp))
        return out

    for s in range(S_SATS):
        prob += b[s, 0] == b_init[s]
        for t in range(HORIZON):
            occ = active_starts(s, t)
            # at most one execution active per satellite per slot
            prob += pulp.lpSum(x[s, tp, m, i] for (m, i, tp) in occ) <= 1
            # energy draw and battery dynamics (solar may be curtailed via om<=avail)
            e_st = pulp.lpSum(x[s, tp, m, i] * CFG[m]["e_ps"] for (m, i, tp) in occ)
            prob += b[s, t + 1] == b[s, t] + om[s, t] - e_st
            # per-slot peak-power cap
            prob += pulp.lpSum(x[s, tp, m, i] * CFG[m]["P"] for (m, i, tp) in occ) <= P_CAP - P_BASE

    # max-min: U <= total delivered quality to every source
    for i in range(I_SRC):
        prob += U <= pulp.lpSum(x[s, t, m, i] * CFG[m]["Q"]
                                for (s, t, m, ii) in x if ii == i)
    prob += U

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    status = pulp.LpStatus[prob.status]
    return float(pulp.value(U)), status


# ============================ online Lyapunov scheduler ============================
def run_online(V, inst):
    solar, b_init = inst["solar"], inst["b_init"]
    b = np.array(b_init, float)
    QE = np.zeros(S_SATS)               # energy virtual queue
    QP = np.zeros(S_SATS)               # power-cap virtual queue
    QU = np.zeros(I_SRC)                # quality-deficit virtual queue
    busy_until = np.full(S_SATS, -1)
    run_e = np.zeros(S_SATS)
    run_p = np.zeros(S_SATS)
    delivered = np.zeros(I_SRC)
    pcap_viol = 0

    for t in range(HORIZON):
        om = np.array([min(solar[s, t], B_MAX - b[s]) for s in range(S_SATS)])
        free = [s for s in range(S_SATS) if t > busy_until[s]]

        # drift-plus-penalty: pick (sat, config, source) minimizing score, if < 0
        best_score, best = None, None
        for s in free:
            for m in MODELS:
                c = CFG[m]
                if P_BASE + c["P"] > P_CAP:
                    continue
                if c["e_ps"] > b[s] + om[s]:        # battery feasibility
                    continue
                for i in range(I_SRC):
                    score = QE[s]*c["e_ps"] + QP[s]*c["P"] - (QU[i] + V)*c["Q"]
                    if best_score is None or score < best_score:
                        best_score, best = score, (s, m, i)

        q_credit = np.zeros(I_SRC)
        if best is not None and best_score < 0:
            s, m, i = best
            c = CFG[m]
            busy_until[s] = t + c["N"] - 1
            run_e[s] = c["e_ps"]
            run_p[s] = c["P"]
            delivered[i] += c["Q"]
            q_credit[i] = c["Q"]

        e_t = np.zeros(S_SATS); p_t = np.zeros(S_SATS)
        for s in range(S_SATS):
            if t <= busy_until[s]:
                e_t[s] = run_e[s]; p_t[s] = run_p[s]
            if P_BASE + p_t[s] > P_CAP + 1e-9:
                pcap_viol += 1
        for s in range(S_SATS):
            b[s] = min(max(0.0, b[s] + om[s] - e_t[s]), B_MAX)

        QE = np.maximum(QE + e_t - om, 0.0)
        QP = np.maximum(QP + (P_BASE + p_t) - P_CAP, 0.0)
        QU = np.maximum(QU + U_TGT - q_credit, 0.0)

    return dict(min_quality=float(delivered.min()),
                total_quality=float(delivered.sum()),
                pcap_viol=pcap_viol,
                avg_QE=float(QE.mean()))


def main():
    print("=" * 78)
    print("  EXP-1  Offline MILP optimality-gap benchmark")
    print("=" * 78)
    print(f"  instance: S={S_SATS} sats, I={I_SRC} sources, horizon={HORIZON} slots "
          f"(orbit={ORBIT}, sunlit={SUNLIT})")
    print(f"  P_solar={P_SOLAR}J/slot  P_cap={P_CAP}W  B_max={B_MAX}J  B_init={B_INIT}J")
    print("-" * 78)

    Vs = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    # solve each instance offline once, then sweep V online on the SAME instance
    insts = [make_instance(k) for k in range(N_INSTANCES)]
    Ustars, statuses = [], []
    for inst in insts:
        u, st = solve_offline(inst)
        Ustars.append(u); statuses.append(st)
    Ustars = np.array(Ustars)
    n_opt = sum(s == "Optimal" for s in statuses)
    print(f"  solved {N_INSTANCES} offline MILP instances ({n_opt} proven Optimal); "
          f"mean U* = {Ustars.mean():.4f}")
    print("-" * 78)

    # normalized gap per instance: (U* - online_min) / U*, averaged over instances
    norm_gap = np.zeros((N_INSTANCES, len(Vs)))
    norm_q   = np.zeros((N_INSTANCES, len(Vs)))
    raw_viol = 0
    for k, inst in enumerate(insts):
        for j, V in enumerate(Vs):
            r = run_online(V, inst)
            raw_viol += r["pcap_viol"]
            norm_gap[k, j] = (Ustars[k] - r["min_quality"]) / max(1e-9, Ustars[k])
            norm_q[k, j]   = r["min_quality"] / max(1e-9, Ustars[k])

    mean_gap = norm_gap.mean(0); std_gap = norm_gap.std(0)
    mean_q   = norm_q.mean(0);   std_q   = norm_q.std(0)
    print(f"  total peak-power violations across all instances & V = {raw_viol}")
    print(f"  {'V':>6s} | {'mean norm gap':>13s} | {'mean online/U*':>14s}")
    for j, V in enumerate(Vs):
        print(f"  {V:6d} | {mean_gap[j]:13.4f} | {mean_q[j]:14.4f}")

    # ---- save raw numbers ----
    os.makedirs("data", exist_ok=True)
    with open("data/results_milp_gap.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_instances", N_INSTANCES, "n_optimal", n_opt, "mean_U_star", Ustars.mean()])
        w.writerow(["V", "mean_norm_gap", "std_norm_gap", "mean_online_over_Ustar", "std"])
        for j, V in enumerate(Vs):
            w.writerow([V, mean_gap[j], std_gap[j], mean_q[j], std_q[j]])
    print("  saved data/results_milp_gap.csv")

    # ---- figure: gap vs V (log-x), with 1/V reference ----
    Varr = np.array(Vs, float)
    gaps = mean_gap
    minq = mean_q

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    # (a) online min-quality (normalized by U*) approaching the offline optimum
    ax[0].axhline(1.0, color=PALETTE["neutral"], ls="--", lw=2, label=r"offline optimum $U^\star$")
    ax[0].plot(Varr, minq, "o-", color=PALETTE["blue_main"], lw=2.5, ms=7, label="online (ours)")
    ax[0].fill_between(Varr, minq - std_q, np.minimum(1.0, minq + std_q),
                       color=PALETTE["blue_main"], alpha=0.15)
    ax[0].set_xscale("log")
    ax[0].set_xlabel(r"control parameter $V$")
    ax[0].set_ylabel(r"online min-quality $/\ U^\star$")
    ax[0].set_title("(a)", loc="left")
    ax[0].legend(loc="lower right")
    # (b) normalized gap vs V with c/V reference fitted to the largest-V point
    ax[1].plot(Varr, gaps, "o-", color=PALETTE["blue_main"], lw=2.5, ms=7,
               label=r"normalized gap")
    j0 = int(np.argmin(gaps))                 # anchor 1/V guide to the best (min-gap) point
    c = gaps[j0] * Varr[j0]
    ax[1].plot(Varr, c / Varr, "--", color=PALETTE["red_strong"], lw=2,
               label=r"$O(1/V)$ reference")
    ax[1].set_xscale("log")
    ax[1].set_ylim(0, float(gaps.max()) * 1.15)   # keep on data scale; 1/V guide visible in decay region
    ax[1].set_xlabel(r"control parameter $V$")
    ax[1].set_ylabel(r"$(U^\star-$online$)\,/\,U^\star$")
    ax[1].set_title("(b)", loc="left")
    ax[1].legend(loc="upper right")
    savefig_pub(fig, "milp_gap.pdf")
    fig.savefig("milp_gap.png", dpi=130)
    print("  saved milp_gap.pdf")
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
