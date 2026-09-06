#!/usr/bin/env python3
"""TMC Experiment 2 (MILP part): total-quality gap at a fixed feasible target.

Small instances (2 sats, 2 sources, 48 slots), 12 randomized solar instances.
For each instance: (1) solve the max-min optimum U* (epigraph form); (2) fix the
target at U_tgt = 0.9*U* and solve the clairvoyant MILP maximizing TOTAL quality
subject to the per-source floor U_tgt*sum(g_i) <= sum(x*Q) -- this is q_Sigma*
of the revised theorem; (3) run the online Lyapunov scheduler on the same
instance (same target) and measure its total quality; (4) gap =
(q_Sigma* - online)/q_Sigma* vs 1/V, fitted with a line.

Outputs:
  data/results_milp_gap2.csv
  milp_gap.pdf  (paper fig:gap -- total-quality gap vs 1/V, fixed target)
"""
import os, sys, csv
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import apply_house_style, savefig_pub, PALETTE  # noqa: E402

import pulp

apply_house_style()

# ----------------------------- instance ------------------------------
TAU = 1.0
ORBIT = 24
SUNLIT = 14
N_ORBITS = 2
HORIZON = N_ORBITS * ORBIT           # 48 slots
S_SATS = 2
I_SRC = 2
P_SOLAR = 8.0
P_CAP = 15.0
P_BASE = 9.0
B_MAX = 150.0
B_INIT = 100.0
ARRIVAL_LAMBDA = 0.5                 # per-source arrival probability per slot
ETA_PANEL = 0.30
ETA_SIGMA = 0.05
N_INSTANCES = 12
TARGET_FRAC = 0.9                    # target = 90% of the instance's max-min optimum
VS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]

CFG = {"2B": dict(Q=0.398, E=3.93, P=4.15, N=2),
       "3B": dict(Q=0.464, E=5.78, P=4.51, N=2)}
MODELS = list(CFG)
for m in CFG.values():
    m["e_ps"] = m["E"] / m["N"]


def make_instance(seed):
    rng = np.random.default_rng(seed)
    phase = [int(rng.integers(0, ORBIT)) for _ in range(S_SATS)]
    b_init = [float(rng.uniform(0.5, 0.8) * B_MAX) for _ in range(S_SATS)]
    solar = np.zeros((S_SATS, HORIZON))
    for s in range(S_SATS):
        for t in range(HORIZON):
            if (t + phase[s]) % ORBIT < SUNLIT:
                eff = max(0.0, ETA_PANEL + rng.normal(0.0, ETA_SIGMA))
                solar[s, t] = P_SOLAR * eff / ETA_PANEL * TAU
    arrivals = (rng.random((I_SRC, HORIZON)) < ARRIVAL_LAMBDA).astype(int)
    return dict(solar=solar, b_init=b_init, arrivals=arrivals)


# ============================ offline MILP ============================
def _build(inst, U):
    """Maximize total delivered quality subject to the per-source floor
    U * sum_t g_i(t) <= sum x*Q (fixed target U), occupancy, energy, power."""
    solar, b_init, arrivals = inst["solar"], inst["b_init"], inst["arrivals"]
    prob = pulp.LpProblem("P_off_fixed", pulp.LpMaximize)
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
            prob += pulp.lpSum(x[s, tp, m, i] for (m, i, tp) in occ) <= 1
            e_st = pulp.lpSum(x[s, tp, m, i] * CFG[m]["e_ps"] for (m, i, tp) in occ)
            prob += b[s, t + 1] == b[s, t] + om[s, t] - e_st
            prob += pulp.lpSum(x[s, tp, m, i] * CFG[m]["P"] for (m, i, tp) in occ) <= P_CAP - P_BASE
    # per-source quality floor at the fixed target
    for i in range(I_SRC):
        prob += U * float(arrivals[i].sum()) <= pulp.lpSum(
            x[s, t, m, i] * CFG[m]["Q"] for (s, t, m, ii) in x if ii == i)
    # objective: total delivered quality
    prob += pulp.lpSum(x[s, t, m, i] * CFG[m]["Q"] for (s, t, m, i) in x)
    return prob


def solve_maxmin(inst):
    """Max-min optimum U* (epigraph form, like the paper's P_off)."""
    solar, b_init = inst["solar"], inst["b_init"]
    prob = pulp.LpProblem("P_off_U", pulp.LpMaximize)
    U = pulp.LpVariable("U", lowBound=0)
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
    for s in range(S_SATS):
        prob += b[s, 0] == b_init[s]
        for t in range(HORIZON):
            occ = [(m, i, tp) for m in MODELS
                   for tp in range(max(0, t - CFG[m]["N"] + 1), t + 1)
                   for i in range(I_SRC) if (s, tp, m, i) in x]
            prob += pulp.lpSum(x[s, tp, m, i] for (m, i, tp) in occ) <= 1
            e_st = pulp.lpSum(x[s, tp, m, i] * CFG[m]["e_ps"] for (m, i, tp) in occ)
            prob += b[s, t + 1] == b[s, t] + om[s, t] - e_st
            prob += pulp.lpSum(x[s, tp, m, i] * CFG[m]["P"] for (m, i, tp) in occ) <= P_CAP - P_BASE
    for i in range(I_SRC):
        prob += U * float(inst["arrivals"][i].sum()) <= pulp.lpSum(
            x[s, t, m, i] * CFG[m]["Q"] for (s, t, m, ii) in x if ii == i)
    prob += U
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    return float(pulp.value(U)), pulp.LpStatus[prob.status]


def solve_fixed_total(inst, U):
    prob = _build(inst, U)
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    status = pulp.LpStatus[prob.status]
    tot = float(pulp.value(prob.objective))
    return tot, status


# ============================ online Lyapunov ============================
def run_online(V, inst, utgt):
    solar, b_init, arrivals = inst["solar"], inst["b_init"], inst["arrivals"]
    b = np.array(b_init, float)
    QE = np.zeros(S_SATS); QP = np.zeros(S_SATS); QU = np.zeros(I_SRC)
    busy_until = np.full(S_SATS, -1)
    run_e = np.zeros(S_SATS); run_p = np.zeros(S_SATS)
    delivered = np.zeros(I_SRC)
    pcap_viol = 0
    for t in range(HORIZON):
        om = np.array([min(solar[s, t], B_MAX - b[s]) for s in range(S_SATS)])
        free = [s for s in range(S_SATS) if t > busy_until[s]]
        best_score, best = None, None
        for s in free:
            for m in MODELS:
                c = CFG[m]
                if P_BASE + c["P"] > P_CAP or c["e_ps"] > b[s] + om[s]:
                    continue
                for i in range(I_SRC):
                    sc = QE[s] * c["e_ps"] + QP[s] * c["P"] - (QU[i] + V) * c["Q"]
                    if best_score is None or sc < best_score:
                        best_score, best = sc, (s, m, i)
        q_credit = np.zeros(I_SRC)
        if best is not None and best_score < 0:
            s, m, i = best
            c = CFG[m]
            busy_until[s] = t + c["N"] - 1
            run_e[s] = c["e_ps"]; run_p[s] = c["P"]
            delivered[i] += c["Q"]; q_credit[i] = c["Q"]
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
        QU = np.maximum(QU + arrivals[:, t] * utgt - q_credit, 0.0)
    return dict(total_quality=float(delivered.sum()), pcap_viol=pcap_viol)


def main():
    print("=" * 78)
    print("  EXP2-MILP  total-quality gap at a fixed feasible target")
    print("=" * 78)
    insts = [make_instance(k) for k in range(N_INSTANCES)]
    rows = []
    for k, inst in enumerate(insts):
        ustar, st = solve_maxmin(inst)
        utgt = TARGET_FRAC * ustar
        tot_star, st2 = solve_fixed_total(inst, utgt)
        rows.append((ustar, st, utgt, tot_star, st2, inst))
        print(f"  inst {k:2d}: U*={ustar:.3f} ({st})  target={utgt:.3f}  "
              f"q*_Sigma={tot_star:.2f} ({st2})", flush=True)
    ok = [r for r in rows if r[1] == "Optimal" and r[4] == "Optimal"]
    print(f"  optimal instances: {len(ok)}/{N_INSTANCES}", flush=True)

    gaps = np.full((len(ok), len(VS)), np.nan)
    for k, (ustar, st, utgt, tot_star, st2, inst) in enumerate(ok):
        for j, V in enumerate(VS):
            r = run_online(V, inst, utgt)
            gaps[k, j] = (tot_star - r["total_quality"]) / max(1e-9, tot_star)
        print(f"  online sweeps inst {k}: lastV gap={gaps[k,-1]:.3f}", flush=True)

    mean_gap = np.nanmean(gaps, axis=0); std_gap = np.nanstd(gaps, axis=0)
    Varr = np.array(VS, float)
    # theoretical form gap <= B/V: fit gap = a*(1/V) through the origin
    iv = 1.0 / Varr
    a = float((iv @ mean_gap) / (iv @ iv))
    pred = a * iv
    resid = mean_gap - pred
    dof = len(iv) - 1
    s2 = float(resid @ resid) / dof if dof > 0 else 0.0
    se_a = float(np.sqrt(s2 / (iv @ iv)))
    print(f"  fit (through origin): gap = {a:.3f}/V  (95% CI "
          f"[{a-1.96*se_a:.3f}, {a+1.96*se_a:.3f}])", flush=True)

    with open(os.path.join(HERE, "..", "data", "results_milp_gap2.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_instances", N_INSTANCES, "n_optimal", len(ok), "target_frac", TARGET_FRAC])
        w.writerow(["V", "mean_gap", "std_gap", "fit_a"])
        for j, V in enumerate(VS):
            w.writerow([V, mean_gap[j], std_gap[j], a])
    print("  wrote data/results_milp_gap2.csv", flush=True)

    # ---- figure: gap vs 1/V with fitted line ----
    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    ax.errorbar(Varr, mean_gap, yerr=std_gap, fmt="o", color=PALETTE["blue_main"],
                ms=8, lw=2.2, capsize=4, label=r"total-quality gap (mean $\pm$ std)")
    ivx = np.linspace(iv.min(), iv.max(), 100)
    ax.plot(1.0 / ivx, a * ivx, "--", color=PALETTE["red_strong"],
            lw=2.4, label=fr"fit $a/V$, $a={a:.2f}$")
    ax.set_xscale("log")
    ax.set_xlabel(r"control weight $V$")
    ax.set_ylabel(r"$(q_\Sigma^* - q_\Sigma^{\mathrm{ALG}})/q_\Sigma^*$")
    ax.grid(alpha=0.3); ax.legend(loc="upper right", fontsize=15)
    savefig_pub(fig, os.path.join(HERE, "..", "milp_gap.pdf"))
    print("  wrote milp_gap.pdf", flush=True)


HERE = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    main()
