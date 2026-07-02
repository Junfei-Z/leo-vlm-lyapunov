#!/usr/bin/env python3
"""Idle decomposition on the LOCKED model (imports src/04): why does the proposed
scheduler idle? Each called decision slot is classified as
  act / reserve_blocked (eclipse-reserve guard of the selection rule)
  / score_idle (DPP score >= 0) / no_feasible.
The wrapper replicates choose_lyapunov EXACTLY (feasible set -> reserve filter ->
DPP score) and only counts; the trajectory is unchanged.

Verified findings this script reproduces (cited in the paper):
- n_sat 8 -> 16: the service drop traces to the eclipse-reserve guard
  (blocked slots 423 -> 1134 of ~7960), not to score idling.
- comfortable battery (B_max 18k -> 24k): the guard goes inactive (31 -> 0)
  and ALL conservatism is rate-based QE score idling — the limitation's
  root-cause attribution.
"""
import importlib.util, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("splitpipe", os.path.join(HERE, "04_split_pipeline.py"))
M = importlib.util.module_from_spec(spec); sys.modules["splitpipe"] = M; spec.loader.exec_module(M)

CNT = {}

def counting_policy(pending, free, b, omega, QE, QP, QU, V, rng, t):
    CNT["called"] += 1
    acts = M.feasible_actions(pending, free, b, omega, t)
    if not acts:
        CNT["no_feasible"] += 1; return None
    acts2 = [a for a in acts if M._keeps_reserve(a, b, omega, t)]
    if not acts2:
        CNT["reserve_blocked"] += 1; return None
    best = min(acts2, key=lambda a: M._score_dpp(a, b, omega, QE, QP, QU, V))
    if M._score_dpp(best, b, omega, QE, QP, QU, V) < 0:
        CNT["act"] += 1; return best
    CNT["score_idle"] += 1; return None

def decompose(seeds=(7, 8, 9), V=10):
    agg = dict(called=0, act=0, score_idle=0, no_feasible=0, reserve_blocked=0, tasks=0)
    for sd in seeds:
        CNT.clear(); CNT.update(called=0, act=0, score_idle=0, no_feasible=0, reserve_blocked=0)
        r = M.run_sim(counting_policy, V=V, seed=sd)
        for k in ("called", "act", "score_idle", "no_feasible", "reserve_blocked"):
            agg[k] += CNT[k]
        agg["tasks"] += r["tasks_started"]
    n = len(seeds)
    return {k: v / n for k, v in agg.items()}

def main():
    hdr = f"{'case':>16} {'called':>8} {'act':>7} {'reserveBlk':>11} {'scoreIdle':>10} {'noFeas':>7} {'tasks':>7}"
    print("== n_sat axis (B_max=18000) =="); print(hdr)
    for ns in (8, 16):
        M.N_SAT = ns
        M.SAT_PHASE = [int(k * M.N_SLOTS_PER_ORBIT / ns) for k in range(ns)]
        d = decompose()
        print(f"{'nsat=%d' % ns:>16} {d['called']:8.0f} {d['act']:7.0f} {d['reserve_blocked']:11.0f} "
              f"{d['score_idle']:10.0f} {d['no_feasible']:7.0f} {d['tasks']:7.0f}")
    M.N_SAT = 4; M.SAT_PHASE = [int(k * M.N_SLOTS_PER_ORBIT / 4) for k in range(4)]

    print("== battery axis (n_sat=4) =="); print(hdr)
    for B in (18000, 20000, 24000):
        M.B_MAX = float(B); M.B_INIT = 0.6 * B
        d = decompose()
        print(f"{'B=%d' % B:>16} {d['called']:8.0f} {d['act']:7.0f} {d['reserve_blocked']:11.0f} "
              f"{d['score_idle']:10.0f} {d['no_feasible']:7.0f} {d['tasks']:7.0f}")
    M.B_MAX = 18000.0; M.B_INIT = 0.6 * 18000

if __name__ == "__main__":
    main()
