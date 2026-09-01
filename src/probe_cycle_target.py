#!/usr/bin/env python3
"""Quick probe: does the cycle-recovery condition hold at lower targets (nominal 12.2/16.6/0.85)?"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_common import M, apply_config  # noqa: E402
from exp3_cycle_recovery import fit_kappa_D, coverage, WARMUP  # noqa: E402

HORIZON = M.N_SLOTS_PER_ORBIT * 55
SEEDS = list(range(7, 17))


def _probe(args):
    u, seed = args
    apply_config(dict(U_TGT=u))
    r = M.run_sim(M.POLICIES["Lyapunov (ours)"], V=10.0, horizon=HORIZON, seed=seed)
    L = np.asarray(r["log"]["L_orbit"], float)[WARMUP:]
    down = r["blackout_slots"] / (M.N_SAT * HORIZON)
    return L, down, r["min_fill"]


def main():
    import concurrent.futures as cf
    for u in (0.13, 0.15, 0.16, 0.17, 0.18):
        with cf.ProcessPoolExecutor(max_workers=10) as ex:
            res = list(ex.map(_probe, [(u, sd) for sd in SEEDS]))
        Ls = np.array([r[0] for r in res])
        Lbar = Ls.mean(0)
        nf = int(len(Lbar) * 0.6) - 1
        k, d = fit_kappa_D(Lbar[:nf + 1])
        cov = coverage(Lbar, k, d, nf)
        print(f"U={u}: Lbar min={Lbar.min():.0f} max={Lbar.max():.0f} last={Lbar[-1]:.0f} "
              f"kappa={k:.3f} D={d:.0f} cov={cov:.2f} "
              f"fill={np.mean([r[2] for r in res]):.3f} down={np.mean([r[1] for r in res])*100:.3f}%",
              flush=True)


if __name__ == "__main__":
    main()
