#!/usr/bin/env python3
"""Rerun the fixed-target V sweep at U=0.15 and U=0.16 (backup for the paper decision)."""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_common import M, apply_config  # noqa: E402

HORIZON = M.N_SLOTS_PER_ORBIT * 20
SEEDS = list(range(7, 17))
VS = [1.0, 3.0, 10.0, 30.0, 100.0, 300.0]


def _p(args):
    v, u, sd = args
    apply_config(dict(U_TGT=u))
    r = M.run_sim(M.POLICIES["Lyapunov (ours)"], V=v, horizon=HORIZON, seed=sd)
    return (v, r["total_quality"], r["min_fill"], r["avg_QU"],
            float(np.max(r["QU_end"])) / HORIZON)


def main():
    import concurrent.futures as cf
    for u in (0.15, 0.16):
        with cf.ProcessPoolExecutor(max_workers=10) as ex:
            res = list(ex.map(_p, [(v, u, sd) for v in VS for sd in SEEDS]))
        print(f"=== U_tgt={u} ===", flush=True)
        for v in VS:
            rr = [r for r in res if r[0] == v]
            print(f"  V={v:4.0f} totQ={np.mean([r[1] for r in rr]):7.0f} "
                  f"maxmin={np.mean([r[2] for r in rr]):.3f} "
                  f"avgQU={np.mean([r[3] for r in rr]):6.1f} "
                  f"QUend/H={np.mean([r[4] for r in rr]):.4f}", flush=True)


if __name__ == "__main__":
    main()
