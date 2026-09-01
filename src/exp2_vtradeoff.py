#!/usr/bin/env python3
"""TMC Experiment 2: fixed-target V trade-off.

At a fixed feasible target (10% below the Experiment-1 boundary), sweep V over
[1,3,10,30,100,300] on the same 20-orbit horizon and paired seeds.  Record total
quality, max-min quality, average and terminal quality-queue backlog.  The MILP
part (exp2_milp.py) computes the total-quality gap vs 1/V on small instances.

Outputs:
  data/exp2_vsweep.csv      -- per-V statistics
  figures/exp2_vsweep.pdf   -- total & max-min quality vs V; backlog vs V
"""
import os, sys, csv
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_common import M, apply_config  # noqa: E402
from figstyle import apply_house_style, savefig_pub, PALETTE  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ORBITS = 20
HORIZON = M.N_SLOTS_PER_ORBIT * ORBITS
SEEDS = list(range(7, 17))
VS = [1.0, 3.0, 10.0, 30.0, 100.0, 300.0]
OUT_CSV = os.path.join(HERE, "..", "data", "exp2_vsweep.csv")
OUT_PDF = os.path.join(HERE, "..", "figures", "exp2_vsweep.pdf")
BOUND_CSV = os.path.join(HERE, "..", "data", "exp1_utgt.csv")


def read_boundary():
    with open(BOUND_CSV, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("note") == "bisection_boundary":
                return float(row["min_fill"])
    raise SystemExit("exp1 boundary not found; run exp1_target_boundary.py first")


def _probe(args):
    v, u, seed = args
    apply_config(dict(U_TGT=float(u)))
    r = M.run_sim(M.POLICIES["Lyapunov (ours)"], V=float(v), horizon=HORIZON, seed=seed)
    H = float(HORIZON)
    return dict(v=float(v), seed=seed, totq=float(r["total_quality"]),
                mf=float(r["min_fill"]), avg_qu=float(r["avg_QU"]),
                qu_end_max=float(np.max(r["QU_end"]) / H))


def main():
    apply_house_style()
    bd = read_boundary()
    if len(sys.argv) > 1:
        utgt = float(sys.argv[1])
    else:
        utgt = round(0.9 * bd, 3) if bd > 0 else 0.20
    print(f"=== EXP2 fixed-target V trade-off (U_tgt={utgt}, {ORBITS} orbits, "
          f"{len(SEEDS)} seeds) ===", flush=True)

    tasks = [(v, utgt, sd) for v in VS for sd in SEEDS]
    import concurrent.futures as cf
    with cf.ProcessPoolExecutor(max_workers=10) as ex:
        res = list(ex.map(_probe, tasks))
    rows = []
    for v in VS:
        rr = [r for r in res if r["v"] == v]
        rows.append(dict(V=v, totq=np.mean([r["totq"] for r in rr]),
                         totq_sd=np.std([r["totq"] for r in rr]),
                         mf=np.mean([r["mf"] for r in rr]),
                         mf_sd=np.std([r["mf"] for r in rr]),
                         avg_qu=np.mean([r["avg_qu"] for r in rr]),
                         qu_end=np.mean([r["qu_end_max"] for r in rr])))
        print(f"  V={v:6.0f}  totQ={rows[-1]['totq']:7.0f}  maxmin={rows[-1]['mf']:.3f}  "
              f"avgQU={rows[-1]['avg_qu']:8.1f}  QUend/H={rows[-1]['qu_end']:.4f}", flush=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"  wrote {OUT_CSV}   (target U_tgt={utgt})", flush=True)

    # ---- figure ----
    import matplotlib.pyplot as plt
    V = np.array([r["V"] for r in rows])
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.6))
    ax[0].plot(V, [r["totq"] for r in rows], "o-", color=PALETTE["blue_main"],
               lw=2.5, ms=8, label="total delivered quality")
    ax[0].axhline(utgt, color=PALETTE["neutral"], ls="--", lw=1.8,
                  label=f"target $U^{{\\mathrm{{tgt}}}}={utgt}$")
    ax[0].set_xscale("log"); ax[0].set_xlabel(r"control weight $V$")
    ax[0].set_ylabel("total delivered quality"); ax[0].grid(alpha=0.3)
    ax[0].legend(loc="lower right", fontsize=14)
    ax[0].set_title("(a) total quality at fixed target", loc="left")
    ax[1].plot(V, [r["mf"] for r in rows], "^-", color=PALETTE["red_strong"],
               lw=2.5, ms=8, label="max-min quality")
    ax[1].axhline(utgt, color=PALETTE["neutral"], ls="--", lw=1.8, label="target")
    ax[1].set_xscale("log"); ax[1].set_xlabel(r"control weight $V$")
    ax[1].set_ylabel("max-min quality"); ax[1].grid(alpha=0.3)
    ax[1].legend(loc="lower right", fontsize=14)
    ax[1].set_title("(b) max-min quality vs target", loc="left")
    fig.suptitle(f"average quality-queue backlog grows from {rows[0]['avg_qu']:.0f} "
                 f"to {rows[-1]['avg_qu']:.0f}", fontsize=13)
    savefig_pub(fig, OUT_PDF)
    print(f"  wrote {OUT_PDF}", flush=True)


if __name__ == "__main__":
    main()
