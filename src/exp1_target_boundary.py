#!/usr/bin/env python3
"""TMC Experiment 1: max-min target feasibility boundary.

Sweep U_TGT over a dense grid and via outer bisection at V=30 over 20 orbits,
10 paired seeds, on the TLE primary configuration.  A target is feasible for a
seed if (i) every source fill_i = delivered_i/arrivals_i >= U - DELTA,
(ii) max_i QU_i(T)/H <= QTOL (normalized quality-queue endpoint), and
(iii) downtime < SLA (5%).  The boundary is the largest U feasible for >= FRAC
of the seeds.  Outer bisection (tolerance 0.01) must recover the same boundary
with far fewer target evaluations.

Outputs:
  data/exp1_utgt.csv      -- per-U statistics (mean max-min quality, mean
                             normalized queue endpoint, feasible fraction,
                             downtime) and the two boundary estimates
  figures/exp1_utgt.pdf   -- max-min quality and normalized queue endpoint vs
                             U, shaded feasible interval, mainline U=0.30
Stdout: the boundary numbers for the paper table.
"""
import os, sys, csv, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_common import M, apply_config, PRIMARY  # noqa: E402
from figstyle import apply_house_style, savefig_pub, PALETTE  # noqa: E402

V = 30.0
ORBITS = 20
HORIZON = M.N_SLOTS_PER_ORBIT * ORBITS
SEEDS = list(range(7, 17))          # 10 paired seeds
DELTA = 0.01                        # fill tolerance
QTOL = 0.01                         # normalized per-source queue endpoint tolerance
SLA = 0.05                          # downtime SLA
FRAC = 0.9                          # fraction of seeds that must pass
U_LO, U_HI = 0.02, 0.45
BIS_TOL = 0.01

OUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "exp1_utgt.csv")
OUT_PDF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures", "exp1_utgt.pdf")


def _probe_one(u, seed):
    apply_config(dict(U_TGT=float(u)))
    r = M.run_sim(M.POLICIES["Lyapunov (ours)"], V=V, horizon=HORIZON, seed=seed)
    H = float(HORIZON)
    fill = np.asarray(r["fill"], float)
    qu_end_max = float(np.max(r["QU_end"]) / H)
    down = float(r["blackout_slots"] / (M.N_SAT * H))
    ok = bool(np.all(fill >= u - DELTA)) and qu_end_max <= QTOL and down < SLA
    return dict(seed=seed, fill_min=float(fill.min()), fill_mean=float(fill.mean()),
                qu_end_max=qu_end_max, down=down, ok=ok,
                totq=float(r["total_quality"]))


def probe(u, seeds=SEEDS, workers=10):
    """Evaluate target u over seeds (parallel), return list of per-seed dicts."""
    if workers > 1 and len(seeds) > 1:
        import concurrent.futures as cf
        with cf.ProcessPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(_probe_one, [u] * len(seeds), seeds))
    return [_probe_one(u, sd) for sd in seeds]


def feasible_fraction(rows):
    return float(np.mean([r["ok"] for r in rows]))


def dense_sweep():
    grid = np.arange(0.05, 0.40 + 1e-9, 0.025)
    out = []
    feasible_u = []
    for u in grid:
        rows = probe(float(u))
        fr = feasible_fraction(rows)
        if fr >= FRAC:
            feasible_u.append(float(u))
        out.append(dict(u=float(u),
                        mf=np.mean([r["fill_min"] for r in rows]),
                        mf_sd=np.std([r["fill_min"] for r in rows]),
                        qu=np.mean([r["qu_end_max"] for r in rows]),
                        qu_sd=np.std([r["qu_end_max"] for r in rows]),
                        down=np.mean([r["down"] for r in rows]),
                        fr=fr, totq=np.mean([r["totq"] for r in rows])))
        print(f"  U={u:.3f}  feasible_frac={fr:.2f}  min_fill={out[-1]['mf']:.3f} "
              f"qu_end/H={out[-1]['qu']:.4f}  down%={out[-1]['down']*100:.2f}", flush=True)
    boundary = max(feasible_u) if feasible_u else U_LO
    return out, boundary


def bisection():
    lo, hi = U_LO, U_HI
    evals = 0
    while hi - lo > BIS_TOL:
        mid = 0.5 * (lo + hi)
        fr = feasible_fraction(probe(mid))
        evals += 1
        if fr >= FRAC:
            lo = mid
        else:
            hi = mid
        print(f"  bisect U={mid:.4f} fr={fr:.2f} -> [{lo:.4f}, {hi:.4f}]", flush=True)
    return 0.5 * (lo + hi), evals


def main():
    apply_house_style()
    print(f"=== EXP1 max-min target feasibility boundary (V={V:.0f}, {ORBITS} orbits, "
          f"{len(SEEDS)} seeds, delta={DELTA}, QTOL={QTOL}, SLA={SLA*100:.0f}%) ===", flush=True)
    dense, bd = dense_sweep()
    bb, bevals = bisection()
    print(f"  dense boundary = {bd:.3f} ; bisection boundary = {bb:.3f} "
          f"({bevals} evaluations)", flush=True)

    rows = [dict(u=r["u"], min_fill=r["mf"], min_fill_sd=r["mf_sd"],
                 qu_end=r["qu"], qu_end_sd=r["qu_sd"], down=r["down"],
                 feasible_frac=r["fr"], totq=r["totq"]) for r in dense]
    rows.append(dict(u=-1, min_fill=bd, min_fill_sd=0.0, qu_end=0.0, qu_end_sd=0.0,
                     down=0.0, feasible_frac=-1.0, totq=0.0, note="dense_boundary",
                     bisect_evals=-1))
    rows.append(dict(u=-2, min_fill=bb, min_fill_sd=0.0, qu_end=0.0, qu_end_sd=0.0,
                     down=0.0, feasible_frac=-1.0, totq=0.0, note="bisection_boundary",
                     bisect_evals=bevals))
    with open(OUT_CSV, "w", newline="") as f:
        fn = ["u", "min_fill", "min_fill_sd", "qu_end", "qu_end_sd", "down",
              "feasible_frac", "totq", "note", "bisect_evals"]
        w = csv.DictWriter(f, fieldnames=fn)
        w.writeheader(); w.writerows(rows)
    print(f"  wrote {OUT_CSV}", flush=True)

    # ---- figure ----
    import matplotlib.pyplot as plt
    uu = np.array([r["u"] for r in dense])
    mf = np.array([r["mf"] for r in dense]); mf_sd = np.array([r["mf_sd"] for r in dense])
    qu = np.array([r["qu"] for r in dense]); qu_sd = np.array([r["qu_sd"] for r in dense])
    fr = np.array([r["fr"] for r in dense])
    feas = fr >= FRAC
    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    ax.fill_between(uu, 0, 0.5, where=feas, color=PALETTE["green_3"], alpha=0.35,
                    label="empirically feasible")
    ax.plot(uu, mf, "o-", color=PALETTE["blue_main"], lw=2.5, ms=7, label="max-min quality")
    ax.fill_between(uu, mf - mf_sd, mf + mf_sd, color=PALETTE["blue_main"], alpha=0.15)
    ax.axvline(0.30, color=PALETTE["neutral"], ls="--", lw=2, label=r"mainline $U^{\mathrm{tgt}}=0.30$")
    ax.axvline(bd, color=PALETTE["red_strong"], ls=":", lw=2.2, label=f"empirical boundary ({bd:.2f})")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel(r"quality target $U^{\mathrm{tgt}}$")
    ax.set_ylabel("max-min quality")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=14)
    ax2 = ax.twinx()
    ax2.plot(uu, qu, "s--", color=PALETTE["highlight"], ms=6, lw=2, label="max QU end / H")
    ax2.set_ylabel(r"normalized quality-queue endpoint $\max_i Q_i^U(T)/H$")
    ax2.grid(False)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center right", fontsize=13)
    savefig_pub(fig, OUT_PDF)
    print(f"  wrote {OUT_PDF}", flush=True)

    # paper-table numbers
    print("\nPAPER TABLE (exp1):")
    print(f"  dense sweep boundary  = {bd:.3f}")
    print(f"  bisection boundary    = {bb:.3f}  (evals={bevals})")
    print(f"  mainline margin       = {0.30 - bb:.3f} below boundary"
          if bb < 0.30 else f"  WARNING mainline 0.30 ABOVE boundary {bb:.3f}")


if __name__ == "__main__":
    main()
