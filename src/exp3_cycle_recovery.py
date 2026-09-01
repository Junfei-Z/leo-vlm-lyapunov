#!/usr/bin/env python3
"""TMC Experiment 3: eclipse cycle-recovery condition (expectation form).

Sweep panel [10..14] W, battery [15..23] kJ, reserve [0.8,0.9,1.0] (+ nominal
12.2/16.6/0.85) at the feasible mainline target U=0.19, 10 seeds, 55 cycles
(5 warm-up + 50 measured).  The theory (Assumption: Cycle recovery) is an
EXPECTATION bound  E[L(Q(t_{k+1}))] <= kappa E[L(Q(t_k))] + D, so we average
L over seeds at each orbit boundary (Lbar_k) and fit the bound on Lbar, with
coverage checked on held-out cycles of Lbar (per-seed coverage also reported).
Battery survival (blackout-free over the measured cycles) is recorded
separately, matching the reserve-guard vs queue-recovery separation.

Outputs:
  data/exp3_raw.npz   -- raw per-(setting,seed) L trajectories & battery stats
  data/exp3_cycle.csv -- per-setting fit and outcomes (expectation form)
  figures/exp3_cycle.pdf
"""
import os, sys, csv
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_common import M, apply_config  # noqa: E402
from figstyle import apply_house_style, savefig_pub, PALETTE  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
WARMUP = 5
N_CYCLES = 50
HORIZON = M.N_SLOTS_PER_ORBIT * (WARMUP + N_CYCLES)
SEEDS = list(range(7, 17))
FIT_FRAC = 0.6
COV_REQ = 0.9
U_TGT = 0.15
OUT_RAW = os.path.join(HERE, "..", "data", "exp3_raw.npz")
OUT_CSV = os.path.join(HERE, "..", "data", "exp3_cycle.csv")
OUT_PDF = os.path.join(HERE, "..", "figures", "exp3_cycle.pdf")

PANELS = [10.0, 11.0, 12.0, 13.0, 14.0]
BATTS = [15.0, 17.0, 19.0, 21.0, 23.0]
RESERVES = [0.8, 0.9, 1.0]


def fit_kappa_D(L):
    """Constrained regression L_{k+1} = kappa L_k + D, 0<=kappa<1, D>=0."""
    from scipy.optimize import minimize
    x = np.array(L[:-1], float); y = np.array(L[1:], float)

    def obj(p):
        k, d = p
        return float(np.sum((y - (k * x + d)) ** 2))
    best, bestv = None, None
    for k0 in (0.1, 0.5, 0.9):
        r = minimize(obj, [k0, 1.0], bounds=[(0.0, 0.999999), (0.0, None)],
                     method="L-BFGS-B")
        if best is None or r.fun < bestv:
            best, bestv = r, r.fun
    return float(best.x[0]), float(best.x[1])


def coverage(L, k, d, n_fit):
    n = len(L) - 1
    hit = sum(1 for kk in range(n_fit, n) if L[kk + 1] <= k * L[kk] + d)
    return hit / max(1, n - n_fit)


def _probe(args):
    panel, batt_kj, rho, seed = args
    apply_config(dict(P_SOLAR=panel, B_MAX=batt_kj * 1000.0,
                      B_INIT=0.6 * batt_kj * 1000.0, RESERVE_FRAC=rho, U_TGT=U_TGT))
    r = M.run_sim(M.POLICIES["Lyapunov (ours)"], V=10.0, horizon=HORIZON, seed=seed)
    L = np.asarray(r["log"]["L_orbit"], float)[WARMUP:]
    bo = np.asarray(r["log"]["blackout_orbit"], int)
    mb = np.asarray(r["log"]["minb_orbit"], float)
    measured_slots = (len(L) - 1) * M.N_SLOTS_PER_ORBIT * M.N_SAT
    down = (int(bo[-1]) - int(bo[WARMUP])) / max(1, measured_slots)
    min_batt = float(np.min(mb[WARMUP:]))
    return dict(panel=panel, batt=batt_kj, rho=rho, seed=seed,
                L=np.asarray(L, float), down=down, min_batt=min_batt)


def main():
    apply_house_style()
    settings = [(p, b, rho) for p in PANELS for b in BATTS for rho in RESERVES]
    settings.append((12.2, 16.6, 0.85))
    print(f"=== EXP3 cycle recovery (expectation form), U_tgt={U_TGT}, "
          f"{len(settings)} settings x {len(SEEDS)} seeds x {WARMUP + N_CYCLES} orbits ===", flush=True)

    tasks = [(p, b, rho, sd) for (p, b, rho) in settings for sd in SEEDS]
    import concurrent.futures as cf
    with cf.ProcessPoolExecutor(max_workers=10) as ex:
        res = list(ex.map(_probe, tasks))

    # save raw trajectories for reproducibility
    raw = {}
    for r in res:
        key = f"{r['panel']}_{r['batt']}_{r['rho']}"
        raw.setdefault(f"L_{key}", []).append(r["L"])
    np.savez_compressed(OUT_RAW, **{k: np.array(v) for k, v in raw.items()})
    print(f"  wrote {OUT_RAW}", flush=True)

    rows = []
    for (p, b, rho) in settings:
        rr = [r for r in res if (r["panel"], r["batt"], r["rho"]) == (p, b, rho)]
        Lbar = np.mean([r["L"] for r in rr], axis=0)            # seed-averaged trajectory
        n_fit = int(len(Lbar) * FIT_FRAC) - 1
        k, d = fit_kappa_D(Lbar[:n_fit + 1])
        cov = coverage(Lbar, k, d, n_fit)
        cov_seed = np.mean([coverage(r["L"], k, d, n_fit) for r in rr])
        down = np.mean([r["down"] for r in rr])
        minb = np.min([r["min_batt"] for r in rr])
        batt_ok = down == 0.0
        rec_ok = bool(k < 1.0 and cov >= COV_REQ)
        rows.append(dict(panel=p, battery=b, reserve=rho, kappa=k, D=d, cov=cov,
                         cov_seed=cov_seed, down=down, min_batt=minb,
                         battery_ok=batt_ok, recovery_ok=rec_ok))
        tag = ("BOTH" if (batt_ok and rec_ok) else
               "batt-only" if batt_ok else
               "recovery-only" if rec_ok else "neither")
        print(f"  P={p:4.1f} B={b:4.0f}kJ rho={rho:.1f}  kappa={k:.3f} D={d:7.1f} "
              f"cov(mean)={cov:.2f} cov(seed)={cov_seed:.2f} down={down*100:.2f}% "
              f"minB={minb:6.0f}J [{tag}]", flush=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"  wrote {OUT_CSV}", flush=True)

    # ---- figure ----
    import matplotlib.pyplot as plt
    nom = [r for r in rows if r["panel"] == 12.2 and r["battery"] == 16.6 and r["reserve"] == 0.85][0]
    fig, ax = plt.subplots(1, 2, figsize=(15, 6.2), gridspec_kw=dict(width_ratios=[1, 1.15]))
    # (a) nominal: seed-averaged trajectory envelope
    rr = [r for r in res if (r["panel"], r["batt"], r["rho"]) == (12.2, 16.6, 0.85)]
    Ls = np.array([r["L"] for r in rr]); Lbar = Ls.mean(0)
    ax[0].plot(Lbar[:-1], Lbar[1:], "o-", ms=5, color=PALETTE["blue_main"], lw=1.8,
               label=r"seed-averaged $E[L(t_k)]$ trajectory")
    xs = np.linspace(0, Lbar.max() * 1.05, 60)
    ax[0].plot(xs, nom["kappa"] * xs + nom["D"], "-", color=PALETTE["red_strong"], lw=2.5,
               label=fr"fitted envelope $\kappa={nom['kappa']:.2f},\ D={nom['D']:.0f}$")
    ax[0].plot(xs, xs, "--", color=PALETTE["neutral"], lw=1.8, label="$L_{k+1}=L_k$")
    ax[0].set_xlabel(r"$\mathbb{E}[L(Q(t_k))]$"); ax[0].set_ylabel(r"$\mathbb{E}[L(Q(t_{k+1}))]$")
    ax[0].grid(alpha=0.3); ax[0].legend(loc="upper left", fontsize=13)
    ax[0].set_title("(a) nominal cycle envelope (expectation)", loc="left")
    # (b) phase diagram at rho=0.85
    sub = [r for r in rows if r["reserve"] == 0.85]
    P = np.array([r["panel"] for r in sub]); B = np.array([r["battery"] for r in sub])
    both = np.array([r["battery_ok"] and r["recovery_ok"] for r in sub])
    bonly = np.array([r["battery_ok"] and not r["recovery_ok"] for r in sub])
    ronly = np.array([(not r["battery_ok"]) and r["recovery_ok"] for r in sub])
    none = np.array([not r["battery_ok"] and not r["recovery_ok"] for r in sub])
    ax[1].scatter(P[both], B[both], s=220, marker="o", color=PALETTE["green_3"], edgecolor="k",
                  label="battery + queue recovery")
    ax[1].scatter(P[bonly], B[bonly], s=220, marker="^", color=PALETTE["highlight"], edgecolor="k",
                  label="battery only (no queue recovery)")
    ax[1].scatter(P[ronly], B[ronly], s=220, marker="P", color=PALETTE["blue_secondary"], edgecolor="k",
                  label="queue recovery only")
    ax[1].scatter(P[none], B[none], s=220, marker="x", color=PALETTE["red_strong"], label="neither")
    ax[1].axvline(12.2, color=PALETTE["neutral"], ls=":", lw=1.8)
    ax[1].axhline(16.6, color=PALETTE["neutral"], ls=":", lw=1.8)
    ax[1].set_xlabel(r"panel power $P^{\mathrm{solar}}$ (W)")
    ax[1].set_ylabel(r"battery $B^{\max}$ (kJ)")
    ax[1].grid(alpha=0.3); ax[1].legend(loc="upper left", fontsize=12)
    ax[1].set_title(r"(b) phase diagram at $\rho=0.85$", loc="left")
    savefig_pub(fig, OUT_PDF)
    print(f"  wrote {OUT_PDF}", flush=True)

    nbatt = sum(r["battery_ok"] for r in rows)
    nboth = sum(r["battery_ok"] and r["recovery_ok"] for r in rows)
    nrec = sum(r["recovery_ok"] for r in rows)
    print(f"\nPAPER NUMBERS: battery survives {nbatt}/{len(rows)} settings; queue recovery "
          f"{nrec}/{len(rows)}; both {nboth}/{len(rows)}; nominal kappa={nom['kappa']:.3f}, "
          f"D={nom['D']:.1f}, cov(mean)={nom['cov']:.2f}, down={nom['down']*100:.3f}%")


if __name__ == "__main__":
    main()
