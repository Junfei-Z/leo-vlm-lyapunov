#!/usr/bin/env python3
"""P0-2: MPC baseline (windowed MILP + rolling executor) on the TLE mainline.

Design: paper_staging/MPC_baseline_design.md (advisor-approved with three fixes).
  - Two solver variants per window tier W in {1035, 2070, 5736} s:
      real-time : per-solve cap = re-plan interval R=60 s; the incumbent is used;
                  the new plan takes effect only ceil(wall) slots after the solve
                  starts (the old plan keeps executing meanwhile). No solve acts
                  on state it did not pay the wall-clock to see.
      oracle    : solve time NOT charged to the sim clock (cap 300 s wall for
                  practicality); plan takes effect immediately. Upper bound.
  - Carried-in deficit (fix 2): window objective max U s.t. for every source i
      U <= D_i(t) + sum_window delivered_i,  D_i = cum delivered - cum target.
  - Information disclosure: deterministic schedules (sunlit / ISL / forecast)
    + current state; future arrivals only as the rate lambda (expected counts).
  - Guard exemption, shared hard constraints: execution goes through the SAME
    feasible_actions() oracle as every other policy; a planned action that is
    infeasible at execution (no arrival, sat busy, energy) is dropped (declared
    repair), never re-optimized between re-solves.
  - Planning grid Delta=30 s macro-slots; counts n[k,s,i,m] (integer) with
    per-sat time capacity, per-(k,i) expected-arrival cap lambda*Delta, battery
    dynamics with terminal condition b(K) >= platform need to next sunrise.
  - Optimality-gap instrument (declared): gap = (LP_bound - incumbent)/LP_bound
    with the LP relaxation solved separately (CBC via PuLP does not expose the
    MIP bound); conservative, same formula for every solve.

Usage:
  python3 src/15_mpc_baseline.py smoke              # 1 solve/tier: size + time
  python3 src/15_mpc_baseline.py run W TIER SEED    # one full 3-orbit run
  outputs data/mpc_runs.csv (metrics) + data/mpc_solves.csv (per-solve log)
"""
import importlib.util, os, sys, json, csv, time
import numpy as np
import pulp

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("splitpipe", os.path.join(HERE, "04_split_pipeline.py"))
M = importlib.util.module_from_spec(spec); sys.modules["splitpipe"] = M; spec.loader.exec_module(M)

# ---------------- TLE overrides (identical to src/14 apply_primary) ----------------
npz = np.load(os.path.join(HERE, "..", "data", "tle_traces.npz"))
SUNLIT = npz["sunlit"]; ISL = npz["isl"]; META = json.loads(str(npz["meta"]))
assert META["gate"] == "PASS"
T, S = SUNLIT.shape
S2 = np.concatenate([SUNLIT, SUNLIT], axis=0)
F = np.zeros((T, S), dtype=np.int32)
for s in range(S):
    col = S2[:, s]
    for t in range(T):
        if col[t]:
            j = t
            while j < 2 * T and col[j]:
                j += 1
            k = j
            while k < 2 * T and not col[k]:
                k += 1
            F[t, s] = k - j
        else:
            j = t
            while j < 2 * T and not col[j]:
                j += 1
            F[t, s] = j - t
BASE_ISL = ISL.copy()

def apply_primary():
    M.sunlit_indicator = lambda t, sat: 1 if SUNLIT[t % T, sat] else 0
    M.isl_connected = lambda t, s1, s2: bool(BASE_ISL[min(t, T - 1), s1, s2]) if s1 != s2 else False
    M.eclipse_slots_to_sunrise = lambda t, s: int(F[t % T, s])
    M.P_SOLAR = 12.2
    M.B_MAX = 16580.0
    M.B_INIT = 0.6 * 16580.0
    M.P_BASE = 7.0
    M.ARRIVAL_PROB = 0.10
    M.HANDOFF_S = 0.69

# ---------------- MPC constants (declared) ----------------
DELTA = 30              # macro-slot [s]
R_SOLVE = 60            # re-plan interval [s]
W_TIERS = {"half": 1035, "eclipse": 2070, "orbit": 5736}
CAP = {"realtime": 60, "oracle": 300}   # per-solve wall cap [s]
EPS_TQ = 1e-3           # tiny total-quality tiebreak in the objective (declared)

STANDALONE = ["2B", "3B"]
PAIRS = [(a, b) for a in range(4) for b in range(4) if a != b]

def _split_occup():
    """Front/rear slot occupancy under the mainline handoff (h=0.69 overflow)."""
    front_busy_s = M.CONFIGS["7B"]["T"] / 2.0 + M.HANDOFF_S
    extra = 0 if front_busy_s <= M.SPLIT["N_front"] * M.TAU else 1
    return M.SPLIT["N_front"] + extra, M.SPLIT["N_rear"]

def build_milp(t0, b0, D, W_s, lam):
    """Windowed MILP at real slot t0 with battery b0[4], carried deficits D[6]."""
    K = int(np.ceil(W_s / DELTA))
    cfg = M.CONFIGS
    occ_f, occ_r = _split_occup()
    # deterministic per-macro-slot harvest [J] and ISL pair availability
    harvest = np.zeros((K, 4))
    pair_ok = np.zeros((K, len(PAIRS)), dtype=bool)
    for k in range(K):
        a, b = t0 + k * DELTA, t0 + (k + 1) * DELTA
        mid = min((a + b) // 2, T - 1) % T
        for s in range(4):
            harvest[k, s] = M.P_SOLAR * sum(SUNLIT[tt % T, s] for tt in range(a, b)) * M.TAU
        for pi, (sf, sr) in enumerate(PAIRS):
            pair_ok[k, pi] = bool(BASE_ISL[mid, sf, sr])
    prob = pulp.LpProblem("mpc", pulp.LpMaximize)
    U = pulp.LpVariable("U", lowBound=None)
    zs = []   # battery-shortfall slacks (soft lower bound, declared amendment)
    # integer action counts
    x = {}   # standalone (k,s,i,m)
    y = {}   # split (k,pi,i)
    for k in range(K):
        for s in range(4):
            for i in range(6):
                for m in STANDALONE:
                    ub = DELTA // cfg[m]["N"]
                    x[k, s, i, m] = pulp.LpVariable(f"x_{k}_{s}_{i}_{m}", 0, ub, "Integer")
        for pi in range(len(PAIRS)):
            if not pair_ok[k, pi]:
                continue
            for i in range(6):
                y[k, pi, i] = pulp.LpVariable(f"y_{k}_{pi}_{i}", 0, DELTA // (occ_f + occ_r), "Integer")
    bat = {(k, s): pulp.LpVariable(f"b_{k}_{s}", 0, M.B_MAX) for k in range(K + 1) for s in range(4)}
    for s in range(4):
        prob += bat[0, s] == b0[s]
    lam_cap = lam * DELTA          # expected arrivals per source per macro-slot
    for k in range(K):
        # global commitment capacity: the simulator commits <=1 action per 1 s slot
        prob += (pulp.lpSum(x[k, s, i, m] for s in range(4) for i in range(6) for m in STANDALONE)
                 + pulp.lpSum(v for (kk, pi, i), v in y.items() if kk == k)) <= DELTA
        for i in range(6):
            prob += (pulp.lpSum(x[k, s, i, m] for s in range(4) for m in STANDALONE)
                     + pulp.lpSum(y[k, pi, i] for pi in range(len(PAIRS)) if (k, pi, i) in y)) <= lam_cap
        for s in range(4):
            tload = pulp.lpSum(cfg[m]["N"] * x[k, s, i, m] for i in range(6) for m in STANDALONE)
            eload = pulp.lpSum(cfg[m]["E"] * x[k, s, i, m] for i in range(6) for m in STANDALONE)
            for pi, (sf, sr) in enumerate(PAIRS):
                for i in range(6):
                    if (k, pi, i) not in y:
                        continue
                    if sf == s:
                        tload += occ_f * y[k, pi, i]; eload += M.SPLIT["E_front"] * y[k, pi, i]
                    if sr == s:
                        tload += occ_r * y[k, pi, i]; eload += M.SPLIT["E_rear"] * y[k, pi, i]
            prob += tload <= DELTA
            # soft battery lower bound: slack z mirrors the simulator's safe-mode
            # recourse (declared amendment; penalty makes it never trade for quality)
            z = pulp.LpVariable(f"z_{k}_{s}", 0)
            zs.append(z)
            prob += bat[k + 1, s] <= bat[k, s] + harvest[k, s] - M.P_BASE * DELTA * M.TAU - eload + z
    # terminal: enough charge to carry the platform to the next sunrise (full need, no guard factor)
    t_end = t0 + K * DELTA
    for s in range(4):
        zT = pulp.LpVariable(f"zT_{s}", 0)
        zs.append(zT)
        prob += bat[K, s] + zT >= M.P_BASE * M.TAU * int(F[t_end % T, s])
    # objective: max-min with carried-in deficits (fix 2) + tiny total-quality tiebreak
    tq = pulp.lpSum(M.q_im(i, m) * x[k, s, i, m] for (k, s, i, m) in x) + \
         pulp.lpSum(M.q_im(i, "7B") * y[k, pi, i] for (k, pi, i) in y)
    for i in range(6):
        di = pulp.lpSum(M.q_im(i, m) * x[k, s, ii, m] for (k, s, ii, m) in x if ii == i) + \
             pulp.lpSum(M.q_im(i, "7B") * y[k, pi, ii] for (k, pi, ii) in y if ii == i)
        prob += U <= D[i] + di
    PEN_Z = 10.0   # quality-units per Joule of shortfall (>> 0.23 max quality/J)
    prob += U + EPS_TQ * tq - PEN_Z * pulp.lpSum(zs)
    return prob, x, y, K

def extract_plan(x, y):
    plan = {}
    for (k, s, i, m), v in x.items():
        n = int(round(v.varValue or 0))
        if n > 0:
            plan[(k, "standalone", s, i, m)] = n
    for (k, pi, i), v in y.items():
        n = int(round(v.varValue or 0))
        if n > 0:
            sf, sr = PAIRS[pi]
            plan[(k, "split", sf, sr, i)] = n
    return plan

def solve_window(prob, x, y, cap_s):
    """Solve with wall cap; return (wall, status, incumbent_obj, plan, lp_bound, gap).
    The plan is extracted BEFORE the LP-relaxation solve: pulp's copy() shares the
    variable objects, so the relaxation overwrites varValue with fractional values."""
    t_a = time.time()
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=cap_s, gapRel=0.005))
    wall = time.time() - t_a
    status = pulp.LpStatus[prob.status]
    has_inc = (prob.status in (pulp.LpStatusOptimal, pulp.LpStatusNotSolved)
               and all(v.varValue is not None for v in prob.variables()))
    inc = pulp.value(prob.objective) if has_inc else None
    plan = extract_plan(x, y) if has_inc else None
    # LP-relaxation bound (declared gap instrument; its time is NOT part of the
    # MPC wall clock -- it is measurement apparatus, not part of the policy)
    rel = prob.copy()
    for v in rel.variables():
        v.cat = "Continuous"
    rel.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=cap_s))
    lpb = pulp.value(rel.objective) if rel.status == pulp.LpStatusOptimal else None
    gap = None
    if inc is not None and lpb is not None and abs(lpb) > 1e-9:
        gap = max(0.0, (lpb - inc) / abs(lpb))
    return wall, status, inc, plan, lpb, gap

def make_mpc_policy(W_s, tier, solve_log, capture=None):
    """Closure implementing the rolling executor as a run_sim policy.
    Note: run_sim only calls the policy on slots with pending arrivals AND a free
    satellite, so a re-solve fires at the first decision opportunity on/after each
    R_SOLVE boundary (declared; boundaries stay on the fixed 60 s grid).
    Carried deficit D_i = -QU_i, the design's pinned identification: QU is updated
    by run_sim every slot, so no separate tracker can drift from it."""
    st = dict(plan={}, pending_plan=None, activate_at=-1, next_solve=0, t0_plan=0)

    def pol(pending, free_sats, b, omega, QE, QP, QU, V, rng, t):
        if t >= st["next_solve"]:
            if capture is not None:
                capture.append((t, b.copy(), -QU.copy()))
            prob, x, y, K = build_milp(t, b.copy(), -QU.copy(), W_s, M.ARRIVAL_PROB)
            wall, status, inc, newplan, lpb, gap = solve_window(prob, x, y, CAP[tier])
            solve_log.append([t, tier, W_s, round(wall, 2), status,
                              None if inc is None else round(inc, 4),
                              None if lpb is None else round(lpb, 4),
                              None if gap is None else round(gap, 4), len(x) + len(y)])
            if newplan is not None:
                if tier == "oracle":
                    st["plan"] = newplan; st["t0_plan"] = t
                    st["pending_plan"] = None; st["activate_at"] = -1
                else:
                    st["pending_plan"] = (newplan, t)
                    st["activate_at"] = t + int(np.ceil(min(wall, CAP[tier])))
            st["next_solve"] = (t // R_SOLVE + 1) * R_SOLVE
        if st["pending_plan"] and st["activate_at"] >= 0 and t >= st["activate_at"]:
            st["plan"], st["t0_plan"] = st["pending_plan"]
            st["pending_plan"] = None; st["activate_at"] = -1
        # execute: match a feasible action against the current macro-slot quota
        k = (t - st["t0_plan"]) // DELTA
        acts = M.feasible_actions(pending, free_sats, b, omega, t)
        cand = []
        for a in acts:
            key = (k, a[0], a[1], a[2], a[3])
            if st["plan"].get(key, 0) > 0:
                cand.append((a, key))
        if not cand:
            return None
        choice, key = cand[rng.integers(len(cand))]
        st["plan"][key] -= 1
        return choice
    return pol

def run_one(W_key, tier, seed, horizon=None):
    apply_primary()
    W_s = W_TIERS[W_key]
    solve_log = []
    pol = make_mpc_policy(W_s, tier, solve_log)
    t_a = time.time()
    r = M.run_sim(pol, V=10, seed=seed, horizon=horizon or M.HORIZON)
    wall = time.time() - t_a
    TOT = M.N_SAT * (horizon or M.HORIZON)
    row = [W_key, tier, seed,
           round(r["blackout_slots"] / TOT * 100, 3), round(r["min_fill"], 4),
           r["split_tasks"], round(r["total_quality"], 1),
           round(r["service_rate"], 4), r["pcap_violations"],
           len(solve_log),
           round(float(np.mean([s[3] for s in solve_log])), 2),
           round(float(np.max([s[3] for s in solve_log])), 2),
           sum(1 for s in solve_log if s[5] is None),
           round(wall, 1)]
    return row, solve_log

MPC_DIR = os.path.join(HERE, "..", "data", "mpc")
os.makedirs(MPC_DIR, exist_ok=True)
RUNS_HDR = ["W", "tier", "seed", "down_pct", "min_fill", "splits", "totQ",
            "service", "pkv", "n_solves", "solve_mean_s", "solve_max_s",
            "no_incumbent", "run_wall_s"]
SOLVE_HDR = ["t", "tier", "W_s", "wall_s", "status", "incumbent", "lp_bound", "gap", "n_int_vars"]

def append_csv(path, hdr, rows):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(hdr)
        w.writerows(rows)

def smoke():
    apply_primary()
    print("=== SMOKE: one solve per tier (t0=0, b=B_INIT, D=0) ===", flush=True)
    for wk, ws in W_TIERS.items():
        prob, x, y, K = build_milp(0, np.full(4, M.B_INIT), np.zeros(6), ws, M.ARRIVAL_PROB)
        nint = len(x) + len(y)
        wall, status, inc, plan, lpb, gap = solve_window(prob, x, y, 60)
        nact = sum((plan or {}).values())
        print(f"  W={wk:8s} ({ws}s, K={K}) int_vars={nint:6d} wall={wall:7.2f}s "
              f"status={status} inc={inc} lp={lpb} gap={gap} planned_actions={nact}", flush=True)

def timing_proposed():
    """Per-slot decision time of the proposed scheduler, same machine, same
    harness: wrap choose_lyapunov with a wall-clock timer inside run_sim."""
    apply_primary()
    times = []
    def timed(pending, free, b, omega, QE, QP, QU, V, rng, t):
        t_a = time.perf_counter()
        c = M.choose_lyapunov(pending, free, b, omega, QE, QP, QU, V, rng, t)
        times.append(time.perf_counter() - t_a)
        return c
    for sd in [7, 8, 9]:
        M.run_sim(timed, V=10, seed=sd)
    a = np.array(times) * 1e6
    out = dict(n=len(a), mean_us=float(a.mean()), median_us=float(np.median(a)),
               p95_us=float(np.percentile(a, 95)), max_us=float(a.max()))
    with open(os.path.join(MPC_DIR, "proposed_decision_time.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("PROPOSED per-slot decision:", out, flush=True)

if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "smoke":
        smoke()
    elif len(sys.argv) >= 2 and sys.argv[1] == "timing":
        timing_proposed()
    elif len(sys.argv) >= 2 and sys.argv[1] == "export-windows":
        # PRE-REGISTERED sampling (design doc): pool = seed-7 runs of BOTH
        # time-regimes per tier, inputs recovered by deterministic replay;
        # 10 windows = 8 stratified by measured wall (median of each octile)
        # + the 2 hardest in the pool.
        apply_primary()
        only = sys.argv[2] if len(sys.argv) >= 3 else None
        for wk, ws in W_TIERS.items():
            if only and wk != only:
                continue
            pool = []   # (wall, tier, t, b, D)
            for tier in ["realtime", "oracle"]:
                cap_list, slog = [], []
                pol = make_mpc_policy(ws, tier, slog, capture=cap_list)
                M.run_sim(pol, V=10, seed=7)
                assert len(cap_list) == len(slog)
                for (t, b, D), srow in zip(cap_list, slog):
                    assert srow[0] == t
                    pool.append((float(srow[3]), tier, t, b, D))
            pool.sort(key=lambda r: r[0])
            n = len(pool)
            picks = []
            for j in range(8):                       # median of each octile
                lo, hi = j * n // 8, (j + 1) * n // 8
                picks.append(pool[(lo + hi) // 2])
            picks += pool[-2:]                       # 2 hardest
            np.savez_compressed(
                os.path.join(MPC_DIR, f"windows_{wk}.npz"),
                wall_ws=np.array([p[0] for p in picks]),
                tier=np.array([p[1] for p in picks]),
                t0=np.array([p[2] for p in picks]),
                b=np.array([p[3] for p in picks]),
                D=np.array([p[4] for p in picks]), W_s=ws)
            print(f"exported 10/{n} windows for {wk}: "
                  f"walls {[round(p[0], 2) for p in picks]}", flush=True)
    elif len(sys.argv) >= 3 and sys.argv[1] == "solve-windows":
        # re-solve exported instances on THIS machine (run on Jetson for the
        # real-time criterion); prints per-solve wall times
        wk = sys.argv[2]
        apply_primary()
        d = np.load(os.path.join(MPC_DIR, f"windows_{wk}.npz"))
        walls = []
        for j in range(len(d["t0"])):
            prob, x, y, K = build_milp(int(d["t0"][j]), d["b"][j], d["D"][j],
                                       int(d["W_s"]), M.ARRIVAL_PROB)
            t_a = time.time()
            prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=60, gapRel=0.005))
            walls.append(time.time() - t_a)
            print(f"  {wk} win{j} t0={int(d['t0'][j])} wall={walls[-1]:.2f}s "
                  f"status={pulp.LpStatus[prob.status]}", flush=True)
        import platform
        print(f"SAMPLE[{platform.node()}] {wk}: mean {np.mean(walls):.2f}s max {np.max(walls):.2f}s", flush=True)
    elif len(sys.argv) >= 5 and sys.argv[1] == "mini":
        # executor validation: quarter-orbit run, nothing persisted
        wk, tier, seed = sys.argv[2], sys.argv[3], int(sys.argv[4])
        row, slog = run_one(wk, tier, seed, horizon=M.N_SLOTS_PER_ORBIT // 4)
        print("MINI", row, flush=True)
        for s in slog[:8]:
            print("  solve", s, flush=True)
    elif len(sys.argv) >= 5 and sys.argv[1] == "run":
        wk, tier, seed = sys.argv[2], sys.argv[3], int(sys.argv[4])
        row, slog = run_one(wk, tier, seed)
        # per-run files: concurrent runs must not race on a shared append
        tag = f"{wk}_{tier}_{seed}"
        append_csv(os.path.join(MPC_DIR, f"run_{tag}.csv"), RUNS_HDR, [row])
        append_csv(os.path.join(MPC_DIR, f"solves_{tag}.csv"), SOLVE_HDR, slog)
        print("RUN", row, flush=True)
    else:
        print(__doc__)
