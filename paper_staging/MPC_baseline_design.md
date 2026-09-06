# MPC baseline design (P0-2) — declared BEFORE any run; for advisor approval

Standing pre-registered verdict (unchanged): **MPC may win in ALL regimes** —
the ephemeris is deterministic, so its forecasts are good; if it wins
everywhere, the differentiation claim is per-decision complexity + no forecast
machinery, quantified, not spun.

## The four pinned calibers (as agreed)
1. **Information disclosure.** MPC receives the SAME deterministic schedules the
   proposed scheduler can compute offline (sunlit trace, ISL availability,
   eclipse forecast — all TLE-derived) and the SAME causal observations
   (current battery, current arrivals, queue states). Future arrivals enter
   ONLY as the rate λ (expected-value terms in the window objective), never as
   realizations. No oracle beyond determinism.
2. **Guard exemption, shared hard constraints.** MPC does NOT run the
   eclipse-reserve guard or the virtual queues; it faces exactly the shared
   feasible set (split availability, ISL handoff check at t+N_front, per-slot
   energy e ≤ b+ω including P_base·τ, peak cap, memory). Battery safety must
   emerge from its own optimization, with one standard MPC device declared
   below (terminal condition), not from our guard.
3. **Window tiers, pre-declared:** W ∈ {half-eclipse ≈ 1035 s, one eclipse ≈
   2070 s, one orbit ≈ 5736 s} on the TLE mainline. All three reported; no
   post-hoc tier selection.
4. **Three pre-registered numbers:**
   (a) **Complexity line:** if MPC mean per-decision wall time ≥ 100× the
       proposed scheduler's per-slot decision time, the complexity
       differentiation claim stands as stated.
   (b) **Real-time criterion:** a solve exceeding τ = 1 s (the slot length)
       marks that W-tier "not per-slot real-time on-board"; we still report its
       quality (solved offline-style), labeled as such.
   (c) **Reporting sentence template (pre-written, filled with numbers only):**
       "MPC with window W attains min-fill X and downtime Y at Z× the
       per-decision compute of the proposed scheduler; the proposed scheduler
       attains P% of MPC's min-fill at O(1) per-slot cost with no forecast
       machinery." Used verbatim whether MPC wins or loses.

## Declared design details (the degrees of freedom, fixed now)
- **Formulation:** windowed MILP mirroring the offline benchmark (auxiliary
  max-min variable U over the window's cumulative per-source fill, carried-in
  deficits included so fairness is long-run, not window-local), same action
  space as the shared feasible set.
- **Terminal battery condition:** b(t+W) ≥ platform energy needed from t+W to
  the next sunrise (computable from the deterministic schedule). This is the
  standard MPC end-of-horizon device and uses only information MPC is entitled
  to; without it any finite window drains the battery at the horizon edge (a
  strawman we decline to build). Declared, not tuned.
- **Battery lower bound is SOFT (amendment declared before any run, after the
  smoke test exposed it):** the trace starts in eclipse and B_init cannot carry
  the platform to sunrise (sat0 shortfall 3478 J), so a hard b ≥ 0 makes every
  blackout-doomed window infeasible and MPC would get NO plan precisely where
  planning matters. The simulator gives every policy a recourse (safe-mode
  load-shed + recharge); the MILP mirrors it as a slack z ≥ 0 on the battery
  dynamics and terminal condition with penalty 10 quality-units/J — far above
  the best quality-per-Joule of any action (0.9/3.93 ≈ 0.23), so slack is never
  traded for inference; it only absorbs physically unavoidable shortfall.
  Declared, not tuned.
  **The relaxation exists ONLY in the planning layer.** Execution is the same
  for every policy: the MPC executor is a policy closure inside the SAME
  run_sim harness; every candidate action passes the SHARED feasible_actions()
  oracle (per-slot energy incl. P_base, peak cap, ISL/handoff, visibility) and
  the plan quota can only narrow that set, never widen it; battery depletion
  triggers the same safe-mode blackout with the same downtime accounting as
  all baselines. The MILP slack never reaches execution — plan extraction
  yields action counts only, and the simulator's battery is its own state.
  (Confirmed in code before any result was read; MPC runs log nonzero downtime
  through exactly this path.)
- **Planning grid:** macro-slots of Δ = 30 s inside the window (W/Δ = 35 / 69 /
  191 steps), execution on real 1 s slots following the plan; a plan step that
  becomes infeasible at execution (arrival realization mismatch) is repaired by
  dropping that action (no re-optimization between re-solves).
- **Re-solve cadence:** R = 60 s (re-plan with fresh state), all tiers.
- **Solver budget, TWO DECLARED VARIANTS (fix 1):**
  * **Real-time tier:** per-solve cap = the re-plan interval (60 s); on timeout
    the CBC incumbent (best feasible found) is executed. This is the standard
    rolling-horizon engineering form and the fair opponent: no solve may act on
    stale state.
  * **Oracle tier:** solve time NOT charged to the simulation clock (unlimited
    compute assumption; wall cap 300 s per solve for practicality), reported
    separately as the performance UPPER BOUND. The gap between the two tiers is
    the complexity claim made visible.
  Both variants run for every W; prior (a) "MPC wins" is adjudicated on the
  oracle tier, prior (b) "not real-time" on the real-time tier. CBC relative
  MIP-gap tolerance 0.5% on both tiers (declared; the smoke test showed CBC
  reaching gap < 1e-6 well inside the cap and then burning the remaining budget
  proving optimality — the reported per-solve time is "time to a usable plan").
  The tolerance goes into the paper's MPC description verbatim: "solved to
  0.5% MIP gap".
- **Carried-in deficit, exact form (fix 2):** the window objective is
  max U subject to U ≤ D_i(t) + Σ_{window} q_i, for every source i, where
  D_i(t) is the source's cumulative delivered quality up to t minus its
  cumulative target (arrivals × U_tgt), i.e. the negative of our QU queue
  content. MPC thus carries cross-window fairness memory equivalent to the
  proposed scheduler's QU state; a window-local max-min (no D_i) is explicitly
  rejected as a strawman.
- **W = orbit variable account (fix 3):** 191 macro-slots × (48 standalone +
  72 split-pair) binaries ≈ 23k binaries + 4×191 continuous battery variables.
  Pre-registered usability criterion: if on the real-time tier the incumbent is
  missing or its optimality gap exceeds 50% on more than 20% of re-solves, the
  tier is declared "not usable at the real-time budget" and reported as such
  (that outcome is itself evidence, not a discard).
- **Timing venues:** the sweep runs on the workstation (trend numbers); a
  sample of ≥10 windows per tier is re-solved on the Jetson for the real-time
  criterion (deployment-consistent). Both reported.
- **Jetson sampling rule (pre-registered BEFORE the matrix finished, after the
  heavy-tail discovery):** per tier, 10 windows = 8 stratified + 2 hardest.
  Pool = the seed-7 runs of both time-regimes for that tier (their exact solve
  inputs recovered by deterministic replay). Stratified 8: sort the pool by
  measured wall time, split into 8 equal-count strata, take the median-wall
  instance of each (deterministic, no selection freedom). Hardest 2: the top-2
  wall times in the pool (the eclipse-tier 300 s instance is in by construction).
  Random-only sampling would measure the easy body and miss the tail that
  decides the real-time verdict; post-hoc selection would be cherry-picking.
- **Solve-time reporting is distribution-first (upgraded after the heavy-tail
  discovery, before the matrix finished):** per tier median / p95 / max + the
  FRACTION of solves over τ = 1 s (prior (b) is adjudicated on this fraction,
  not on a single max). Hard windows (wall > τ) are phase-tagged with their
  distance to the nearest eclipse boundary; the boundary-clustering sentence
  enters the paper ONLY if the hard-window median distance is clearly below
  the all-solve median.
- **Runs:** TLE mainline primary config (rule sizes, measured energies,
  h = 0.69), seeds 7–16 where runtime allows; if a tier's full 10-seed cost is
  prohibitive, report fewer seeds for that tier EXPLICITLY (never silently).
- **Timing comparison:** proposed per-slot decision time measured on the same
  machine in the same run harness (μs–ms expected); MPC per-decision time =
  wall time per re-solve amortized per executed slot AND per-solve raw; both
  reported.

## What we expect to happen (prior, written down)
MPC with W = one orbit should dominate min-fill and possibly downtime (it sees
the whole eclipse); W = half-eclipse may struggle with the terminal condition
binding early. Solve times: 191-step MILP with 4 sats and split pairing will
likely exceed τ = 1 s per solve by orders of magnitude → pre-registration (a)
likely triggers; (b) likely marks W = orbit non-real-time. If instead CBC
solves in < 1 s, the complexity claim weakens and we report that honestly.

## Reading-stage amendments (logged per discipline)
- **+0.1pp downtime margin in the "spends safety margin" pattern check
  (2026-07-04, measured-Q_SRC reading):** run CSVs round down_pct to 3
  decimals and the start-in-eclipse blackout transient is identical across
  policies, so a 4th-decimal excess is rounding, not spent margin; 0.1pp is
  the minimum credible difference at that storage granularity. Scope: this
  threshold affects ONLY the pattern check (a limitation-section input); no
  pre-registered main adjudication (complexity line, τ fraction, usability)
  uses it or any downtime comparison.

## Deliverables
src/15_mpc_baseline.py (windowed MILP + rolling executor on the TLE overrides),
data/mpc_{W}.csv (quality + per-solve times), timing of proposed per-slot
decisions, and the filled reporting sentence for each tier.
