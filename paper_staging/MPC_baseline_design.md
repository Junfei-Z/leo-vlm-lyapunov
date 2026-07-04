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
- **Planning grid:** macro-slots of Δ = 30 s inside the window (W/Δ = 35 / 69 /
  191 steps), execution on real 1 s slots following the plan; a plan step that
  becomes infeasible at execution (arrival realization mismatch) is repaired by
  dropping that action (no re-optimization between re-solves).
- **Re-solve cadence:** R = 60 s (re-plan with fresh state), all tiers.
- **Solver:** CBC via PuLP (same as the offline benchmark), per-solve time
  limit 300 s wall; hitting the limit uses the incumbent and is logged (feeds
  pre-registration (b)).
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

## Deliverables
src/15_mpc_baseline.py (windowed MILP + rolling executor on the TLE overrides),
data/mpc_{W}.csv (quality + per-solve times), timing of proposed per-slot
decisions, and the filled reporting sentence for each tier.
