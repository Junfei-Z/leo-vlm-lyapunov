# Evaluation Section — Structure Spec (authoritative)

Agreed plan for the Performance Evaluation section. The night run (W2/W4/W5) and the
extra sensitivity sweep build against this. Pairs with `claim_evidence_matrix.md`.

## Spine (narrative order)

**motivate → compare → validate → stress-test.** Each results paragraph follows the
four-part discipline (from the exemplar paper):
`setup line (swept param + fixed values) → figure ref + per-policy numbers → mechanism as an
explicit x→a→b→y causal chain → optional second observation`.
Example (Fig 9): tighten the power cap (x) ⇒ 7B/8B peak draw exceeds the cap (a) ⇒ baselines
without the per-slot shield are forced to violate (b) ⇒ violations spike to 16k–32k (y); the
proposed shield filters over-cap configs, so it never violates.

## Subsection outline

**§A. Simulation Setup** — orbital/constellation/slot params, solar model, parameter table.

**§B. Hardware Calibration & the Quality–Energy Landscape** — the new soul of the section.
- Table 1: 7 configs' measured (Q, E, T, P̂) on **RESISC45** (improved prompt).
- Quality–energy Pareto figure (house line-plot style).
- Two motivating observations: (i) quality is **non-monotone in model size** (architecture +
  generation matter as much as scale) ⇒ configs must be empirically profiled, a size/energy
  heuristic cannot pick the best ⇒ supports C2's offline calibration; (ii) the top-quality
  model (8B) does not fit on one satellite ⇒ motivates the split pipeline.
- Action space for the scheduler = the **Pareto-efficient** subset (dominated configs excluded;
  state this explicitly).

**§C. Benchmark Solutions** — introduce the 4 online baselines once, as algorithms
(Greedy-Q / Greedy-E / Random / Static: what each selects, which constraint it ignores, the
resulting pathology). **MILP caveat (must state):**
> We treat the offline MILP solely as a clairvoyant upper bound for the optimality-gap study
> (Section EV_GAP), not as a competing policy: it is non-causal—requiring full future knowledge
> of solar arrivals, channels, and eclipse—so comparing it on equal footing with causal online
> schedulers would be unfair. It is moreover intractable at constellation scale, which is why we
> solve it only on a small certified-optimal instance.

**§D. Simulation Results** — three signposted modes:

| Mode | Experiment | Figure/Table | Supports |
|---|---|---|---|
| **1 Compare** (all 4 baselines) | Multi-sat split-pipeline main comparison | Fig 7 + Table 2 | C1 capability, C4 safety |
| | Sensitivity — safety (blackout + peak) | Fig 9 (2×4, 8 sweeps) | C4 (strongest figure) |
| | Sensitivity — quality (min-fill) | Fig 10 (2×4, 8 sweeps) | C1 max-min (honest trade-off) |
| **2 Validate** (vs optimum/theory) | Single-sat time series + Greedy-Q overlay | Fig 4 | C3 ISS robust stability |
| | Optimality gap vs MILP (incl. V effect) | Fig 6 (absorb Fig 5) | C3 O(1/V) |
| | Long-horizon mean-rate stability | Fig 12 | C3 mean-rate |
| **3 Stress-test** (robustness) | Dynamic ISL availability | Fig 8 | C1 dynamic topology |
| | Eclipse-fraction stress | Fig 11 | C3 ISS soft margin |

## Figure roster decisions

- **Keep (7 core):** Table 1 + Pareto, Fig 7+Table 2, Fig 9, Fig 10, Fig 6, Fig 4, Fig 12.
- **Merge:** Fig 5 (V sweep) → into Fig 6 as (a)/(b); weak alone (min-fill moves only 0.012).
- **Group:** Fig 8 + Fig 11 together under stress-test.
- **Upgrade:** Fig 4 becomes a comparison — overlay at least Greedy-Q (the curve that collapses),
  per the rule "a comparison figure shows all four baselines"; Fig 4 shows ours stays bounded
  while greedy ratchets to blackout under the same periodic orbit.

## Sensitivity: 7 → 8 sweeps (clean 2×4)

Current 7: #satellites, arrival rate, battery capacity, solar power, peak-power cap, eclipse
fraction, #sources. **Add an 8th: per-source quality demand** — sweep the required min-quality;
directly stresses the max-min objective and is most favorable in the overload regime. Gives a
clean 2×4 grid for both Fig 9 and Fig 10. (Alt if trimming to 6: drop #sources — "limited impact".)
TODO: add this sweep in `src/06_sensitivity.py` once the RESISC45 action space is fixed.

## Dependencies / TODO order
1. RESISC45 7-model re-bench finishes → rebuild `calibration_all.csv` + Pareto → fix action space.
2. (If action space changed) re-run sims so Fig 4/6/7/8/9/10/11/12 reflect new configs.
3. Add quality-demand sweep to src/06 → regenerate Fig 9/10 as 2×4.
4. Apply text: §B/§C/§D restructure + MILP caveat + per-figure x→a→b→y (W4).
5. Re-render all line figures in the figstyle line-plot style (W5).
