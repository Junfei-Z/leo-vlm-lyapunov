# Claim ↔ Evidence Traceability Matrix

Goal: every contribution maps to concrete evidence (figure/table/theorem), and every
figure/table serves at least one claim. Surfaces orphan figures (no claim) and naked
claims (no evidence). Built 2026-06-27 (night/paper-polish).

## Contributions (from Intro, verbatim labels)

- **C1** — First *formulation* of energy-aware collaborative VLM inference over solar LEO,
  jointly capturing: (a) two-stage pipeline splitting, (b) instantaneous peak-power caps,
  (c) orbital eclipse dynamics, (d) dynamic ISL topology, (e) max-min quality fairness.
- **C2** — Two-timescale robust Lyapunov framework: offline ILP for deployment + lightweight
  per-slot online scheduler, with calibrated power bounds making peak feasibility checkable.
- **C3** — Theoretical guarantees: O(1/V) optimality gap, mean-rate constraint satisfaction
  (Slater), robust stability during eclipse via ISS, extended to coupled energy–power queues.
- **C4** — Empirical validation, showing zero peak-power violations while keeping min-quality
  within a small gap of the offline MILP, vs baselines that ignore ≥1 of the four constraints.

## Theory inventory

| Theorem | Statement | Validated by |
|---|---|---|
| thm_robust | ISS robust stability through zero-harvest eclipse | Fig 4, Fig 11 |
| thm_gap | O(1/V) optimality gap | Fig 5, Fig 6 |
| thm_stable | mean-rate stability | Fig 12 |

## Matrix: contribution → evidence

| Claim | Sub-claim | Evidence | What it shows | Status |
|---|---|---|---|---|
| C1 | (a) pipeline split | Tab 2, Fig 7 | only proposed runs 8B as 2-sat split (1100 inf) | ✅ |
| C1 | (b) peak-power cap | Fig 9 (safety sweep) | peak shield: 0 viol vs baselines 16k–32k when cap tightened | ✅ |
| C1 | (c) eclipse dynamics | Fig 4, Fig 11 | queues bounded through eclipse; graceful degrade w/ eclipse frac | ✅ |
| C1 | (d) dynamic ISL | Fig 8 | handoff constraint binds (1100→461 splits), graceful reroute | ✅ |
| C1 | (e) max-min fairness | Fig 10, Tab 2 (min-fill) | advantage grows w/ constellation & overload | ⚠️ weak at 4-sat headline (see OQ) |
| C2 | offline ILP deploy | §Solution + Fig 6 | offline MILP benchmark solved to optimum | ⚠️ ILP-deploy itself not ablated |
| C2 | calibrated power bound | Tab 1, Fig 9 | measured Ppeak (95p) → closed-form per-slot check | ✅ |
| C2 | online scheduler | Fig 4,5,7 | per-slot drift-plus-penalty operation | ✅ |
| C3 | O(1/V) gap | Fig 6 (strong), Fig 5 (weak) | gap 0.54→0.14 vs V; tracks O(1/V) ref | ✅ (Fig 5 redundant) |
| C3 | mean-rate stability | Fig 12 | Q^E/T→3.7e-4, Q^P/T≡0 over 20 orbits | ✅ |
| C3 | ISS robust stability | Fig 4, Fig 11 | bounded under recurring eclipse disturbance | ✅ |
| C4 | zero peak violations | Tab 2, Fig 9 | proposed = 0 everywhere | ✅ |
| C4 | min-quality near MILP | Fig 6 | within 0.14 of offline optimum at large V | ✅ |
| C4 | vs constraint-ignoring baselines | Tab 2, Fig 7/9/10 | full baseline comparison | ✅ |
| C4 | **models: ViT-B/16…ViT-G/14** | — | **NO EVIDENCE — actual = Qwen2.5-VL/InternVL3** | ❌ MISMATCH |
| C4 | **dataset: RESISC45** | — | **NO EVIDENCE — only EuroSAT run** | ❌ MISMATCH |

## Orphan-figure check (every figure serves a claim)

All 9 figures map to ≥1 claim. Fig 5 is the only near-orphan: it duplicates Fig 6's
O(1/V) message with a much weaker visual (min-fill moves only 0.182→0.194). Candidate
for merge into Fig 6 or demotion to a sentence.

## Naked-claim check (every claim has evidence)

- C4 model line (ViT-B/16…ViT-G/14) and dataset line (RESISC45): **naked — no experiment exists.** → OPEN_QUESTIONS Q1.
- C2 "offline ILP for deployment": the ILP *deployment* decision is not independently
  ablated; Fig 6 tests the online gap against an offline MILP, which is related but not the
  same as showing the ILP placement helps. Minor → OPEN_QUESTIONS Q2.
