# Night Run Log

Autonomous overnight session — goal: fill the **Performance Evaluation** section
(+ verify Conclusion) of the paper with results **really run on this machine**,
push to Overleaf, follow `EXPERIMENTS.md` priority. Iron rule: **no fabricated
numbers**; any experiment that fails is marked `% TODO: EXP-X failed ...` in the
paper, never filled with estimates.

Session start: 2026-06-26 (local).

---

## Environment / ground truth

- Python 3.12.2 (anaconda). numpy 1.26.4, pandas 2.2.2, matplotlib 3.9.2.
- PuLP installed for EXP-1 (bundles CBC). OR-Tools not present.
- Overleaf project `69d86443095882e8ced27a71` cloned for direct read/write
  (figures are binary; the text-only MCP `write_file` cannot upload PNGs).
- GitHub repo `leo-vlm-lyapunov` cloned locally; `EXPERIMENTS.md` copied in.

## Reproduced existing results (Track A grounding) — all re-run live

| Script | Scenario | Key real numbers (this machine) |
|--------|----------|---------------------------------|
| 01_single_sat_lyapunov | single sat, V sweep | 0 peak-viol all V; min-fill 0.182 (V=5) → 0.194 (V=200); QU_mean 34.20 → 32.48 |
| 02_single_sat_baselines | single sat, 5 policies | ours: 0 downtime/100% uptime, totQ 350.1; Greedy-Q: 2719 blackout slots |
| 03_multisat_baselines | 4 sats staggered eclipse | ours: 0 downtime, totQ 2069.9; Greedy-Q: **19058** blackout slots |
| 04_split_pipeline | 4 sats + 2-stage 8B split | ours: 0 downtime, **1100 split-8B tasks**, totQ 2378.4; baselines 0 split (capped at 7B) |

Calibration (`data/jetson_calibration.csv`, real Jetson Orin NX / EuroSAT / Q4):
3B → (T 1.32s, E 5.4J, P 6.6W, acc 0.30); 7B → (3.89s, 27.9J, 10.7W, 0.47);
8B → (8.66s, 62.8J, 11.0W, 0.51).

### Honest positioning carried into the writing (per EXPERIMENTS.md Part 5)
- Dependable win = **sustainability**: ours is the only policy with 0 blackout
  downtime in every multi-sat scenario; greedy-quality drains eclipse-side
  batteries (2719 / 19058 / 2438 blackout slots).
- **Unique capability**: only ours runs 8B, via the cross-satellite split pipeline.
- **Quality is a trade-off, not a blanket win**: ours does NOT top raw total
  quality. Reported honestly. Peak-power violations are 0 for *all* policies in
  the tested regimes (each config is individually below the cap), so the
  empirically separating safety axis here is battery blackout, not peak power;
  a peak-power-binding regime is left to the sensitivity sweep (EXP-6).

---

## Progress log

- [DONE] Track A: Performance Evaluation written from real results (calibration
  table, single-sat validation w/ Theorem 1/2/3 evidence, multi-sat + split
  comparison table, honest quality trade-off + peak-power caveat). 5 figures
  restyled to figures4papers house style and exported as vector PDF. Pushed to
  Overleaf (commit live) and GitHub. Conclusion already existed and is complete;
  left as-is (adding results there would break section discipline).
- [DONE] Track B / EXP-1: offline MILP optimality-gap curve (`src/05_milp_gap.py`,
  `data/results_milp_gap.csv`, `milp_gap.pdf`).
  - Setup: S=2 sats, I=2 sources, 48-slot scaled orbit; 3B/7B configs; PuLP+CBC.
    12 randomized instances (eclipse phase, init battery, solar noise), each solved
    to **proven optimum** (mean U* = 7.20). Online swept over V in [1..1000].
  - Result (mean over 12 instances): online min-quality / U* rises 0.46 -> 0.86;
    normalized gap falls 0.54 (V=1) -> 0.14 (V=500); 0 peak violations at all V.
  - Surprise / honesty: gap does NOT go to 0 — it plateaus ~0.14. The single-trace
    run was also slightly non-monotone at V=1000 (myopic over-commit). Averaging over
    instances smoothed it; the residual floor is the price of a causal scheduler vs a
    clairvoyant optimum over a short horizon. Reported as such; large-V follows O(1/V).
    Did NOT fake convergence to 0.
  - Updated belief: Theorem 2's O(1/V) trend holds empirically as a monotone gap
    shrinkage; the absolute gap floor is horizon/myopia-limited, worth a one-line
    caveat in the paper (added). Written into EVAL as subsection "Optimality Gap
    Against the Offline Benchmark" + Fig. milp_gap. Pushed to Overleaf + GitHub.
- [in progress] EXP-6 systematic sensitivity analysis (honest boundary map).
