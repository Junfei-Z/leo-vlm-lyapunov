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
- [DONE] EXP-6 systematic sensitivity analysis (`src/06_sensitivity.py`, `data/sens_*.csv`,
  `sensitivity_safety.pdf`, `sensitivity_quality.pdf`).
  - Parameterized, FAITHFUL port of src/04 with a reproduction guardrail: at default
    params it asserts ours blackout=0/split=1100/totQ=2378.4 and Greedy-Q blackout=2438.
    Guardrail printed "REPRODUCTION OK" — the port did not drift.
  - 7 one-at-a-time sweeps x 5 policies: n_sat, arrival_prob, B_max, P_solar, P_cap,
    sunlit_fraction, n_sources. Every point of every policy recorded honestly.
  - Honest boundary (the deliverable):
    * Invariant safety: ours = 0 blackout at EVERY swept point; baselines black out
      14k-24k slots under tight battery / low solar / severe eclipse / heavy load.
    * Peak shield is decisive only when the cap binds: at P_cap 8-10 W (< 7B 10.7 W,
      8B 11.0 W) Greedy-Q/Random/MaxBatt violate 16k-32k times; ours & Greedy-E = 0.
    * Advantage regime: many sats (8/16 -> ours leads min-fill 0.315/0.371 AND totQ
      3226/3915), and overload (ours protects worst source 0.066 vs ~0.001 for greedy).
    * Trade-off regime (reported plainly): few sats / light load / abundant resources
      -> baselines reach higher throughput; ours' safety margin not needed there.
- [DONE] EXP-7 eclipse stress (`src/07_eclipse_stress.py`, supports Theorem 3).
  Queues bounded across eclipse fraction 0.20-0.60 (0 viol, 0 blackout); Lyapunov
  backlog grows monotonically, fast-growth onset ~0.5 eclipse. Soft margin, no hard cliff.
- [DONE] EXP-8 long-horizon stability (`src/08_longhorizon.py`, supports Theorem 1).
  20 orbits: Q^E/T -> 3.7e-4, Q^P/T = 0 (mean-rate stable); Q^U/T plateaus ~5.7e-3
  because default load is mildly quality-overloaded (at the Slater boundary). Reported honestly.
- [DONE] EXP-3 dynamic ISL (`src/09_dynamic_isl.py`, supports C1 / Eq.16).
  Time-varying link duty cycle p_ISL: splits fall 1100 (p=1.0) -> 461 (p=0.3), so
  Eq.16 genuinely binds. Reroute to standalone keeps throughput, min-fill, and 0
  blackout intact (cost paid as fewer 8B inferences, not lost throughput). p=1.0
  reproduces the all-connected split count (internal check).

---

## Final state @ handoff (morning)

DELIVERED:
- Overleaf main.tex: Performance Evaluation fully written (Calibration, Setup,
  Single-Sat Validation, Optimality Gap vs MILP, Multi-Sat + Split comparison,
  Dynamic ISL, Sensitivity Analysis, Stability Stress Tests). 8 figures + 2 tables,
  all from REAL runs on this machine. Conclusion was already complete (left as-is).
- main.tex COMPILES CLEANLY locally (latexmk/pdflatex, MacTeX 2026): 17 pages,
  ZERO undefined references/citations, no multiply-defined labels (5 cosmetic
  overfull hboxes only).
- All code + CSV results + PDF/PNG figures committed and pushed to GitHub.
- Each section/experiment was committed and pushed as its own milestone.

NEEDS YOUR MANUAL FOLLOW-UP (require the Jetson — cannot run on this machine):
- EXP-4 (RESISC45 second dataset): rerun measurement/profile_jetson.py on RESISC45.
  C4 in the intro PROMISES EuroSAT + RESISC45; only EuroSAT is measured so far.
- EXP-5 (quantization sweep per model): profile each VLM at additional quant levels
  (Q2_K/Q4_K_M/Q8_0) to widen the config set; intro promises "multiple quantization".
  NOTE: until EXP-4/5 are done, the C4 sentence slightly over-promises vs the current
  evaluation — consider softening C4 or adding these two Jetson runs.

NOT ATTEMPTED (lower priority / time; not blocking):
- EXP-2 (offline ILP deployment stage y_sm): configs still hand-placed.
- EXP-9 (per-slot runtime/scalability plot): quick, supports the "lightweight" claim.
- EXP-10/11 (activation-compression, cislunar): paper future-work hooks only.

REPRODUCE EVERYTHING: each src/0X_*.py is one command from the repo root, fixed seed.
EXP-1 needs PuLP (installed). All figures use src/figstyle.py (figures4papers style),
exported as vector PDF.

---

# Session 2 — 2026-06-27 (night/paper-polish branch)

Goal shift: from "fill EVAL with real numbers" to "drive paper to submission-ready +
make experiments→claims traceable + readable". Iron rules unchanged: no fabricated
numbers; risky/subjective calls go to OPEN_QUESTIONS.md, not auto-applied.

## Iter 1 — claim↔evidence matrix + open questions (W1)
- DID: extracted C1–C4 from Intro; built `claim_evidence_matrix.md` mapping every
  contribution/sub-claim to its figure/table/theorem; ran orphan-figure + naked-claim checks.
- WHY: the matrix is the spine that tells the rest of the restructure what each figure is for.
- FOUND:
  - **C4 mismatch (HIGH)**: C4 says "RESISC45 + ViT-B/16…ViT-G/14", actual = Qwen2.5-VL/
    InternVL3 on EuroSAT only. → OPEN_QUESTIONS Q1 (NOT auto-edited; changes paper scope).
  - Fig 5 near-orphan: duplicates Fig 6's O(1/V) with a far weaker visual → merge/demote candidate.
  - 4-sat headline unflattering on quality → OPEN_QUESTIONS Q3.
- NEXT: W2 — draft up-front "Baselines" subsection (Greedy-Q/Greedy-E/Random/Static),
  rename MaxBatt-7B/Fixed-7B → Static throughout main.tex, add EVAL reading-guide paragraph.

## Pending background (not blocking)
- Jetson bench of gemma-3-4b / Qwen2-VL-2B / SmolVLM2 running; 6-config Table 1 + quality–
  energy Pareto deferred to W3 until `data/jetson_eurosat/results_*_summary.csv` land.
  gemma-4b smoke test PASSED (4 distinct answers, not collapsed like InternVL3-1B).

## Decisions locked (from user, this session)
- 4 baselines, consistent wherever a comparison is shown: Greedy-Q / Greedy-E / Random / **Static**.
- "Static" replaces MaxBatt-7B / Fixed-7B (config-agnostic; the fixed config may change once
  new bench data lands).
- EVAL must open with a dedicated Baselines subsection (introduce once, before any comparison fig).
- Every figure ships with its data CSV + a standalone plot script (user hand-tunes figures).

## Iter 2 — 6-config calibration + Pareto figure (W3) — bench finished early
- DID: all 3 Jetson benches finished (gemma-3-4b / Qwen2-VL-2B / SmolVLM2). All passed the
  no-collapse check (8–9 distinct answers). Synced to data/jetson_eurosat/. Built
  `src/build_calibration.py` → `data/jetson_eurosat/calibration_all.csv` (all 8 runs, N_im,
  Pareto flags) and `src/plot_tradeoff.py` → `figures/quality_energy_tradeoff.{pdf,png}`
  (data + standalone plot kept separate, per the figure-reproducibility rule).
- NUMBERS (acc / E_J): Qwen2-VL-2B 0.418/2.82, SmolVLM2 0.338/5.90, Qwen2.5-VL-3B 0.30/5.4,
  Gemma3-4B 0.342/41.0, Qwen2.5-VL-7B 0.47/27.9, InternVL3-8B 0.51/62.8.
- FOUND: Pareto frontier = {Qwen2-VL-2B, 7B, 8B}; 3B/SmolVLM2/Gemma3-4B are dominated;
  Qwen2-VL-2B beats the paper's current cheap anchor (3B) on BOTH accuracy and energy.
  → OPEN_QUESTIONS Q4. Did NOT edit Table 1 or re-run sims (changes calibrated cost rows).
- NEXT: W2 — Static rename + Baselines subsection + EVAL reading-guide in main.tex (Overleaf),
  none of which depends on the Table 1 decision.

## Prep for W2 (rename sites located, for next iteration)
- Paper (Overleaf snapshot lines): MaxBatt-7B at ~1036 (4-baseline intro sentence), ~1077,
  ~1079, ~1096 (tab_split row), ~1125, ~1129. Rename all "MaxBatt-7B" -> "Static".
- Code: `Fixed-7B` in src/02 (lines 15,141,156); `MaxBatt-7B` in src/03 (155,176),
  src/04 (216,232), src/06 (181,197,354 incl. PALETTE color key). Rename label strings ->
  "Static"; sims are fast + fixed-seed so re-run 03/04/06 to refresh figure legends
  (Fig 7 splitpipeline_comparison, Fig 9/10 sensitivity_*). NO numbers change — label only.
- OVERLEAF EDIT POLICY (decided): editing main.tex via overleaf MCP IS authorized (user
  explicit + Overleaf keeps its own version history = reversible). Proceed conservatively:
  do the mechanical Static rename first (user explicitly approved), then draft the new
  Baselines subsection + reading-guide; log every overleaf write_file/push in this file.
- W2 plan order: (1) read current main.tex setup/EVAL region via overleaf MCP to get exact
  strings; (2) rename MaxBatt-7B->Static in prose+table+captions; (3) insert Baselines
  subsection (Greedy-Q/Greedy-E/Random/Static, each: pick rule + ignored constraint +
  pathology) before first comparison fig; (4) add EVAL reading-guide paragraph (3 modes);
  (5) rename in code + re-run 03/04/06 to refresh legends; (6) compile-check; commit.

## Iter 3 — W2 text restructure (STAGED, not yet pushed to Overleaf)
- DID: verified my 2h-old main.tex snapshot is byte-identical to the live Overleaf file.
  Built the EVAL text edits with an asserted Python script (`/tmp/edit_maintex.py`) and
  inspected the full diff:
    (a) reading-guide paragraph after the EVAL intro (3 modes: compare / validate / stress-test);
    (b) pulled the baseline descriptions out of Simulation Setup into a new `\subsection{Baselines}`
        before the first comparison, describing Greedy-Q / Greedy-E / Random / Static (pick rule +
        ignored constraint + pathology); kept the 5-metrics sentence in Simulation Setup;
    (c) MaxBatt-7B → Static everywhere (5 prose/table/caption spots).
  Sanity: \end{document} present, figures 12→12, subsections 33→34, braces balanced, 0 MaxBatt-7B.
- WHY STAGED not pushed: overleaf MCP only does full-file write_file to a SERVER-SIDE mirror
  (probe file not findable locally); reproducing 93KB exactly through context is error-prone,
  and overwriting+pushing the live doc unattended (no compile check) is the night's riskiest op.
  Verified file staged at `paper_staging/main.tex` (+ `W2_eval_restructure.diff`). One morning
  step to apply (OPEN_QUESTIONS Q5). write_file does NOT auto-push, so nothing is live yet.
- ALSO: renamed MaxBatt-7B/Fixed-7B → Static in src/02,03,04,06 (code, safe). Figure legends
  (Fig 7/9/10) still show old name until PDFs regenerated — and figures aren't in the Overleaf
  project (MCP shows only main.tex), so upload is user-side. Flagged in Q5.
- NEXT: regenerate Fig 7/9/10 PDFs with the Static legend (run src/04, src/06); then W4 prose
  tightening (academic-writing) + W5 weak-figure fixes (Fig 5 redundant, Fig 4 needs baseline).

## User directives (mid-session, apply across W2/W4/W5)
1. SIMULATION ACTION SPACE = the Pareto-efficient measured configs only (currently {Qwen2-VL-2B,
   7B, 8B}; re-derive after InternVL2.5-4B + any prompt change). Each config is an opaque measured
   4-tuple (Q,E,T,Ppeak) — NOT parameterized by model size. The non-monotonicity of quality vs
   size (2B 0.418 > 3B 0.30 > 4B 0.342) is to be presented as a MOTIVATING finding in the
   calibration subsection: quality depends on architecture+generation, not scale, so configs must
   be empirically profiled and a profile-driven scheduler is needed (supports C2). Rewrite the old
   monotonic "3B→7B→8B costs 5.2× for 0.17" narrative accordingly. [awaiting final user OK on
   Pareto-only action space — do NOT re-run sims with new configs until confirmed; OPEN_QUESTIONS Q4]
2. EVALUATION STRUCTURE (from user's exemplar paper): (a) a dedicated "Benchmark Solutions"
   subsection describing each baseline as an algorithm (what it selects, what info it uses, why it
   fails) — our \subsection{Baselines} becomes this; (b) per-figure FOUR-PART discipline:
   setup line (swept param + fixed values) → figure ref + quantitative result per policy →
   MECHANISM as an explicit x→a→b→y causal chain (change x ⇒ system state a ⇒ method behaves b ⇒
   result y) → optional second observation+reason. Apply to every results paragraph in W4.
3. LINE-PLOT STYLE: all line figures (new AND existing: sim_V, milp_gap, dynamic_isl,
   eclipse_stress, longhorizon, sensitivity_*) use figstyle.py `apply_lineplot_style()` +
   `plot_series()`/`POLICY_STYLE` + `finalize_lineplot()` + `savefig_lines()` (big Times New Roman,
   thick lines, distinct per-series markers, PDF+EPS). Verify no text overlap, data not hugging the
   frame, legend not covering curves. Retrofit existing line figures in W5.

## PIVOT — task switch to RESISC45 + improved prompt (user-approved)
PROBLEM the user raised: on EuroSAT, Qwen2-VL-2B (0.418, 2.82J) nearly matches 8B (0.51) —
which destroys the paper's premise (high quality needs big models that must be split). Root
cause: EuroSAT (10-class) is too easy to discriminate model capability; NOT a reason to drop 2B
(cherry-picking). Fix: harder task that separates models, which ALSO fixes C4 (already claims RESISC45).

DIAGNOSTIC (2B vs 8B, improved prompt, 60 EuroSAT / 90 RESISC45):
  EuroSAT:  2B 0.467, 8B 0.567  (gap 0.100)
  RESISC45: 2B 0.378, 8B 0.556  (gap 0.178)  <-- gap widens ~78%; 8B robust to difficulty, 2B degrades.
  Improved prompt also lifts both (2B 0.418->0.467, 8B 0.51->0.567 on EuroSAT). 45-class numbers
  (0.38-0.56) read as more credible than 10-class 0.30-0.51.
DECISION (user): re-run ALL 7 models on RESISC45-450 with the improved prompt. Launched
  run_resisc.sh (bench_generic.py = energy-measuring bench + improved prompt + robust normalized
  parsing). ~4h. Monitor armed.
Also noted: InternVL2.5-4B EuroSAT full = 0.492 acc / 10.2J / 3.40s — beats 7B on BOTH acc and
  energy (more non-monotonicity evidence).
AFTER re-run: rebuild calibration_all.csv + Pareto from RESISC45 numbers, then decide action space
  + whether to re-run sims. The W2 text staging (Static/Baselines/reading-guide) is INDEPENDENT of
  this and still valid; calibration-narrative + Table 1 edits wait for the RESISC45 numbers.
PUSH STATUS: user authorized pushing to Overleaf, but MCP only does whole-file overwrite of a
  server-side mirror (can't reproduce 93KB losslessly by hand); gave user paste-ready blocks for
  the W2 changes instead. Calibration edits deferred to post-RESISC45.

## RESISC45 re-bench DONE — premise problem surfaced (autonomous run HALTED at sim re-run)
- DID: all 7 RESISC45 results synced to data/jetson_resisc45/; rebuilt calibration_all.csv;
  re-derived Pareto; plotted figures/quality_energy_tradeoff_RESISC45.pdf.
- RESULT (acc / E_J): 7B 0.573/9.1, InternVL2.5-4B 0.558/9.4, 8B 0.544/14.8, 3B 0.464/5.8,
  gemma-4b 0.513/37.1, 2B 0.398/3.9, SmolVLM2 0.327/5.3. Ranking now sensible (2B-anomaly gone).
- CRITICAL FINDING: Pareto = {2B, 3B, 7B}, all single-satellite. **InternVL3-8B is DOMINATED by
  Qwen2.5-VL-7B on BOTH acc and energy** (verified not an artifact). The split-pipeline premise
  ("largest model doesn't fit one satellite") collapses: best model (7B) fits standalone; 8B is
  neither best nor too big (8B-Q4 ~5GB fits 16GB). → OPEN_QUESTIONS Q0 (CRITICAL).
- DECISION: per HARD RULES, did NOT re-run sims with {2B,3B,7B} — that would silently gut C1's
  split-pipeline contribution. Pipeline steps 3-4 (action space + sim re-run) are BLOCKED on the
  user's Q0 decision (add a 32B+ model / re-motivate split / reposition). Notified the user.
- NEXT (non-blocked only until Q0 resolved): W5 retrofit existing line figures to figstyle
  line-plot style; W4 draft EVALUATION restructure text into paper_staging. NO sim re-run.
