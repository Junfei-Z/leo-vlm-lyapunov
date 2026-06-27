# Open Questions — decisions for the user (night/paper-polish)

Items the overnight run will NOT decide autonomously. Review in the morning.

## Q6 — [IMPORTANT] RESISC45 configs weaken two headline claims; reframe needed

After re-running the sims with the RESISC45 8GB-satellite configs, two headline stories are
materially weaker (the data is honest; the old framing was tuned to the expensive 8B numbers):

1. **Zero-blackout-at-default is gone.** At the 4-sat split default NO policy blacks out now
   (Greedy-Q downtime 2438→0). The RESISC45 configs are cheap (≤9J), so the default constellation
   isn't energy-stressed. The safety advantage still holds, but only under TIGHTER sweeps (1kJ
   battery, 4W solar, 70% load) — see sensitivity Fig 9. **Recommendation:** stop claiming the
   blackout win in the headline Table 2 prose; move it to the sensitivity section ("the advantage
   emerges as resources tighten"), which is more honest and still strong.

2. **Peak-power shield is thin.** All measured peak draws are ≤5.57W and close together, so the
   cap only binds in a narrow 4.0–4.5W window (vs the old 8–10W). It still works (baselines violate
   ~16k times at cap≤4.5W, ours 0), but it's less dramatic. **Also suspect:** RESISC45 power may be
   under-read — `bench_generic.py` did not lock the Jetson power mode/clocks (RESISC45 energy was ~4×
   lower than EuroSAT for the same 8B). **Options:** (a) re-measure with locked MAXN power mode for
   clean peak numbers; (b) keep as-is and present the peak shield as a "by-construction guarantee
   that binds under tight caps" rather than a headline empirical win; (c) de-emphasize C2's peak axis.

3. The cumulative-quality trade-off is unchanged (ours competitive, not the raw-throughput leader);
   the win remains "only policy that runs the top 7B tier via split (3755 splits)."

These reframings change the EVALUATION prose emphasis. I did NOT rewrite the narrative — flagging
for your call. The numbers themselves are in data/sens_*.csv and the regenerated figures.

## Q0 — [RESOLVED 2026-06-27] premise restored via realistic 8GB-satellite memory model

USER DECISION: do NOT add a giant model and do NOT drop the split. Instead re-ground the
**satellite** memory budget at a realistic ~8GB (typical edge Jetson), of which the OS + I/O +
the on-board sensing/detection task consume a large share, leaving only ~5GB for the VLM. Under
this budget, **any model ≥4B cannot run standalone on one satellite and must be split across two**.
This restores the split-pipeline premise honestly AND requires no new model — the Pareto-best
Qwen2.5-VL-7B becomes the split-only top tier.

ACTION SPACE (RESISC45, 3 configs, mirrors original 2-standalone + 1-split structure):
- Qwen2-VL-2B   (Q 0.398) — standalone
- Qwen2.5-VL-3B (Q 0.464) — standalone
- Qwen2.5-VL-7B (Q 0.573) — **split-only** (≥4B doesn't fit one 8GB satellite w/ sensing overhead)
Dropped: InternVL3-8B (dominated + unreliable on test HW), InternVL2.5-4B / gemma-4b / SmolVLM2
(Pareto-dominated). Use RESISC45 numbers from data/jetson_resisc45/calibration_all.csv.

PAPER must distinguish: calibration measured on the 16GB Orin NX (measurement platform) vs the
SIMULATED satellite assumed at 8GB shared with the sensing task (deployment constraint) → ≥4B splits.
This UNBLOCKS the sim re-run (pipeline steps 3-4). The old Q0 problem write-up is kept below for record.

### (original problem, for record) RESISC45 breaks the split-pipeline premise

The RESISC45 re-bench (all 7 models, improved prompt, `data/jetson_resisc45/`,
`figures/quality_energy_tradeoff_RESISC45.pdf`) fixed the "2B too good" problem — the ranking is
now sensible — but it broke the paper's CORE premise a different way:

| config | RESISC45 acc | E (J) | Pareto? |
|---|---|---|---|
| Qwen2.5-VL-7B | **0.573** | 9.1 | ✅ best |
| InternVL2.5-4B | 0.558 | 9.4 | dominated |
| **InternVL3-8B** | **0.544** | 14.8 | **dominated** |
| Qwen2.5-VL-3B | 0.464 | 5.8 | ✅ |
| gemma-3-4b | 0.513 | 37.1 | dominated |
| Qwen2-VL-2B | 0.398 | 3.9 | ✅ |
| SmolVLM2-2.2B | 0.327 | 5.3 | dominated |

**Pareto frontier = {Qwen2-VL-2B, Qwen2.5-VL-3B, Qwen2.5-VL-7B}** — ALL fit on one satellite.
**InternVL3-8B is Pareto-dominated**: Qwen2.5-VL-7B beats it on BOTH accuracy (0.573>0.544) and
energy (9.1<14.8). Verified it is NOT a parsing/prompt artifact (8B: 97% parsed, 47 distinct
answers, sensible outputs) — InternVL3-8B is genuinely a bit worse than Qwen2.5-VL-7B here.

**Why this is fatal for the current framing:** the split pipeline exists to run "the largest model,
which does not fit on one satellite." But (a) the best model now is 7B, which fits standalone, and
(b) the 8B is neither best nor too big — InternVL3-8B-Q4 is ~5 GB and already ran standalone in
16 GB. So no model is simultaneously top-quality AND too big → a rational scheduler never splits.
(Note: the "8B doesn't fit in 16 GB" claim was already thin even on EuroSAT.)

**I did NOT re-run the sims** — proceeding with action space {2B,3B,7B} would silently gut the
split-pipeline contribution (C1). Options (need your call):
- **(A) Add a genuinely large top model (recommended, principled).** Scale up the current winner's
  family — Qwen2.5-VL-32B or -72B (Q4 ~18–40 GB) — which would be the most accurate AND genuinely
  exceed one Orin's 16 GB → must split. Caveat: with a single Jetson you can't run it standalone;
  you'd measure accuracy via CPU/RAM offload (slow) and cost via the two-stage split estimate.
  Needs a big download (network was flaky tonight — several ssh timeouts) + a slow run.
- **(B) Re-motivate the split (lighter).** Frame the split around peak-power halving / load
  balancing / latency across two satellites under a tight power cap, rather than "doesn't fit."
  Keeps the current models; rewrites C1's framing.
- **(C) Reposition the paper** away from "enabling the largest VLM" toward energy-aware eclipse
  scheduling; de-emphasize the split pipeline. Largest change.

**My recommendation: (A)** if you can spare the download + slow run (most honest, keeps the
contribution); **(B)** as the fast fallback if you want to avoid more Jetson work. Until you decide,
the autonomous run will do only non-blocked work (figure-style retrofit, text drafts) and NOT touch
the sims/action space. Also note: RESISC45 energy numbers differ ~4× from the old EuroSAT ones
(bench_generic did not lock power mode/clocks); they are internally consistent so usable, but
absolute energy is setup-specific — worth re-confirming the measurement setup before finalizing.

## Q5 — [ACTION] Apply staged main.tex to Overleaf + figure sync (one morning step)

W2 text edits are DONE and verified but **not yet pushed to Overleaf** — the MCP only does
full-file `write_file` to a server-side mirror, which I judged too risky to overwrite + push
unattended on the live doc. The complete, verified file is staged at **`paper_staging/main.tex`**
(diff: `paper_staging/W2_eval_restructure.diff`). Changes: reading-guide paragraph, new
`\subsection{Baselines}` (all 4 incl. Static), MaxBatt-7B→Static, metrics sentence kept.
**To apply (with me, in the morning):** `mcp__overleaf__write_file(main.tex, <staged content>)`
→ `read_file` verify → `push_changes`, then a compile check.

Two sync gaps to resolve when applying:
- **Figures aren't in the Overleaf project** (MCP `list_files` shows only `main.tex`). How do
  your figure PDFs reach the Overleaf compile — manual upload? a separate sync? I renamed the
  baseline label to "Static" in `src/02–06` but the figure legends (Fig 7/9/10:
  splitpipeline_comparison, sensitivity_safety/quality) still show the old name until the PDFs
  are regenerated AND uploaded. I can regenerate the PDFs locally; uploading is your side.
- **Stray file**: my mirror probe left `_probe_night.txt` in the Overleaf mirror (NOT pushed,
  so not live). The MCP has no delete — remove it via the Overleaf UI before/after the push,
  or ignore (it's an empty placeholder).

## Q1 — [HIGH] C4 describes models/datasets that were never run

**Contribution C4** (Intro) claims empirical validation:
> "on EuroSAT and **RESISC45** with **ViT-B/16 through ViT-G/14** at multiple quantization levels"

But the actual evaluation (Table 1 calibration + all sim figures) uses:
- Models: **Qwen2.5-VL-3B, Qwen2.5-VL-7B, InternVL3-8B** (VLMs, not ViT classifiers),
  soon also gemma-3-4b / Qwen2-VL-2B / SmolVLM2-2.2B once the Jetson bench finishes.
- Dataset: **EuroSAT only** (no RESISC45 run exists).

This is a real claim↔evidence mismatch a reviewer will catch immediately.
**Options:**
- (A) Rewrite C4 to match what was actually run (VLMs on EuroSAT). Cheapest, honest.
- (B) Actually run RESISC45 + a ViT-family sweep on the Jetson to back the current C4.
  Large extra effort; RESISC45 is a different (45-class) benchmark.
- (C) Keep VLMs but drop the specific "ViT-B/16…ViT-G/14" and "RESISC45" wording.

**Recommendation: (A)/(C).** I did NOT edit C4 — it changes the paper's stated scope.
The night run will align the rest of the text to the *actual* experiments and leave C4's
exact wording for you. (Note: ViT-B/16…ViT-G/14 may be left over from an earlier draft.)

## Q2 — [LOW] Offline ILP deployment not independently ablated

C2 credits an offline ILP for VLM *deployment/placement*. Fig 6 tests the online scheduler's
gap vs an offline MILP, which is related but does not isolate the value of the ILP placement
decision. Is a small placement-ablation worth adding, or is C2 fine as a framework-description
claim (no ablation expected)? Default: leave as-is unless you want the ablation.

## Q3 — [MED] 4-satellite headline scenario is unflattering on quality

In the main comparison (Table 2, Fig 7) the proposed scheduler has LOWER total quality
(2378) than MaxBatt/Static-7B (3751) and even lower min-fill (0.231) than Random (0.298).
The quality advantage only appears at 8/16 satellites (Fig 10). The win at 4 sats is purely
"zero blackout + only one to run 8B."
**Question:** keep 4-sat as headline (honest, but reviewers anchor on the table), or promote
an 8-sat scenario to the headline where the method also wins on quality?
This changes the story's framing → left for you. (Raised earlier in conversation.)

## Q4 — [MED] New bench data: Qwen2-VL-2B Pareto-dominates the 3B; which configs go in Table 1?

All 6 configs benchmarked on EuroSAT (see `data/jetson_eurosat/calibration_all.csv`,
`figures/quality_energy_tradeoff.pdf`). Surprise result:

| Config | Q | E (J) | note |
|---|---|---|---|
| **Qwen2-VL-2B** | **0.418** | **2.82** | Pareto — cheapest AND more accurate than 3B/4B |
| Qwen2.5-VL-7B | 0.470 | 27.9 | Pareto |
| InternVL3-8B | 0.510 | 62.8 | Pareto |
| Qwen2.5-VL-3B | 0.300 | 5.4 | dominated by 2B |
| SmolVLM2-2.2B | 0.338 | 5.9 | dominated by 2B |
| Gemma3-4B | 0.342 | 41.0 | dominated (slow: 11.3 s/img) |
| InternVL3-1B(×2) | 0.10 | — | failed (collapsed to one class) |

**Implication:** the Pareto frontier (the scheduler's *useful* action space) is exactly
**{Qwen2-VL-2B, Qwen2.5-VL-7B, InternVL3-8B}**. The current paper Table 1 uses
{Qwen2.5-VL-3B, 7B, 8B} — i.e. its cheap anchor (3B) is now Pareto-dominated by Qwen2-VL-2B.

**Options (NOT auto-applied — changes the calibrated numbers driving every sim):**
- (A) Replace 3B → Qwen2-VL-2B in Table 1 (3 clean Pareto points; strongest curve). Requires
  re-running all sims with the new cost row. Recommended if you want the tightest story.
- (B) Keep 3B/7B/8B as-is (already simulated; conservative).
- (C) Show all 6 in a calibration figure (quality_energy_tradeoff.pdf) but drive the sim with
  the 3 Pareto points — lets you *use* the dominated points to motivate "why a scheduler must
  avoid bad configs".
**I did NOT touch Table 1 or re-run sims.** The night run will keep the existing 3B/7B/8B sim
results intact and leave this swap for you. Note: the non-monotonic Q-vs-size (2B > 3B > 4B on
accuracy) slightly complicates any "bigger = better" wording in the text.
