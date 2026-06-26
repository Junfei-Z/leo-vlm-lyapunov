# Open Questions — decisions for the user (night/paper-polish)

Items the overnight run will NOT decide autonomously. Review in the morning.

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
