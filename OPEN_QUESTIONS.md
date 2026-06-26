# Open Questions — decisions for the user (night/paper-polish)

Items the overnight run will NOT decide autonomously. Review in the morning.

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
