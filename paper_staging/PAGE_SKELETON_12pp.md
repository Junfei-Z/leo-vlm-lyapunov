# 12-page skeleton plan (TMC regular-paper quota, incl. references)

Current: 17 pp. Target: 12 pp. Cut ≈ 5 pp. New sections still to land: MPC baseline
(+0.5), 45-source extension (+0.5), V guideline (+0.25) → effective cut ≈ 6.25 pp.

## Per-section quotas (target)
| Section | now (est) | quota | how |
|---|---|---|---|
| Abstract + Intro + C1-C4 | 2.0 | 1.75 | trim intro constraint prose (the four-constraint story is re-told in §II/§III; one telling) |
| Related work + Table I | 1.5 | 1.0 | compress per-area paragraphs to positioning sentences; Table I keep |
| System model + formulation | 3.5 | 2.25 | notation Table II → supplementary (keep a 10-row inline core); comm-model derivation compress; constraint prose tighten |
| Algorithm + analysis | 2.5 | 1.5 | **three full proofs → supplementary**, keep statements + 2-3-line sketches; block diagram shrink |
| Evaluation | 6.0 | 4.5 | see below |
| Conclusion | 0.5 | 0.4 | keep |
| References | 1.0 | 0.6 | prune to cited-in-text only |
| **Total** | 17 | **12.0** | |

## Evaluation internal quotas (4.5 pp incl. new sections)
| Subsection | quota | notes |
|---|---|---|
| CALIB + PEAK + HANDOFF (measurement block) | 1.1 | identity block, keep dense; nine-tier full table → supplementary (prose keeps the two fitted facts) |
| SETUP | 0.5 | trace pipeline details (plane selection mechanics) → supplementary |
| MAINLINE + Table IV | 0.5 | keep |
| SENS + fig | 0.6 | keep 2×4 figure; prose tighten |
| ISL + STAB + figs | 0.7 | keep (linear-coupling + ISS reading are load-bearing) |
| ROUTE | 0.35 | compress pre-registration procedural detail → supplementary |
| CONTRAST + 2 figs | 0.5 | frontier fig may shrink to 0.8\columnwidth; nsat sentence stays |
| GAP + 2 figs | 0.25 | **candidate: merge figs 2-in-1 or move sim_V fig → supplementary** |
| MPC (new) | 0.5 | write to quota directly |
| 45-source + V guideline (new) | 0.5 | write to quota directly |

## Supplementary material list
1. Full proofs of Theorems 1-3 (+ Lemma 1 algebra).
2. Notation table (full version).
3. TLE trace pipeline: selection rules, physical gate, chain diagnostics.
4. Handoff nine-tier full table + per-tier CSVs pointer + 56-RT fit details.
5. Peripheral sweep full tables (tleP CSVs) + synthetic-contrast full numbers.
6. MILP toy instance details + sim_V single-satellite figure (if cut).
7. nsat-under-TLE (16-chain) — future/optional, listed as extension.
8. Idle-decomposition instrument description.

## Order of operations
Compression happens AFTER MPC/45-source/V-guideline text lands (write those to
quota directly), in ONE dedicated compression pass with a fresh compile + gates
(undefined + multiply-defined + O(1/V) qualifier + old-number sweep).
