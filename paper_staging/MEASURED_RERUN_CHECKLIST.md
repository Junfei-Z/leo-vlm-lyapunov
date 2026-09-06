# Measured-Q_SRC global rerun checklist (option 1) — 2026-07-04

Single change: src/04 Q_SRC now loads from data/q_src45.json (measured
per-class, n=1300). Everything downstream of Q_SRC re-runs; nothing else
moves. Consistency gate: src/14 core Proposed(V=10) must reproduce
anchor6_core (down 1.6% / mf 0.165 / sp 673 / tq 2151) — same config by
construction.

## Q_SRC-INDEPENDENT (verified, no rerun)
- src/01 single-sat sim_V figure (uses config-Q only, homogeneous)
- src/05 MILP gap figure (local q_im = m["Q"], homogeneous)
- CALIB / PEAK / HANDOFF measurement identities (hardware, no Q_SRC)
- TLE trace generation (src/10), sizing rules, h=0.69 derivation

## RERUNS (batch launched)
| what | script | outputs | feeds |
|---|---|---|---|
| TLE core (Table IV) | 14 core | tleP_core.csv | MAINLINE table + abstract/intro/conclusion numbers |
| TLE pbase/battery/panel/arrival | 14 <tag> | tleP_<tag>.csv | SENS fig + prose (feasibility boundary, partition band, Random takeover point) |
| TLE ISL degradation | 14 isl | tleP_isl.csv | ISL fig + prose: 0.020/100-splits slope, 655→333, mountain concentration — RE-VERIFY under measured values (7B-dependent-source definition changed; slope and concentration may move; report new values) |
| TLE stability 20-orbit | 14 stability | tleP_stability.npz | STAB fig + QE/QU rates, backlogs 239/950 |
| TLE idle decomposition | 14 idle | printed | 2306/2661/7961 guard-dominance numbers (limitation pillar) |
| Handoff band | 13 | printed | band-ends absorbed statement + Greedy-Q-hurts-more sentence |
| Synthetic core | 04 | results_splitpipeline.csv | contrast subsection baseline points |
| Synthetic scope sweeps | 07 <tag> ×5 | scope_<tag>.csv | contrast prose + nsat 2.7× guard-block claim (nsat) |
| Synthetic ISL | 08 | scope_isl.csv | ISL cross-plane linear-coupling comparison (slope constant claim!) |
| Synthetic idle decomposition | 09 | printed | guard 31→0 comfortable-battery numbers |
| Contrast frontier/Vsweep | 18 (NEW) | contrast_frontier.csv | frontier + Vsweep figs (previously hand-transcribed in plot scripts — now CSV per reproducibility discipline) + EV_VCONF numbers (0.175@V1, 3850@V300, 5σ vs Greedy-E, partition SLA band) |
| MPC matrix 60 runs | 15 (AFTER above, clean timing) | data/mpc/* | EV_MPC section: flip table, all adjudications, pattern check, phase clustering |
| MPC aggregate + timing | 16, 15 timing | mpc_summary.csv, proposed_decision_time.json | ratios |
| MPC windows + paired re-solve | 15 export/solve-windows + Jetson | paired_resolve.csv | Jetson paragraph |
| B2 45-source | DONE (src/17 self-overrides Q_SRC; unaffected by swap) | qsrc/src45_*.csv | judged under new pre-registered presentation rule |

## TEXT SWEEP (after data lands) — every Q_SRC-downstream sentence in main.tex
- Abstract: mainline numbers.
- Intro/C4: any quoted min-fill/downtime numbers.
- MAINLINE: Table IV full swap + SLA band + V-collapse note numbers.
- SENS: feasibility boundary, partition band [16.6,17.5]→new, Random-overtake
  ~19 kJ→new, idle decomposition 2306/2661/7961→new.
- ISL: slope/concentration re-verified values (both settings).
- STAB: QE/QU rates, blackout %, backlog comparison numbers.
- CONTRAST: frontier numbers, 5σ claim recompute, partition invariance band,
  nsat 2.7×, "V-navigable range compresses" numbers.
- GAP: untouched (Q_SRC-independent) — verify no stray Q_SRC numbers.
- Q_SRC declaration text: stylized table → measured values + provenance
  (per-class n≈29, q_src45.json), difficulty-spectrum description updated
  (steep classes 3→1; church ceiling 0.49 with no 7B advantage).
- CLAIM LAYER final form (advisor-pinned): capability main claim
  {deployable + top-tier splitter + guarantees, all band-robust};
  min-fill → band description (leader migrates Proposed→Static→Random;
  Proposed within X% of leader at lowest downtime — X from new data);
  Static paragraph hooked to guard-conservatism pillar as the two-directional
  corroboration with MPC; min-fill-optimality claim BURIED (three knockdowns:
  B_max band, post-TLE Q_SRC error, measured values).
- Gates: 0 undefined + 0 multiply-defined refs; O(1/V) qualifier grep;
  old-number sweep (grep stylized values 0.174/0.177/0.170/655/0.117/...);
  raw CSV reconciliation before push.
