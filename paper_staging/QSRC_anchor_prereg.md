# Q_SRC anchor chain (pre-registered BEFORE any run) — 2026-07-04

Trigger: preparing the 45-source extension exposed that the mainline Q_SRC is
stylized (declared merge/round at the time) and deviates from the full n=1300
per-class measurements (largest: church 7B stylized 0.70 vs measured 0.49).
This is a seam already on paper; the anchor chain welds it or exposes it.

## Step B1 — 6-source MEASURED-value anchor (single variable: quality mapping)
Config = TLE mainline EXACTLY (trace overrides, rule sizes 12.2 W / 16580 J,
h = 0.69, arrival 0.10, U_tgt 0.30, seeds 7–16), with ONLY Q_SRC replaced by
the measured per-class values of the same six classes (data/q_src45.json):

| class | stylized 2B/3B/7B | measured 2B/3B/7B | max |Δ| |
|---|---|---|---|
| beach | .90/.90/.90 | .88/.94/.91 | .04 |
| intersection | .40/.50/.70 | .38/.77/.88 | .27 |
| church | .30/.60/.70 | .41/.49/.49 | .21 |
| baseball_diamond | .30/.70/.80 | .32/.68/.83 | .03 |
| roundabout | .30/.80/.90 | .25/.88/.88 | .05 |
| mountain | .20/.20/.80 | .19/.33/.78 | .13 |

Note what the measured values change structurally: church loses its 7B
advantage entirely (3B = 7B = 0.49); the steep "only-7B-helps" story now
rests on mountain alone (0.19/0.33/0.78, structure intact).

**Pre-registered criterion ("conclusions unchanged"), fixed now:** the three
graded facts hold — (i) Greedy-Q undeployable (downtime > SLA) across the
studied battery range; (ii) at the tight mainline point the proposed
scheduler is the only deployable splitter; (iii) simple policies take over
min-fill in the energy-abundant regime — AND the min-fill ordering among
deployable policies at the mainline point is unchanged. Any numeric drift is
accepted; only structural change counts as "moved".
- If unchanged: mainline upgrades to measured Q_SRC (or at minimum the paper
  declares the deviation and the anchor verdict); seam welded.
- If moved: STOP, this outranks the 45-source work (a conclusion sensitive to
  quality-mapping error cannot survive review) — report before proceeding.

## Step B2 — 45-source run (second variable: the source set, moved ONLY now)
All 45 RESISC classes as sources, ZERO selection: q_src45.json enters the sim
raw (no merging, no rounding, dead classes included — they are real
low-value sources and how the scheduler treats them is part of the result).
Declared load calibration: total offered load kept equal to mainline
(per-source arrival 0.10 x 6/45 = 0.01333), so the source STRUCTURE is the
only moved variable, not the load. U_tgt unchanged (0.30). Judged against
the B1 anchor, not against the stylized mainline.

**Watch item (recorded now, decided only after data, in pre-registered
style):** 45-source min-fill may be pinned by classes no model can serve
(7B ~= 0); if so the metric presentation may need a servable-source min-fill
or quantile view IN ADDITION to (never instead of) the raw min-fill. Fill
distributions are captured per run so this needs no re-run.

## Mechanics
src/17_qsrc_anchor.py; jobs = {anchor6, src45} x {core, battery sweep
[14000, 15000, 16580, 17500, 19000, 22000]}; outputs data/qsrc/*.csv.
run_sim gains two telemetry-only return fields (fill array, arrivals array);
no behavioral change to the locked model.

---

## B1 VERDICT (recorded 2026-07-04, adjudicated per the criterion above): MOVED
(i) Greedy-Q undeployable across range: HOLDS (5.6-9.4%, all > 5%).
(ii) Only deployable splitter at the tight point: HOLDS.
(iii) Simple-policy takeover in abundance: HOLDS (Random 4.9%/0.336 from 19 kJ).
Ordering among deployables at the rule-sized point: BROKEN -- measured values
give Static 0.194 > Proposed 0.165 > Greedy-E 0.142 (stylized had Proposed
0.174 > GE 0.154 > Static 0.117). Mechanism: measured 3B is far stronger than
stylized (church 7B advantage vanishes entirely, 3B=7B=0.49; steep classes
collapse 3 -> 1), so the always-3B policy's service volume beats the guard's
conservatism on min-fill inside the [~16.5, ~18.5] kJ band; Proposed retakes
Static from 19 kJ but Random is deployable there.

## DECISION (advisor, 2026-07-04): Option 1 -- mainline upgrades to measured
Q_SRC globally; every Q_SRC-downstream number re-runs; the claim layer is
rewritten to its FINAL form: capability main claim (unique {deployable +
top-tier splitter + guarantees} across the band -- all three components
B1-robust and Q_SRC-value-independent), min-fill DEMOTED permanently to a
band-dependent description (leader migrates: Proposed-only tight end, Static
mid-band, Random abundance; Proposed stays within X% of the leader at lowest
downtime). The min-fill-optimality claim was knocked down three times by
independent perturbation axes (B_max band; post-TLE Q_SRC error; measured
values) -- it is structurally fragile and is now buried, not to be picked up
again. Static finding joins MPC as the two-directional guard-conservatism
corroboration (MPC: lookahead + spends margin, wins fairness 0.265/2.39%;
Static: no lookahead + eats downtime, wins service 0.194/3.2%).

## B2 PRESENTATION RULE (pre-registered NOW, before B2 is judged)
With 45 raw classes, dead classes pin min-fill at 0 for every policy: raw
min-fill degenerates into an unservable-class indicator with zero
discriminative content. Declared presentation: primary metrics are the
per-source fill QUANTILE view (q25 / median, telemetry already captured) and
SERVABLE-SET min-fill, where the servable set excludes classes structurally
unreachable at the model ceiling: threshold = the widest gap in the sorted
7B per-class accuracies of q_src45.json (bimodal gap, zero selection
freedom). Raw min-fill = 0 is still reported with its semantic-degeneration
explanation. This rule precedes any B2 reading.
