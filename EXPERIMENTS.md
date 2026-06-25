# Experiment Log & Roadmap

Project: **Online Collaborative VLM Inference over Solar-Powered LEO Satellite Networks via Robust Lyapunov Optimization**

This document records (1) what has been done, (2) what each paper claim needs in
order to be defensible, and (3) a prioritized list of experiments still to run —
written so it can drive an automated research / experiment pipeline.

---

## Part 1 — What we have done so far

### 1.1 Hardware calibration (real measurements) ✅ DONE

Measured the per-configuration cost of VLM inference on real edge hardware, to
populate the model parameters `E[T_im]`, `E[E_im]`, `P̂^CP_im`, `Q_im`.

- **Device:** NVIDIA Jetson Orin NX 16GB, JetPack 6.2.1 (L4T r36.4.7), CUDA 12.6
- **Backend:** llama.cpp (CUDA), GGUF `Q4_K_M`
- **Dataset:** EuroSAT (land-cover classification), tiles upscaled to 224×224
- **Method:** llama.cpp server + OpenAI-compatible endpoint; latency timed per image;
  power sampled by `tegrastats` (`VDD_CPU_GPU_CV` rail, 50 ms); energy = power integral
  over the inference window; peak power = 95th percentile (`P̂^CP`).
- **Controls:** locked power mode, locked clocks, ≥20 images/config, median latency.

Result table (`data/jetson_calibration.csv`):

| Model         | Quant | Acc (0-shot) | T_im (s) | E_im (J) | P_peak (W) | N_im (τ=1s) |
|---------------|-------|-------------:|---------:|---------:|-----------:|------------:|
| Qwen2.5-VL-3B | Q4    | 0.30         | 1.32     | 5.4      | 6.6        | 2           |
| Qwen2.5-VL-7B | Q4    | 0.47         | 3.89     | 27.9     | 10.7       | 4           |
| InternVL3-8B  | Q4    | 0.51         | 8.66     | 62.8     | 11.0       | 9           |

Key empirical findings:
- **Non-linear quality–energy coupling (diminishing returns):** 3B→7B ≈ 5× energy
  for +0.17 accuracy; 7B→8B ≈ 2.2× energy for only +0.04.
- **Single-satellite memory limit:** the 8B model does not fit comfortably on 16 GB
  (needed projector offload / reduced GPU layers), motivating cross-satellite split.

> NOTE: a 1B model (InternVL3-1B, Q8) was also measured but dropped from the main set
> because it used a different quantization (Q8 vs Q4) and degenerated to a constant
> answer under zero-shot prompting. Kept as a backup data point only.

### 1.2 Simulation: single-satellite minimal validation ✅ DONE

Script: `src/01_single_sat_lyapunov.py`. Figures: `sim_timeseries.png`, `sim_V_tradeoff.png`.

Verified three theoretical properties on one solar-powered satellite over 3 orbits
(τ = 1 s, ~17,100 slots, 60/40 sunlit/eclipse):

- **(Theorem 1) Queue stability** — energy/power/quality virtual queues stay bounded.
- **(Theorem 3) ISS robustness under eclipse** — the energy queue Q^E rises during each
  eclipse and recovers to ~0 in sunlight; battery shows clean periodic charge/discharge.
- **(Theorem 2) [O(1/V), O(V)] trade-off** — as V increases, min-quality rises while the
  average queue backlog grows (the classic Lyapunov knob).

### 1.3 Simulation: baseline comparisons ✅ DONE (single + multi-sat + split)

Scripts: `src/02_single_sat_baselines.py`, `src/03_multisat_baselines.py`,
`src/04_split_pipeline.py`. Figures: `baseline_comparison.png`,
`multisat_comparison.png`, `splitpipeline_comparison.png`.

Baselines: Greedy-Q (always best single-sat quality), Greedy-E (always cheapest),
Random, MaxBatt-7B (route to highest-battery sat, fixed 7B).

Scenario features implemented:
- Multi-satellite (4 computing sats) with **staggered eclipse phases**.
- **Solar model = paper Eq.17** (deterministic δ + small Gaussian perturbation, GaAs
  efficiency ≈ 0.30) — replaced the original terrestrial 4-state HMM, which contradicts
  the paper's "LEO solar is deterministic" premise.
- Task offloading: tasks routed to any free satellite.
- **Two-stage split pipeline:** the 8B model is split front/rear across two
  ISL-connected satellites; handoff connectivity check matches Eq.16.

Findings (honest):
- **Safety / sustainability — robust win in every scenario:** the proposed scheduler
  has **zero peak-power violations** and **zero battery-depletion (blackout) slots**;
  greedy-quality baselines drain eclipse-side batteries and incur large downtime
  (thousands of blackout slots).
- **Unique capability:** only the proposed scheduler can run the 8B model (via the
  cross-satellite split pipeline); baselines are structurally capped at 7B.
- **Quality is a genuine trade-off, not a blanket win:** because split-8B is
  resource-expensive and the quality–energy curve has diminishing returns, simple
  energy-aware baselines can match or exceed raw throughput in some regimes. We report
  this honestly rather than tuning the scenario to force a quality win.

### 1.4 Repository ✅ DONE

Packaged as a GitHub-ready repo: `src/`, `data/`, `figures/`, `measurement/`,
`README.md`, `requirements.txt`, `LICENSE`. Calibration and result CSVs included.

---

## Part 2 — Claim-by-claim verification matrix

What the paper claims (C1–C4 + the theorems) vs what evidence exists vs what is missing.

| # | Claim | Evidence we have | Still missing |
|---|-------|------------------|---------------|
| C1 | First formulation of energy-aware collaborative VLM inference over solar LEO (split + peak-power + eclipse + dynamic ISL + max-min) | Model + simulator implement all five pieces | A clean end-to-end run that exercises **dynamic ISL topology** (currently a placeholder); explicit max-min demonstration |
| C2 | Two-timescale robust Lyapunov framework (offline ILP deploy + per-slot online), calibrated power bounds make peak feasibility per-slot checkable | Online per-slot scheduler done; calibrated `P̂^CP` measured | **Offline ILP deployment stage** not implemented (configs are currently hand-placed); per-slot runtime/complexity measurement |
| C3 | Theoretical guarantees: O(1/V) gap, mean-rate constraint satisfaction, ISS eclipse stability | V-tradeoff + queue stability + eclipse recovery shown empirically | **Optimality-gap vs offline MILP benchmark** (needs the MILP solved); a quantitative gap-vs-V curve |
| C4 | Empirical validation on EuroSAT + RESISC45, ViT-B/16…G/14, multiple quantization, zero peak violations, min-quality near MILP | EuroSAT calibration on 3 real VLMs; zero peak violations in sim | **RESISC45** not measured; **multiple quantization levels** per model not swept; min-quality-vs-MILP gap not plotted |
| T1 | Mean-rate stability | queues bounded in sim | formal plot of E[Q]/T → 0 over long horizon |
| T2 | O(1/V) optimality gap | V-tradeoff trend shown | gap measured **against the offline MILP** (absolute gap, not just trend) |
| T3 | Robust ISS stability under eclipse | Q^E rise/recover shown | stress test: vary eclipse fraction toward the ξ+|c|<1 boundary |

---

## Part 3 — Experiments still to run (prioritized for auto-research)

Ordered by importance to the paper's claims. Each item is written as a concrete,
runnable experiment with inputs, procedure, and the figure/number it produces.

### P0 — Core claims that are currently unsupported

#### EXP-1. Offline MILP benchmark + optimality-gap curve  (supports C2, C3, T2)
- **Why:** Theorem 2 claims an O(1/V) gap *relative to the offline optimum*. Right now we
  only show the *trend*; we never compute the actual optimum, so the gap is unquantified.
- **Inputs:** same constellation/config/arrival traces; full future known.
- **Procedure:** formulate problem (P_off) as a MILP (objective = max-min quality;
  constraints Eq.1–16 + long-term energy + per-slot peak cap). Solve with PuLP / OR-Tools /
  Gurobi on a *short* horizon (small |T|, few sats) where MILP is tractable. Run the online
  scheduler on the same trace for a sweep of V.
- **Output:** plot `min-quality(online, V)` vs `min-quality(MILP)`; show gap shrinks ~1/V.
  This is the single most important missing figure (it is the headline theoretical claim).

#### EXP-2. Offline ILP model-deployment stage  (supports C2)
- **Why:** the paper's two-timescale design has an offline ILP that decides `y_sm`
  (which config each satellite hosts). Currently configs are hand-placed.
- **Procedure:** implement the upper-level ILP (Eq.27: maximize Σ y_sm·Q̄_m s.t. RAM and
  pipeline-pairing constraints). Feed its output into the online scheduler.
- **Output:** table comparing hand-placed vs ILP-deployed performance; demonstrates the
  full two-timescale pipeline end-to-end.

#### EXP-3. Dynamic ISL topology from real ephemeris  (supports C1)
- **Why:** ISL connectivity `ℓ^t_{ss'}` is currently a placeholder (all-connected). The
  paper's selling point is that it is deterministic-but-time-varying (+Grid, inter-plane
  links toggling).
- **Procedure:** generate a real staggered +Grid topology — either (a) from Starlink TLE
  via Skyfield (propagate, compute line-of-sight + max-range + 2 intra + 2 inter neighbors),
  or (b) a synthetic +Grid with inter-plane links toggling on a coherence timescale.
  Replace `isl_connected()` with a lookup into this precomputed `ℓ^t_{ss'}` table.
- **Output:** rerun the split-pipeline comparison; show the handoff constraint (Eq.16)
  actually binds sometimes (a split is refused because the link is down at t+N_front),
  and the scheduler reroutes. This makes Eq.16 non-trivial.

### P1 — Strengthening empirical breadth (supports C4)

#### EXP-4. Second dataset: RESISC45  (supports C4)
- **Why:** the paper claims validation on EuroSAT **and** RESISC45. We only did EuroSAT.
- **Procedure:** rerun `measurement/profile_jetson.py` on a 224×224 RESISC45 subset for the
  same 3 VLMs. (45 classes; adjust the prompt class list.)
- **Output:** a second calibration table; confirms the quality–energy trend generalizes.

#### EXP-5. Quantization sweep per model  (supports C1/C4 "multiple quantization levels")
- **Why:** the config set M is defined over architecture × quantization. We only used Q4.
- **Procedure:** profile each VLM at additional quant levels (e.g. Q2_K, Q4_K_M, Q8_0) on
  Jetson. Each (model, quant) becomes a distinct config m with its own (T,E,P,Q).
- **Output:** a richer config set; a quality–energy scatter with a denser Pareto front.
  This is also the cleanest way to widen the "low-energy end" of the trade-off.

#### EXP-6. Sensitivity sweeps  (supports robustness of conclusions)
- **Procedure:** sweep, one at a time: arrival rate λ, battery capacity B_max, solar power
  P_solar, peak cap P_cap, number of satellites, number of sources. For each, record peak
  violations, downtime, min-fill, total quality for all policies.
- **Output:** a set of line plots showing the proposed scheduler's safety advantage holds
  across the parameter space, and identifying the regime where its quality advantage appears.

### P2 — Robustness / theory stress tests

#### EXP-7. Eclipse-fraction stress test  (supports T3, the ISS condition ξ+|c|<1)
- **Procedure:** vary the eclipse fraction from mild (e.g. 20%) to severe (e.g. 55%),
  approaching the stability boundary. Plot lim-sup of E[L(Q)] (Lyapunov function) and the
  energy-queue backlog vs eclipse fraction.
- **Output:** shows queues stay bounded while ξ+|c|<1 and blow up beyond it — direct
  evidence for the robust-stability theorem and its condition.

#### EXP-8. Long-horizon mean-rate stability  (supports T1)
- **Procedure:** run a very long horizon (e.g. 20+ orbits) and plot E[Q(T)]/T for each
  virtual queue.
- **Output:** curves trending to 0, the formal definition of mean-rate stability.

#### EXP-9. Per-slot runtime / scalability  (supports C2 "lightweight per-slot")
- **Procedure:** measure wall-clock time of the per-slot subproblem as the constellation
  grows (sats, sources, configs). Confirm it stays small (the paper argues O(|I_t|·|S|)).
- **Output:** runtime-vs-size plot; supports the "lightweight online scheduler" claim.

### P3 — Nice-to-have / future-work framing

#### EXP-10. Activation-compression sensitivity (future work in the paper)
- Vary ISL handoff cost / activation size; show the split pipeline degrades gracefully.

#### EXP-11. Cislunar / long-eclipse scenario (paper's future-work hook)
- Set eclipse to hours-long (lunar), show the robust framework still bounds queues if the
  sunlit fraction condition holds.

---

## Part 4 — Suggested order for an automated run

1. **EXP-1 (MILP gap)** — highest value, directly proves the headline theorem (T2/C3).
2. **EXP-3 (dynamic ISL)** — makes C1 and Eq.16 real, not a placeholder.
3. **EXP-2 (offline ILP deploy)** — completes the two-timescale story (C2).
4. **EXP-6 (sensitivity sweeps)** — shows conclusions are robust, cheap to run.
5. **EXP-7 (eclipse stress)** + **EXP-8 (long-horizon)** — finish the theory plots (T1/T3).
6. **EXP-4 (RESISC45)** + **EXP-5 (quant sweep)** — finish the empirical breadth (C4)
   (these require the Jetson, so batch them on-device).
7. **EXP-9 (runtime)** — quick, supports the scalability claim.

---

## Part 5 — Honesty notes to carry into the paper

- The dependable, scenario-independent advantage is **safety/sustainability** (zero peak
  violations, zero blackout). Frame this as the main result — it matches the title's "Robust".
- The **quality** advantage is conditional; it is clearest as the **unique ability to run
  the 8B model via split pipeline** (a capability baselines lack), and in regimes where the
  peak/energy constraints actually bind. Do not over-claim a blanket quality win.
- `Q_im` is zero-shot accuracy used to *order* configs. Absolute accuracy could be raised by
  fine-tuning, but the measured **energy/power profiles are the empirical contribution** and
  are independent of accuracy.
