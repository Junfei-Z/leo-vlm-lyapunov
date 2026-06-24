# Online Collaborative VLM Inference over Solar-Powered LEO Satellite Networks via Robust Lyapunov Optimization

Simulation code, hardware-measured calibration data, and figures for energy-aware
collaborative Vision-Language Model (VLM) inference on solar-powered Low Earth Orbit
(LEO) satellite constellations.

The scheduler decides, in each time slot and using only current state, **which VLM
configuration to run, on which satellite, and whether to split a large model across
two satellites**, while respecting two hard constraints that existing energy-harvesting
schedulers ignore:

1. **Instantaneous peak-power cap** — VLM inference draws sharp power transients during
   model loading/compute; exceeding the per-slot hardware ceiling triggers shutdown.
2. **Eclipse-induced energy intermittency** — each ~95-minute LEO orbit alternates
   between sunlit (~60%) and eclipse (~40%); during eclipse, solar input is zero and
   the battery must sustain all computation.

A two-timescale design (offline model deployment + online per-slot scheduling) with a
robust Lyapunov drift-plus-penalty rule keeps the virtual queues stable **through
zero-harvest eclipse intervals** via an input-to-state-stability (ISS) argument, and
makes per-slot peak-power feasibility checkable in closed form from offline-calibrated
power bounds.

---

## What this repository contains

```
.
├── data/
│   ├── jetson_calibration.csv      # measured T_im, E_im, Ppeak_im, accuracy per VLM
│   └── results_splitpipeline.csv   # main multi-sat + split-pipeline comparison results
├── figures/
│   ├── sim_timeseries.png          # single-sat: battery / queues over 3 orbits
│   ├── sim_V_tradeoff.png          # [O(1/V), O(V)] trade-off
│   ├── baseline_comparison.png     # single-sat baseline comparison
│   ├── multisat_comparison.png     # multi-sat (staggered eclipse) comparison
│   └── splitpipeline_comparison.png# multi-sat + split-pipeline comparison (main)
├── measurement/
│   ├── export_eurosat.py           # build a 224x224 EuroSAT subset + index
│   └── profile_jetson.py           # llama.cpp + tegrastats profiler -> calibration CSV
├── src/
│   ├── 01_single_sat_lyapunov.py   # minimal simulator (queue stability, eclipse robustness)
│   ├── 02_single_sat_baselines.py  # single-sat baseline comparison
│   ├── 03_multisat_baselines.py    # multi-sat, staggered eclipse, solar Eq.17
│   └── 04_split_pipeline.py        # multi-sat + two-stage split pipeline (main experiment)
├── requirements.txt
└── README.md
```

---

## Hardware-measured calibration (the core empirical input)

All energy/latency/power parameters are **measured on real hardware**, not assumed.

- **Device:** NVIDIA Jetson Orin NX 16GB, JetPack 6.2.1 (L4T r36.4.7), CUDA 12.6
- **Backend:** llama.cpp (CUDA), GGUF `Q4_K_M`
- **Dataset:** EuroSAT (land-cover classification), tiles upscaled to 224×224
- **Protocol:** `nvpmodel` fixed power mode, `jetson_clocks` locked; per-image latency
  timed at the OpenAI-compatible endpoint; board power sampled by `tegrastats`
  (`VDD_CPU_GPU_CV` rail) and reduced to per-image energy (integral) and a 95th-percentile
  peak-power bound (`P̂^CP`).

| Model           | Quant | Acc (0-shot) | T_im (s) | E_im (J) | P_peak (W) |
|-----------------|-------|-------------:|---------:|---------:|-----------:|
| Qwen2.5-VL-3B   | Q4_K_M| 0.30         | 1.32     | 5.4      | 6.6        |
| Qwen2.5-VL-7B   | Q4_K_M| 0.47         | 3.89     | 27.9     | 10.7       |
| InternVL3-8B    | Q4_K_M| 0.51         | 8.66     | 62.8     | 11.0       |

Two observations that motivate the scheduler:

- **Quality–energy coupling is non-linear (diminishing returns).** Going 3B→7B costs
  ~5× energy for +0.17 accuracy; 7B→8B costs ~2.2× more energy for only +0.04. Blindly
  using the largest model is not worth it under an energy budget.
- **The largest model does not fit on a single satellite.** On 16GB it must be loaded
  with projector offload / reduced GPU layers, motivating **cross-satellite split
  inference** for the 8B model.

`data/jetson_calibration.csv` holds these values; reproduce them with
`measurement/profile_jetson.py`.

---

## Time-scale separation

The design rests on a clean separation of three time scales (verified against the
measurements and orbital mechanics):

| Scale | Quantity | Magnitude |
|-------|----------|-----------|
| fast  | scheduling slot `τ` | ~1 s (set near the fastest measured inference; N_im = ⌈T_im/τ⌉ = 2–9) |
| medium| inter-plane ISL coherence | tens of seconds (handoff constraint Eq.16 is verified against offline ephemeris) |
| slow  | sunlit/eclipse cycle | ~95 min orbit, ~60/40 split (the regime where ISS robustness matters) |

The simulator uses `τ = 1 s` and runs `N_orbits = 3` (≈17,100 slots) so that the full
sunlit→eclipse→sunlit cycle is exercised several times.

---

## Reproducing the simulations

```bash
pip install -r requirements.txt

# 1) single-satellite: queue stability + eclipse robustness + V trade-off
python src/01_single_sat_lyapunov.py

# 2) single-satellite baseline comparison
python src/02_single_sat_baselines.py

# 3) multi-satellite with staggered eclipse (solar model = paper Eq.17)
python src/03_multisat_baselines.py

# 4) MAIN: multi-satellite + two-stage split pipeline
python src/04_split_pipeline.py
```

Each script prints a results table and writes its figure into the working directory.

### Solar-energy model

LEO solar input is **deterministic** (orbital mechanics), not a terrestrial weather
process. We use

```
E^t_s = P_solar · (η_s + ε^t_s) · τ · δ^t_s
```

where `δ^t_s ∈ {0,1}` is the offline-computable sunlit indicator, `η_s ≈ 0.30`
(space-grade GaAs panel efficiency), and `ε^t_s ~ N(0, σ²)` a small attitude/aging
perturbation. In the multi-satellite scripts each satellite has a **phase-offset
eclipse** so the constellation is never fully dark at once — which is what makes
cross-satellite offloading worthwhile.

---

## Reproducing the hardware calibration

```bash
# (on the Jetson) export a balanced 224x224 EuroSAT subset
python measurement/export_eurosat.py

# start a llama.cpp server for one model, e.g. Qwen2.5-VL-3B
./llama-server -m Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf \
    --mmproj mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf \
    -ngl 99 -c 4096 --host 127.0.0.1 --port 8080

# profile latency + power + accuracy
python measurement/profile_jetson.py Qwen2.5-VL-3B
```

For 7B/8B on 16GB you may need `--no-mmproj-offload` and/or a lower `-ngl`.

---

## Findings (read this before citing numbers)

The experiments support the following claims, in order of strength:

1. **Safety / sustainability (robust, holds in every scenario).** The proposed scheduler
   achieves **zero peak-power violations** and **zero battery-depletion (blackout)
   slots** across single-sat, multi-sat, and split-pipeline settings. Quality-greedy
   baselines that ignore the energy state drain eclipse-side batteries and incur large
   downtime (e.g. thousands of blackout slots in the multi-sat run). This is the
   principal, dependable advantage and matches the "robust" claim.

2. **Unique capability: cross-satellite split inference.** Only the proposed scheduler
   can execute the 8B model, by orchestrating a two-stage front/rear pipeline across two
   ISL-connected satellites (with the handoff connectivity check, Eq.16). Greedy/random/
   fixed baselines cannot organize a cross-satellite pipeline and are therefore capped at
   the largest single-satellite model (7B).

3. **Quality is a trade-off, not a blanket win.** Because split 8B is resource-expensive
   (occupies two satellites) and the quality–energy curve has diminishing returns, total
   delivered quality is a genuine trade-off: simple energy-aware baselines can achieve
   competitive or higher raw throughput in some regimes. The proposed scheduler trades a
   portion of raw throughput for **constraint satisfaction and sustained operation**.
   We report these numbers honestly rather than tuning the scenario to force a quality win.

`data/results_splitpipeline.csv` contains the exact numbers behind
`figures/splitpipeline_comparison.png`.

---

## Baselines

| Name            | Behavior |
|-----------------|----------|
| Lyapunov (ours) | drift-plus-penalty over (satellite, config, split) with hard peak-power & battery feasibility |
| Greedy-Q        | always the highest single-sat quality (7B); no energy/eclipse awareness; cannot split |
| Greedy-E        | always the cheapest config (3B) |
| Random          | random feasible (satellite, config) |
| MaxBatt-7B      | route to the highest-battery satellite, fixed 7B |

---

## Notes and limitations

- Quality (`Q_im`) is **zero-shot** EuroSAT accuracy of general-purpose VLMs; it is used
  to order configurations, not as a tuned classifier. Fine-tuned or remote-sensing-
  specific models would raise absolute accuracy but not change the energy/power profiles,
  which are the paper's measured contribution.
- The ISL topology in the simulator is a minimal placeholder; the handoff check matches
  the structure of Eq.16. A full run would feed `ℓ^t_{ss'}` from offline TLE/ephemeris.
- These scripts are a research prototype for reproducing the paper's experiments, not a
  flight-grade scheduler.

---

## Citation

If you use this code or data, please cite the accompanying paper (preprint):

```bibtex
@article{vlm_leo_lyapunov,
  title   = {Online Collaborative VLM Inference over Solar-Powered LEO Satellite
             Networks via Robust Lyapunov Optimization},
  year    = {2025},
  note    = {Preprint}
}
```
