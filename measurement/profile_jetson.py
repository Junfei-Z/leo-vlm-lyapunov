#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jetson Orin NX power/latency profiling for VLM inference on EuroSAT.

Reproduces the per-configuration calibration table (T_im, E_im, Ppeak_im, accuracy)
used by the simulator. Runs a llama.cpp server with a given GGUF VLM + mmproj,
streams EuroSAT images through the OpenAI-compatible /v1/chat/completions endpoint,
and concurrently logs board power via `tegrastats`.

Hardware used in the paper: NVIDIA Jetson Orin NX 16GB, JetPack 6.2.1 (L4T r36.4.7).
Inference backend: llama.cpp (CUDA), GGUF Q4_K_M, EuroSAT images upscaled to 224x224.

USAGE
-----
1. Start a llama.cpp server in another terminal, e.g.:

   ./llama-server \
       -m  /path/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf \
       --mmproj /path/mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf \
       -ngl 99 -c 4096 --host 127.0.0.1 --port 8080

   (For 7B/8B you may need --no-mmproj-offload and/or a lower -ngl on 16GB.)

2. Export a 224x224 EuroSAT subset + index CSV (see export_eurosat.py).

3. Run this profiler:

   python3 profile_jetson.py MODEL_TAG

Outputs results_<MODEL_TAG>_perimg.csv and a one-line summary.

NOTE: tegrastats field name differs across boards. On Orin NX it is
VDD_CPU_GPU_CV (CPU+GPU+CV rail, in mW). Adjust POWER_FIELD_RE if needed.
"""

import os, sys, re, time, signal, base64, subprocess
import numpy as np
import pandas as pd
import requests

SERVER = "http://127.0.0.1:8080"
INDEX_CSV = "eurosat_index.csv"          # produced by export_eurosat.py
MODEL_TAG = sys.argv[1] if len(sys.argv) > 1 else "model"
POWER_LOG = f"power_{MODEL_TAG}.log"
OUT_PER_IMG = f"results_{MODEL_TAG}_perimg.csv"

EUROSAT_CLASSES = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
                   "Industrial", "Pasture", "PermanentCrop", "Residential",
                   "River", "SeaLake"]
PROMPT = ("This is a satellite image. Classify the land cover into exactly one of: "
          "AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, "
          "PermanentCrop, Residential, River, SeaLake. Answer with only the category name.")

# tegrastats power field (mW). Adjust to your board if different.
POWER_FIELD_RE = re.compile(r"(VDD_GPU_SOC|VDD_CPU_GPU_CV|POM_5V_GPU)\s+(\d+)mW")
TEGRA_INTERVAL_MS = 50   # use the smallest your board supports to capture transients


def start_tegrastats(logfile):
    if os.path.exists(logfile):
        os.remove(logfile)
    return subprocess.Popen(
        f"stdbuf -oL tegrastats --interval {TEGRA_INTERVAL_MS} > {logfile} 2>&1",
        shell=True, preexec_fn=os.setsid)


def stop_tegrastats(proc):
    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)


def parse_power(logfile):
    vals = []
    with open(logfile) as f:
        for line in f:
            m = POWER_FIELD_RE.search(line)
            if m:
                vals.append(int(m.group(2)))   # mW
    return np.array(vals, dtype=float)


def classify(img_path):
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    payload = {
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]}],
        "temperature": 0, "max_tokens": 16,
    }
    r = requests.post(f"{SERVER}/v1/chat/completions", json=payload, timeout=300)
    return r.json()["choices"][0]["message"]["content"].strip()


def main():
    assert os.path.exists(INDEX_CSV), f"missing {INDEX_CSV} (run export_eurosat.py first)"
    index = pd.read_csv(INDEX_CSV)

    teg = start_tegrastats(POWER_LOG)
    time.sleep(2)

    records = []
    t0_global = time.time()
    for n, row in index.iterrows():
        t0 = time.time()
        try:
            ans = classify(row["path"])
        except Exception as e:
            ans = f"ERROR:{e}"
        t1 = time.time()

        pred = next((ci for ci, c in enumerate(EUROSAT_CLASSES)
                     if c.lower() in ans.lower()), -1)
        records.append(dict(n=n, path=row["path"], true=int(row["label"]),
                            answer=ans, pred=pred, correct=int(pred == int(row["label"])),
                            t_start=t0, t_end=t1, latency_s=t1 - t0))
        if n % 20 == 0:
            print(f"[{n}/{len(index)}] {ans[:18]:18s} {t1-t0:.2f}s")
    t1_global = time.time()

    stop_tegrastats(teg)
    time.sleep(1)

    powers = parse_power(POWER_LOG)   # mW
    df = pd.DataFrame(records)

    # align power samples to per-image windows by uniform mapping over the run
    if len(powers) > 0:
        dt = (t1_global - t0_global) / len(powers)
        E, P = [], []
        for _, rr in df.iterrows():
            i0 = max(0, int((rr["t_start"] - t0_global) / dt))
            i1 = min(len(powers), max(i0 + 1, int((rr["t_end"] - t0_global) / dt)))
            seg = powers[i0:i1]
            avg_w = seg.mean() / 1000.0
            E.append(avg_w * rr["latency_s"])
            P.append(np.percentile(seg, 95) / 1000.0)   # P-hat^CP : 95th pct
        df["E_im_J"] = E
        df["Ppeak_W"] = P

    df.to_csv(OUT_PER_IMG, index=False)
    print("\n=== summary ===")
    print(f"model        : {MODEL_TAG}")
    print(f"accuracy     : {df['correct'].mean():.3f}")
    print(f"T_im (p50 s) : {df['latency_s'].median():.3f}")
    if "E_im_J" in df:
        print(f"E_im (mean J): {df['E_im_J'].mean():.3f}")
        print(f"Ppeak (p95 W): {df['Ppeak_W'].max():.3f}")
    print(f"saved {OUT_PER_IMG}")


if __name__ == "__main__":
    main()
