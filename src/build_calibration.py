#!/usr/bin/env python3
"""Build the combined EuroSAT calibration table from all Jetson bench summaries.

Reads every data/jetson_eurosat/results_*_summary.csv, assembles one tidy table,
flags the two failed InternVL3-1B runs (collapsed to a single class, acc~random),
computes N_im = ceil(latency) (number of tau=1s slots), and marks the Pareto
frontier (maximize accuracy, minimize energy/img).

Output: data/jetson_eurosat/calibration_all.csv  (the DATA the plot script consumes).
This script does the data work ONLY; plotting lives in plot_tradeoff.py.
"""
import glob, math, os, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# Dataset dir: arg 1, else RESISC45 if present, else EuroSAT.
_default = os.path.join(HERE, "..", "data", "jetson_resisc45")
if not os.path.isdir(_default):
    _default = os.path.join(HERE, "..", "data", "jetson_eurosat")
DATA = sys.argv[1] if len(sys.argv) > 1 else _default

# Friendly family label per config (for the paper / legend)
FAMILY = {
    "Qwen2-VL-2B-Q4": "Qwen2-VL", "Qwen2.5-VL-3B-Q4": "Qwen2.5-VL",
    "Qwen2.5-VL-7B-Q4": "Qwen2.5-VL", "InternVL3-8B-Q4": "InternVL3",
    "InternVL3-1B-Q8": "InternVL3", "InternVL3-1B-Q8-224": "InternVL3",
    "gemma-3-4b-Q4": "Gemma3", "SmolVLM2-2.2B-Q4": "SmolVLM2",
    "InternVL2.5-4B-Q4": "InternVL2.5",
}

def load():
    rows = []
    # Use ONLY the n=1300 full re-benchmark summaries (one caliber for the whole paper);
    # fall back to the older subset only if the full run is absent.
    pat = "results_*_RESISC45_full_summary.csv"
    files = sorted(glob.glob(os.path.join(DATA, pat)))
    if not files:
        files = sorted(glob.glob(os.path.join(DATA, "results_*_summary.csv")))
    for f in files:
        d = pd.read_csv(f).iloc[0].to_dict()
        d["model"] = str(d["model"]).replace("_RESISC45_full", "")
        rows.append(d)
    df = pd.DataFrame(rows)
    df = df.rename(columns={"accuracy": "Q", "T_im_mean_s": "T_s",
                            "E_im_mean_J": "E_J", "Ppeak_p95_W": "Ppeak_W"})
    df["family"] = df["model"].map(FAMILY).fillna("?")
    # failed = collapsed predictor: the two 1B runs sit at chance (10 classes -> 0.10)
    df["failed"] = df["Q"] <= 0.12
    df["N_im"] = df["T_s"].apply(lambda t: max(1, math.ceil(t)))
    return df.sort_values("E_J").reset_index(drop=True)

def mark_pareto(df):
    """Pareto-optimal = no other (non-failed) config has Q>=this and E<=this (one strict)."""
    pareto = []
    cand = df[~df["failed"]]
    for _, r in df.iterrows():
        if r["failed"]:
            pareto.append(False); continue
        dom = ((cand["Q"] >= r["Q"]) & (cand["E_J"] <= r["E_J"]) &
               ((cand["Q"] > r["Q"]) | (cand["E_J"] < r["E_J"]))).any()
        pareto.append(not dom)
    df["pareto"] = pareto
    return df

def main():
    df = mark_pareto(load())
    cols = ["model", "family", "Q", "T_s", "E_J", "Ppeak_W", "N_im", "pareto", "failed"]
    out = os.path.join(DATA, "calibration_all.csv")
    df[cols].to_csv(out, index=False)
    print("wrote", out, "\n")
    with pd.option_context("display.width", 120):
        print(df[cols].to_string(index=False))
    print("\nPareto-optimal configs (scheduler's useful action space):")
    print("  " + ", ".join(df[df["pareto"]]["model"].tolist()))
    print("Dominated:", ", ".join(df[(~df["pareto"]) & (~df["failed"])]["model"].tolist()))
    print("Failed (excluded):", ", ".join(df[df["failed"]]["model"].tolist()))

if __name__ == "__main__":
    main()
