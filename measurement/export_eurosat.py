#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export a balanced EuroSAT subset (50 images/class = 500 images) to PNG at 224x224,
together with an index CSV (path,label,class_name) used by profile_jetson.py.

The EuroSAT parquet can be obtained from ModelScope (no VPN needed in CN):
    git lfs install
    git clone https://www.modelscope.cn/datasets/tanganke/eurosat.git
The 'image' column stores JPEG bytes in {'bytes': ...}; 'label' is 0-9.

Why 224x224: EuroSAT native tiles are 64x64. Some VLM vision encoders mis-handle
such small inputs (they perceive a near-black image); upscaling to 224x224 makes
the content visible and is reported as the input resolution in the paper.
"""

import io, os
import pandas as pd
from PIL import Image
from collections import defaultdict

PARQUET = "data/test-00000-of-00001.parquet"   # adjust to your clone path
OUT_DIR = "eurosat_imgs"
PER_CLASS = 50
RESIZE = (224, 224)

EUROSAT_CLASSES = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
                   "Industrial", "Pasture", "PermanentCrop", "Residential",
                   "River", "SeaLake"]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_parquet(PARQUET)
    counts, rows = defaultdict(int), []
    for _, row in df.iterrows():
        lbl = int(row["label"])
        if counts[lbl] < PER_CLASS:
            img = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
            img = img.resize(RESIZE, Image.BICUBIC)
            fname = f"{OUT_DIR}/{lbl}_{EUROSAT_CLASSES[lbl]}_{counts[lbl]:02d}.png"
            img.save(fname)
            rows.append((fname, lbl, EUROSAT_CLASSES[lbl]))
            counts[lbl] += 1
        if sum(counts.values()) >= PER_CLASS * len(EUROSAT_CLASSES):
            break
    pd.DataFrame(rows, columns=["path", "label", "class_name"]).to_csv(
        "eurosat_index.csv", index=False)
    print(f"exported {len(rows)} images (224x224) -> {OUT_DIR}/, index -> eurosat_index.csv")


if __name__ == "__main__":
    main()
