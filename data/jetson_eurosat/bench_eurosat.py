import time, base64, subprocess, requests, signal, os, sys
import pandas as pd
import numpy as np

# ---------- 配置 ----------
SERVER = "http://127.0.0.1:8080"
INDEX_CSV = "eurosat_index.csv"          # export_imgs.py 生成的
MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else "InternVL3-1B-Q8"  # 命令行传模型名
POWER_LOG = f"power_{MODEL_NAME}.log"
OUT_PER_IMG = f"results_{MODEL_NAME}_perimg.csv"
OUT_SUMMARY = f"results_{MODEL_NAME}_summary.csv"

EUROSAT_CLASSES = ["AnnualCrop","Forest","HerbaceousVegetation","Highway",
                   "Industrial","Pasture","PermanentCrop","Residential","River","SeaLake"]
PROMPT = ("This is a satellite image. Classify the land cover into exactly one of: "
          "AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, "
          "PermanentCrop, Residential, River, SeaLake. Answer with only the category name.")

# ---------- 启动 tegrastats 后台采样 ----------
print("启动 tegrastats...")
if os.path.exists(POWER_LOG):
    os.remove(POWER_LOG)
# 50ms 采样; 用 stdbuf 保证实时写入
teg = subprocess.Popen(
    f"stdbuf -oL tegrastats --interval 50 > {POWER_LOG} 2>&1",
    shell=True, preexec_fn=os.setsid
)
time.sleep(2)  # 让它先跑起来

# ---------- 灌图推理 ----------
df = pd.read_parquet  # placeholder
index = pd.read_csv(INDEX_CSV)
print(f"开始推理 {len(index)} 张图...")

records = []
t_global_start = time.time()
for n, row in index.iterrows():
    path, true_lbl = row["path"], int(row["label"])
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    payload = {
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{b64}"}}
            ]
        }],
        "temperature": 0,
        "max_tokens": 16
    }

    t0 = time.time()
    try:
        r = requests.post(f"{SERVER}/v1/chat/completions", json=payload, timeout=300)
        answer = r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        answer = f"ERROR:{e}"
    t1 = time.time()

    # 解析预测类别
    pred = -1
    for ci, cname in enumerate(EUROSAT_CLASSES):
        if cname.lower() in answer.lower():
            pred = ci
            break
    correct = int(pred == true_lbl)

    records.append({
        "n": n, "path": path, "true": true_lbl,
        "answer": answer, "pred": pred, "correct": correct,
        "t_start": t0, "t_end": t1, "latency_s": t1 - t0
    })
    if n % 20 == 0:
        print(f"  [{n}/{len(index)}] {answer[:20]:20s} 用时{t1-t0:.1f}s 正确={correct}")

t_global_end = time.time()
print(f"推理完成, 总耗时 {t_global_end - t_global_start:.0f}s")

# ---------- 停止 tegrastats ----------
os.killpg(os.getpgid(teg.pid), signal.SIGTERM)
time.sleep(1)

# ---------- 解析功率日志 ----------
# tegrastats 行里找 VDD_GPU_SOC 或 VDD_CPU_GPU_CV (mW)。先把每行的墙钟时间和功率提取出来
def parse_power(logfile):
    import re
    rows = []
    # tegrastats 自带时间戳格式: MM-DD-YYYY HH:MM:SS
    base_epoch = None
    with open(logfile) as f:
        for line in f:
            # 提取功率: 优先 VDD_GPU_SOC, 退而求其次 VDD_CPU_GPU_CV / POM_5V_GPU
            m = re.search(r"(VDD_GPU_SOC|VDD_CPU_GPU_CV|POM_5V_GPU)\s+(\d+)mW", line)
            if m:
                rows.append(int(m.group(2)))
    return rows

powers = parse_power(POWER_LOG)
print(f"功率采样点数: {len(powers)}")
print(f"tegrastats 字段样例(检查用):")
os.system(f"head -2 {POWER_LOG}")

# ---------- 时间窗对齐(近似: 按采样间隔均匀映射) ----------
# 因为 tegrastats 50ms 一个点, 我们用全程时间轴线性映射每张图的时间窗
df_rec = pd.DataFrame(records)
total_dur = t_global_end - t_global_start
n_samples = len(powers)
if n_samples > 0:
    sample_dt = total_dur / n_samples  # 每个采样点代表的秒数
    powers_arr = np.array(powers, dtype=float)  # mW

    def window_stats(t0, t1):
        i0 = int((t0 - t_global_start) / sample_dt)
        i1 = int((t1 - t_global_start) / sample_dt)
        i0, i1 = max(0, i0), min(n_samples, max(i1, i0+1))
        seg = powers_arr[i0:i1]
        if len(seg) == 0:
            return np.nan, np.nan, np.nan
        avg_mW = seg.mean()
        peak_mW = np.percentile(seg, 95)   # P̂^CP: 95分位峰值
        return avg_mW, peak_mW, len(seg)

    energies, peaks = [], []
    for _, rr in df_rec.iterrows():
        avg_mW, peak_mW, npts = window_stats(rr["t_start"], rr["t_end"])
        T = rr["latency_s"]
        E_J = (avg_mW / 1000.0) * T        # 能耗(焦耳) = 平均功率(W) × 时间(s)
        energies.append(E_J)
        peaks.append(peak_mW / 1000.0)     # 转 W
    df_rec["E_im_J"] = energies
    df_rec["Ppeak_W"] = peaks

df_rec.to_csv(OUT_PER_IMG, index=False)

# ---------- 汇总 ----------
summary = {
    "model": MODEL_NAME,
    "n_images": len(df_rec),
    "accuracy": df_rec["correct"].mean(),
    "T_im_mean_s": df_rec["latency_s"].mean(),
    "T_im_p50_s": df_rec["latency_s"].median(),
    "E_im_mean_J": df_rec["E_im_J"].mean() if "E_im_J" in df_rec else np.nan,
    "Ppeak_p95_W": df_rec["Ppeak_W"].max() if "Ppeak_W" in df_rec else np.nan,
}
pd.DataFrame([summary]).to_csv(OUT_SUMMARY, index=False)
print("\n===== 汇总 =====")
for k, v in summary.items():
    print(f"  {k}: {v}")
print(f"\nper-image 存到 {OUT_PER_IMG}")
print(f"summary 存到 {OUT_SUMMARY}")
