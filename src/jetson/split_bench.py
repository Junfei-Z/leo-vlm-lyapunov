#!/usr/bin/env python3
"""P0-1 split micro-benchmark client (runs ON the Jetson).
Measures, per inference through the RPC-split 7B:
  - loopback bytes (rx on lo) per inference  -> RPC traffic = activations + protocol
  - end-to-end latency per image
  - answers (for the correctness gate vs the monolithic per-image CSV)
Warm-up first (weight upload to the RPC device happens at load, excluded).
Usage: python3 split_bench.py <N_IMAGES> <TAG> [MAX_TOKENS]
"""
import sys, time, base64, json, csv, urllib.request

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
TAG = sys.argv[2] if len(sys.argv) > 2 else "split"
MAXTOK = int(sys.argv[3]) if len(sys.argv) > 3 else 24
IDX = "/home/htj/resisc45/resisc45_index_full.csv"
CLS = "/home/htj/resisc45/raw/classnames.txt"
URL = "http://127.0.0.1:8080/v1/chat/completions"

classes = [c.strip() for c in open(CLS) if c.strip()]
PROMPT = ("You are an expert in remote-sensing image interpretation. Examine this "
          "satellite/aerial image and classify the scene into exactly ONE of the "
          "following categories:\n" + ", ".join(c.replace("_", " ") for c in classes) +
          ".\nRespond with only the exact category name from the list, nothing else.")

def lo_bytes():
    for line in open("/proc/net/dev"):
        if line.strip().startswith("lo:"):
            f = line.split()
            return int(f[1])  # rx bytes (loopback: rx == tx)
    raise RuntimeError("no lo")

def ask(path):
    b64 = base64.b64encode(open(path, "rb").read()).decode()
    payload = json.dumps({"messages": [{"role": "user", "content": [
        {"type": "text", "text": PROMPT},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}}]}],
        "temperature": 0, "max_tokens": MAXTOK}).encode()
    req = urllib.request.Request(URL, data=payload,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as resp:
        out = json.loads(resp.read())
    t1 = time.time()
    return out["choices"][0]["message"]["content"].strip(), t1 - t0, len(payload)

rows = list(csv.DictReader(open(IDX)))[:N + 2]
# warm-up: 2 images (graph allocation, any lazy weight upload)
for r in rows[:2]:
    ask(r["path"])
print("warmed up; measuring", N, "images, max_tokens=", MAXTOK)

recs = []
for r in rows[2:N + 2]:
    b0 = lo_bytes()
    ans, dt, req_bytes = ask(r["path"])
    b1 = lo_bytes()
    # subtract the HTTP request itself (image upload goes over lo too)
    rpc = b1 - b0 - req_bytes
    recs.append((r["path"].split("/")[-1], r["class_name"], ans.replace("\n", " ")[:40],
                 dt, rpc))
    print(f"{recs[-1][0][:28]:30s} {dt:6.2f}s  rpcB={rpc/1e6:7.3f}MB  ans={recs[-1][2]}")

lat = [x[3] for x in recs]; rpc = [x[4] for x in recs]
print(f"== TAG={TAG} N={N} maxtok={MAXTOK}")
print(f"latency  mean={sum(lat)/len(lat):.3f}s  min={min(lat):.3f}  max={max(lat):.3f}")
print(f"rpcBytes mean={sum(rpc)/len(rpc)/1e6:.3f}MB min={min(rpc)/1e6:.3f} max={max(rpc)/1e6:.3f}")
with open(f"/home/htj/split_bench_{TAG}.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["img", "true", "answer", "latency_s", "rpc_bytes"])
    w.writerows(recs)
print("saved /home/htj/split_bench_%s.csv" % TAG)
