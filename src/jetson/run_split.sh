#!/bin/bash
# P0-1: start rpc-server (rear device) + llama-server (front, --rpc) split 7B
B=/home/htj/build-rpc/bin
M=/home/htj/Downloads/models
pkill -9 -f llama-server; pkill -9 -f rpc-server; sleep 4
echo "PHASE rpc-server start $(date +%H:%M:%S)" > ~/split_phase.log
setsid $B/rpc-server -H 127.0.0.1 -p 50052 > ~/rpc_server.log 2>&1 < /dev/null &
disown; sleep 5
echo "PHASE llama-server start" >> ~/split_phase.log
setsid $B/llama-server -m $M/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf \
  --mmproj $M/mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf \
  --rpc 127.0.0.1:50052 --tensor-split 0.5,0.5 \
  -ngl 99 -c 2048 --host 0.0.0.0 --port 8080 > ~/llama_split.log 2>&1 < /dev/null &
disown
for i in $(seq 1 120); do curl -s http://127.0.0.1:8080/health 2>/dev/null | grep -q ok && { echo "HEALTH_OK $i" >> ~/split_phase.log; break; }; sleep 3; done
tail -1 ~/split_phase.log
