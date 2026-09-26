#!/bin/bash
# Part VIII probes, sequential, each started when the one-minute load is below 6 (5 min max wait).
cd "$(dirname "$0")"
gate() { local n=0; while [ "$(sysctl -n vm.loadavg | awk '{print ($2 < 6.0)}')" != "1" ] && [ $n -lt 30 ]; do sleep 10; n=$((n+1)); done
  echo "$(date -u +%FT%TZ) start $1 load=$(sysctl -n vm.loadavg) swap=$(sysctl -n vm.swapusage | awk '{print $6}')" >> out/run.log; }
L=$HOME/.ollama/models/blobs/sha256-dde5aa3fc5ffc17176b5e8bdc82f587b24b2678c6c66101bf7da77af9f7ccdff
G=$HOME/.ollama/models/blobs/sha256-4cbc52994d8ce56d58f3ecadcd451a5dbb2a4f1142098c6b9f030d18ee5e052b
gate batch-llama;   ./probe8 batch "$L" target.txt fillers.txt 256 out/batch-llama32-3b.json 1 2 3 4 8 16 32 64 > out/batch-llama32-3b.log 2>&1; echo "$(date -u +%FT%TZ) end batch-llama rc=$?" >> out/run.log
gate batch-granite; ./probe8 batch "$G" target.txt fillers.txt 256 out/batch-granite.json 1 2 3 4 8 16 31 32 48 64 > out/batch-granite.log 2>&1; echo "$(date -u +%FT%TZ) end batch-granite rc=$?" >> out/run.log
gate route-granite; ./probe8 route "$G" route_prompts.txt 384 out/route-granite.json > out/route-granite.log 2>&1; echo "$(date -u +%FT%TZ) end route-granite rc=$?" >> out/run.log
echo "$(date -u +%FT%TZ) all done" >> out/run.log
