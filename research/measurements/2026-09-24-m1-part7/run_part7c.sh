#!/bin/bash
# Second repack round (interleaved differently), then Hermon's Metal-vs-CPU attention
# benchmark at 2a3fd52 on this M1, each started when the one-minute load is below 6.
cd "$(dirname "$0")"
echo "$(date -u +%FT%TZ) start repack2 load=$(sysctl -n vm.loadavg) swap=$(sysctl -n vm.swapusage | awk '{print $6}')" >> out/run.log
python3 repack2.py out/repack > out/repack2.log 2>&1
echo "$(date -u +%FT%TZ) end repack2" >> out/run.log
n=0; while [ "$(sysctl -n vm.loadavg | awk '{print ($2 < 6.0)}')" != "1" ] && [ $n -lt 30 ]; do sleep 10; n=$((n+1)); done
echo "$(date -u +%FT%TZ) start metalbench load=$(sysctl -n vm.loadavg)" >> out/run.log
HK=<home>/ClaudeProjects/hermon/crates/hermon-kernels
( cd "$HK" && git rev-parse --short HEAD && git status --short | wc -l && ./bench.sh ) > out/hermon-metalbench.txt 2>&1
echo "$(date -u +%FT%TZ) end metalbench rc=$?" >> out/run.log
