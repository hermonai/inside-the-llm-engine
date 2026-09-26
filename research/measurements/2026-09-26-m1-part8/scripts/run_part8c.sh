#!/bin/bash
# Part VIII-IX follow-ups after run_part8b.sh: flash-attention A/B (Ch 39), the capstone's
# roofline (Ch 42), a Hermon release build and the three-engine comparison (Ch 41).
cd "$(dirname "$0")"
until grep -q "part8b done" out/run.log 2>/dev/null; do sleep 10; done
gate() { local n=0; while [ "$(sysctl -n vm.loadavg | awk '{print ($2 < 6.0)}')" != "1" ] && [ $n -lt 30 ]; do sleep 10; n=$((n+1)); done
  echo "$(date -u +%FT%TZ) start $1 load=$(sysctl -n vm.loadavg) swap=$(sysctl -n vm.swapusage | awk '{print $6}')" >> out/run.log; }
L=$HOME/.ollama/models/blobs/sha256-dde5aa3fc5ffc17176b5e8bdc82f587b24b2678c6c66101bf7da77af9f7ccdff
# 1. flash attention off vs on, interleaved (Ch 39)
gate fa-ab; mkdir -p out/fa-ab
for r in 1 2 3 4; do
  for fa in 0 1; do
    llama-bench -m "$L" -fa $fa -p 0 -n 128 -r 3 -o json > out/fa-ab/r$r-fa$fa.json 2> out/fa-ab/r$r-fa$fa.stderr.txt
  done
done
echo "$(date -u +%FT%TZ) end fa-ab" >> out/run.log
# 2. the capstone on the roofline (Ch 42)
ME=<repo>/code/mini-engine
gate capstone
( cd "$ME" && cargo build --release -p capstone --example capstone_roofline ) > out/capstone-build.txt 2>&1
( cd "$ME" && ./target/release/examples/capstone_roofline "$PWD/out/capstone-roofline.json" ) > out/capstone-roofline.txt 2>&1
echo "$(date -u +%FT%TZ) end capstone rc=$?" >> out/run.log
# 3. Hermon release build with Metal, then the three-engine comparison (Ch 41)
H=<home>/ClaudeProjects/hermon
gate hermon-build
( cd "$H" && git rev-parse --short HEAD && /usr/bin/time -p cargo build --release -p hermon-cli --features engine-metal ) > out/hermon-build.txt 2>&1
echo "$(date -u +%FT%TZ) end hermon-build rc=$?" >> out/run.log
gate engines3; python3 engines3.py out/engines3 "$H/target/release/hermon" > out/engines3.log 2>&1
echo "$(date -u +%FT%TZ) end engines3 rc=$?" >> out/run.log
echo "$(date -u +%FT%TZ) part8c done" >> out/run.log
