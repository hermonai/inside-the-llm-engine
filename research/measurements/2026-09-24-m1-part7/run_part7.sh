#!/bin/bash
# Part VII measurements, one at a time, each started only when the machine's
# one-minute load average is below 6 (or after 20 minutes of waiting).
set -u
cd "$(dirname "$0")"
OUT="$PWD/out"
mkdir -p "$OUT"
LOG="$OUT/run.log"
gate() {
  local n=0
  while [ "$(sysctl -n vm.loadavg | awk '{print ($2 < 6.0)}')" != "1" ] && [ $n -lt 80 ]; do
    sleep 15; n=$((n+1))
  done
  echo "$(date -u +%FT%TZ) start $1 load=$(sysctl -n vm.loadavg) swap=$(sysctl -n vm.swapusage | awk '{print $6}')" >> "$LOG"
}
for e in offload contention gpucopy mmapload; do
  gate "$e"
  python3 part7_probe.py "$e" "$OUT" >> "$LOG" 2>&1
  echo "$(date -u +%FT%TZ) end $e rc=$?" >> "$LOG"
done
gate hermon-selftest
HK=<home>/ClaudeProjects/hermon/crates/hermon-kernels
( cd "$HK" && git rev-parse --short HEAD && git status --short | wc -l ) > "$OUT/hermon-rev.txt" 2>&1
for m in none ubsan; do
  t0=$(date +%s)
  ( cd "$HK" && ./sanitize.sh "$m" ) > "$OUT/hermon-selftest-$m.txt" 2>&1
  rc=$?
  echo "mode=$m rc=$rc seconds=$(( $(date +%s) - t0 ))" >> "$OUT/hermon-rev.txt"
done
echo "$(date -u +%FT%TZ) all done" >> "$LOG"
