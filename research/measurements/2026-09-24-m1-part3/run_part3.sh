#!/bin/bash
# Part III probes, one at a time, each started only when the one-minute load average
# is below 5 (up to 10 minutes of waiting); conditions logged before each.
D=<scratch>
OUT=$D/out
mkdir -p $OUT
cd $D
log(){ echo "$(date -u +%FT%TZ) $*" >> $OUT/progress.log; }
for e in lifecycle contbatch kvlayout prefix chunked overload structured lora; do
  for i in $(seq 1 20); do
    l=$(sysctl -n vm.loadavg | awk '{print $2}')
    awk -v l="$l" 'BEGIN{exit !(l < 5)}' && break
    log "waiting: load1=$l before $e"; sleep 30
  done
  echo "--- $e: $(date -u +%FT%TZ) load $(sysctl -n vm.loadavg) $(sysctl -n vm.swapusage)" >> $OUT/conditions.log
  python3 serve_probe.py $e $OUT > $OUT/$e.stdout 2> $OUT/$e.stderr; log "$e exit=$?"
done
echo "--- end: $(date -u +%FT%TZ) load $(sysctl -n vm.loadavg) $(sysctl -n vm.swapusage)" >> $OUT/conditions.log
log done
