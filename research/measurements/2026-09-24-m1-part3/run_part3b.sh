#!/bin/bash
D=<scratch>
OUT=$D/out
cd $D
log(){ echo "$(date -u +%FT%TZ) $*" >> $OUT/progress.log; }
until grep -q ' done$' $OUT/progress.log 2>/dev/null; do sleep 15; done
for e in cancel2; do
  for i in $(seq 1 20); do
    l=$(sysctl -n vm.loadavg | awk '{print $2}')
    awk -v l="$l" 'BEGIN{exit !(l < 5)}' && break
    log "waiting: load1=$l before $e"; sleep 30
  done
  echo "--- $e: $(date -u +%FT%TZ) load $(sysctl -n vm.loadavg) $(sysctl -n vm.swapusage)" >> $OUT/conditions.log
  python3 serve_probe.py $e $OUT > $OUT/$e.stdout 2> $OUT/$e.stderr; log "$e exit=$?"
done
log done-b
