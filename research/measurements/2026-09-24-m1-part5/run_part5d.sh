#!/bin/bash
# Part V follow-up: hybrid prefix reuse with a 512-token batch, after run_part5c.sh reports "done5c".
M=<scratch>
OUT=$M/out
cd $M
log(){ echo "$(date -u +%FT%TZ) $*" >> $OUT/progress.log; }
gate(){ for i in $(seq 1 20); do l=$(sysctl -n vm.loadavg | awk '{print $2}'); awk -v l="$l" 'BEGIN{exit !(l < 6)}' && return; log "waiting: load1=$l before $1"; sleep 30; done; log "gate timed out before $1; running anyway"; }
cond(){ echo "--- $1: $(date -u +%FT%TZ) load $(sysctl -n vm.loadavg) $(sysctl -n vm.swapusage)" >> $OUT/conditions.log; }
until grep -q "done5c" $OUT/progress.log 2>/dev/null; do sleep 30; done
gate hybrid_prefix_b512; cond hybrid_prefix_b512
python3 part5b_probe.py hybrid_prefix_b512 $OUT > $OUT/hybrid_prefix_b512.stdout 2> $OUT/hybrid_prefix_b512.stderr; log "hybrid_prefix_b512 exit=$?"
cond end5d; log "done5d"
