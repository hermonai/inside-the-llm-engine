#!/bin/bash
# Part V rerun of the MUL_MAT_ID timing with SQL output (the CSV printer omits
# time_us), after run_part5d.sh reports "done5d".
D=<scratch>
M=$D/meas5
OUT=$M/out
cd $M
log(){ echo "$(date -u +%FT%TZ) $*" >> $OUT/progress.log; }
gate(){ for i in $(seq 1 20); do l=$(sysctl -n vm.loadavg | awk '{print $2}'); awk -v l="$l" 'BEGIN{exit !(l < 6)}' && return; log "waiting: load1=$l before $1"; sleep 30; done; log "gate timed out before $1; running anyway"; }
cond(){ echo "--- $1: $(date -u +%FT%TZ) load $(sysctl -n vm.loadavg) $(sysctl -n vm.swapusage)" >> $OUT/conditions.log; }
until grep -q "done5d" $OUT/progress.log 2>/dev/null; do sleep 30; done
TBO=$D/llama-build/bin/test-backend-ops
for r in 1 2 3; do
  for t in q4_K f16; do
    gate moesql-$t-$r; cond moesql-$t-$r
    $TBO perf -o MUL_MAT_ID -b MTL0 -p "type_a=$t,type_b=f32,n_mats=128,n_used=8,.*m=768,.*k=2048" \
      --output sql > $OUT/moe-$t-$r.sql 2> $OUT/moe-$t-$r.sql.err; log "moesql $t run $r exit=$?"
  done
done
cond end5e; log "done5e"
