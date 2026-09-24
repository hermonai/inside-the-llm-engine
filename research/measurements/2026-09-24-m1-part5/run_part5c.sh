#!/bin/bash
# Part V rerun of the Ollama context experiment, after run_part5b.sh reports "done5b".
M=<scratch>
OUT=$M/out
cd $M
log(){ echo "$(date -u +%FT%TZ) $*" >> $OUT/progress.log; }
gate(){ for i in $(seq 1 20); do l=$(sysctl -n vm.loadavg | awk '{print $2}'); awk -v l="$l" 'BEGIN{exit !(l < 6)}' && return; log "waiting: load1=$l before $1"; sleep 30; done; log "gate timed out before $1; running anyway"; }
cond(){ echo "--- $1: $(date -u +%FT%TZ) load $(sysctl -n vm.loadavg) $(sysctl -n vm.swapusage)" >> $OUT/conditions.log; }
until grep -q "done5b" $OUT/progress.log 2>/dev/null; do sleep 60; done
gate ollama_ctx2; cond ollama_ctx2
python3 part5c_probe.py ollama_ctx2 $OUT > $OUT/ollama_ctx2.stdout 2> $OUT/ollama_ctx2.stderr; log "ollama_ctx2 exit=$?"
cond end5c; log "done5c"
