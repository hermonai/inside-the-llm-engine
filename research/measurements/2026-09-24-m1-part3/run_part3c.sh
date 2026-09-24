#!/bin/bash
D=<scratch>
OUT=$D/out
cd $D
log(){ echo "$(date -u +%FT%TZ) $*" >> $OUT/progress.log; }
until grep -q ' done-b$' $OUT/progress.log 2>/dev/null; do sleep 10; done
for i in $(seq 1 20); do
  l=$(sysctl -n vm.loadavg | awk '{print $2}')
  awk -v l="$l" 'BEGIN{exit !(l < 5)}' && break
  log "waiting: load1=$l before ollama"; sleep 30
done
echo "--- ollama: $(date -u +%FT%TZ) load $(sysctl -n vm.loadavg) $(sysctl -n vm.swapusage)" >> $OUT/conditions.log
python3 ollama_switch.py $OUT/ollama_switch.json > $OUT/ollama_switch.stdout 2> $OUT/ollama_switch.stderr; log "ollama exit=$?"
log done-c
