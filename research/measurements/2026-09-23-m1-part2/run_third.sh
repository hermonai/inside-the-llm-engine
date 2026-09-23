#!/bin/bash
# GEMM shape probe (fixed oracle), then the confirmation pass; sequential so
# nothing contends for the GPU.
S=<scratch>
cd $S
log(){ echo "$(date -u +%FT%TZ) $*" >> $S/progress2.log; }
for i in $(seq 1 20); do
  l=$(sysctl -n vm.loadavg | awk '{print $2}'); log "pre-gemm load1=$l"
  awk -v l="$l" 'BEGIN{exit !(l < 4)}' && break
  sleep 30
done
echo "--- gemm: $(date -u +%FT%TZ) load $(sysctl -n vm.loadavg) $(sysctl -n vm.swapusage)" >> $S/conditions2.log
python3 gemm_shapes.py > gemm_shapes.json 2> gemm_shapes.err; log "gemm exit=$?"
bash $S/run_second.sh
