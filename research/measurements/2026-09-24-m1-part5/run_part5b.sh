#!/bin/bash
# Part V follow-ups, run after run_part5.sh reports "done": the hybrid prefix-reuse
# probe, then llama.cpp's MUL_MAT_ID (MoE expert matmul) kernel timed on the GPU at
# Qwen3-30B-A3B's expert shapes, built from the same commit as the Homebrew binaries.
D=<scratch>
M=$D/meas5
OUT=$M/out
cd $M
log(){ echo "$(date -u +%FT%TZ) $*" >> $OUT/progress.log; }
gate(){ for i in $(seq 1 20); do l=$(sysctl -n vm.loadavg | awk '{print $2}'); awk -v l="$l" 'BEGIN{exit !(l < 6)}' && return; log "waiting: load1=$l before $1"; sleep 30; done; log "gate timed out before $1; running anyway"; }
cond(){ echo "--- $1: $(date -u +%FT%TZ) load $(sysctl -n vm.loadavg) $(sysctl -n vm.swapusage)" >> $OUT/conditions.log; }
until grep -q " done$" $OUT/progress.log 2>/dev/null; do sleep 60; done
gate hybrid_prefix; cond hybrid_prefix
python3 part5b_probe.py hybrid_prefix $OUT > $OUT/hybrid_prefix.stdout 2> $OUT/hybrid_prefix.stderr; log "hybrid_prefix exit=$?"
log "building test-backend-ops"
cmake -S $D/llama-src -B $D/llama-build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TOOLS=OFF > $OUT/build.log 2>&1
cmake --build $D/llama-build --target test-backend-ops -j 4 >> $OUT/build.log 2>&1; log "build exit=$?"
TBO=$D/llama-build/bin/test-backend-ops
for r in 1 2 3; do
  for t in q4_K f16; do
    gate moe-$t-$r; cond moe-$t-$r
    $TBO perf -o MUL_MAT_ID -b MTL0 -p "type_a=$t,type_b=f32,n_mats=128,n_used=8,.*m=768,.*k=2048" \
      --output csv > $OUT/moe-$t-$r.csv 2> $OUT/moe-$t-$r.err; log "moe $t run $r exit=$?"
  done
done
cond end5b; log "done5b"
