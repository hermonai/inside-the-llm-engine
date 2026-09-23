#!/bin/bash
S=<scratch>
M=$HOME/.ollama/models/blobs/sha256-dde5aa3fc5ffc17176b5e8bdc82f587b24b2678c6c66101bf7da77af9f7ccdff
cd $S
log(){ echo "$(date -u +%FT%TZ) $*" >> $S/progress.log; }
cond(){ echo "--- $1: $(date -u +%FT%TZ) load $(sysctl -n vm.loadavg) $(sysctl -n vm.swapusage)" >> $S/conditions.log; }
log start
# Ch 10: requantize to other K-quant and 8-bit formats (speed only; values come from Q4_K_M)
for q in Q8_0 Q6_K Q5_K_M Q3_K_M Q2_K; do
  llama-quantize --allow-requantize $M $S/llama32-3b-$q.gguf $q > $S/quant-$q.log 2>&1; log "quantize $q exit=$?"
done
cond bench-quant
for q in Q4_K_M Q8_0 Q6_K Q5_K_M Q3_K_M Q2_K; do
  f=$S/llama32-3b-$q.gguf; [ $q = Q4_K_M ] && f=$M
  llama-bench -m $f -p 512 -n 128 -r 3 -o json > $S/bench-$q.json 2> $S/bench-$q.err; log "bench $q exit=$?"
done
# Ch 9 and 12: decode at depth, cache types, flash attention on/off
cond bench-kv
for ct in f16 q8_0 q4_0; do
  llama-bench -m $M -fa 1 -ctk $ct -ctv $ct -p 0 -n 64 -d 0,4096,8192,16384 -r 2 -o json > $S/kv-$ct.json 2> $S/kv-$ct.err; log "kv $ct exit=$?"
done
llama-bench -m $M -fa 0 -p 0 -n 64 -d 0,4096,8192,16384 -r 2 -o json > $S/kv-f16-nofa.json 2> $S/kv-f16-nofa.err; log "kv f16 nofa exit=$?"
# Ch 7, 8, 13: GPU and CPU probes
cond torch
python3 gemm_shapes.py > gemm_shapes.json 2> gemm_shapes.err; log "gemm exit=$?"
python3 attention_memory.py > attention_memory.json 2> attention_memory.err; log "attention exit=$?"
python3 launch_overhead.py > launch_overhead.json 2> launch_overhead.err; log "launch exit=$?"
cond end
log done
