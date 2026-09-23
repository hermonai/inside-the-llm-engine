#!/bin/bash
# Confirmation pass: waits for the load average to settle, then repeats the
# quantization sweep and the cache-type comparison with more repetitions.
S=<scratch>
M=$HOME/.ollama/models/blobs/sha256-dde5aa3fc5ffc17176b5e8bdc82f587b24b2678c6c66101bf7da77af9f7ccdff
cd $S
log(){ echo "$(date -u +%FT%TZ) $*" >> $S/progress2.log; }
cond(){ echo "--- $1: $(date -u +%FT%TZ) load $(sysctl -n vm.loadavg) $(sysctl -n vm.swapusage)" >> $S/conditions2.log; }
for i in $(seq 1 20); do
  l=$(sysctl -n vm.loadavg | awk '{print $2}'); log "load1=$l"
  awk -v l="$l" 'BEGIN{exit !(l < 6)}' && break
  sleep 30
done
cond quant2
for q in Q8_0 Q6_K Q5_K_M Q4_K_M Q3_K_M Q2_K; do
  f=$S/llama32-3b-$q.gguf; [ $q = Q4_K_M ] && f=$M
  llama-bench -m $f -p 512 -n 128 -r 5 -o json > $S/bench2-$q.json 2> $S/bench2-$q.err; log "bench2 $q exit=$?"
done
cond kv2
for ct in f16 q8_0 q4_0; do
  llama-bench -m $M -fa 1 -ctk $ct -ctv $ct -p 0 -n 64 -d 0,8192 -r 3 -o json > $S/kv2-$ct.json 2> $S/kv2-$ct.err; log "kv2 $ct exit=$?"
done
cond end2
log done
