#!/bin/bash
# Part V probes, one at a time; each waits (up to 10 min) for a one-minute load
# average below 6, then runs and records the conditions.
D=<scratch>
OUT=$D/out
B=$HOME/.ollama/models/blobs
mkdir -p $OUT
cd $D
log(){ echo "$(date -u +%FT%TZ) $*" >> $OUT/progress.log; }
gate(){ for i in $(seq 1 20); do l=$(sysctl -n vm.loadavg | awk '{print $2}'); awk -v l="$l" 'BEGIN{exit !(l < 6)}' && return; log "waiting: load1=$l before $1"; sleep 30; done; log "gate timed out before $1; running anyway"; }
cond(){ echo "--- $1: $(date -u +%FT%TZ) load $(sysctl -n vm.loadavg) $(sysctl -n vm.swapusage)" >> $OUT/conditions.log; }
gate kvsizes; cond kvsizes
python3 part5_probe.py kvsizes $OUT > $OUT/kvsizes.stdout 2> $OUT/kvsizes.stderr; log "kvsizes exit=$?"
for m in dde5aa3fc5ffc17176b5e8bdc82f587b24b2678c6c66101bf7da77af9f7ccdff 4a188102020e9c9530b687fd6400f775c45e90a0d7baafe65bd0a36963fbb7ba e6a7edc1a4d7d9b2de136a221a57336b76316cfe53a252aeba814496c5ae439d 2bfb097d01f8633cd5fcf6cc3aeeeded7c7211b41e9d241e871d7d595a42fc49; do
  gate depth-$m; cond depth-$m
  llama-bench -m $B/sha256-$m -fa 1 -p 0 -n 64 -d 0,4096,16384 -r 2 -o json > $OUT/depth-$m.json 2> $OUT/depth-$m.err; log "depth $m exit=$?"
  llama-bench -m $B/sha256-$m -fa 1 -p 512 -n 0 -r 3 -o json > $OUT/pp-$m.json 2> $OUT/pp-$m.err; log "pp $m exit=$?"
done
for e in reasoning ollama_ctx vision; do
  gate $e; cond $e
  python3 part5_probe.py $e $OUT > $OUT/$e.stdout 2> $OUT/$e.stderr; log "$e exit=$?"
done
cond end; log done
