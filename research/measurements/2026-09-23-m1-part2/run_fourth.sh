#!/bin/bash
# After the confirmation pass: attention under an allocator cap, one process per
# (variant, length), then the dispatch-cost probe with torch.compile.
S=<scratch>
cd $S
log(){ echo "$(date -u +%FT%TZ) $*" >> $S/progress2.log; }
until grep -q ' done$' $S/progress2.log 2>/dev/null; do sleep 20; done
echo "--- attention2: $(date -u +%FT%TZ) load $(sysctl -n vm.loadavg) $(sysctl -n vm.swapusage)" >> $S/conditions2.log
: > $S/attention_memory2.jsonl
for n in 1024 2048 4096 8192 16384 32768; do
  for v in naive sdpa tiled; do
    python3 attention_memory2.py $v $n >> attention_memory2.jsonl 2>> attention_memory2.err; log "attn2 $v $n exit=$?"
  done
done
echo "--- launch2: $(date -u +%FT%TZ) load $(sysctl -n vm.loadavg) $(sysctl -n vm.swapusage)" >> $S/conditions2.log
python3 launch_overhead2.py > launch_overhead2.json 2> launch_overhead2.err; log "launch2 exit=$?"
echo "--- end4: $(date -u +%FT%TZ) load $(sysctl -n vm.loadavg) $(sysctl -n vm.swapusage)" >> $S/conditions2.log
log done4
