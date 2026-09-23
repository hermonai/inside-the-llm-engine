#!/bin/bash
# Long-context confirmation of the cache-type sweep: the first pass ran at load 12-16.
S=<scratch>
M=$HOME/.ollama/models/blobs/sha256-dde5aa3fc5ffc17176b5e8bdc82f587b24b2678c6c66101bf7da77af9f7ccdff
cd $S
log(){ echo "$(date -u +%FT%TZ) $*" >> $S/progress2.log; }
until grep -q ' done4$' $S/progress2.log 2>/dev/null; do sleep 20; done
echo "--- kv3: $(date -u +%FT%TZ) load $(sysctl -n vm.loadavg) $(sysctl -n vm.swapusage)" >> $S/conditions2.log
for ct in f16 q8_0 q4_0; do
  llama-bench -m $M -fa 1 -ctk $ct -ctv $ct -p 0 -n 64 -d 4096,16384 -r 3 -o json > $S/kv3-$ct.json 2> $S/kv3-$ct.err; log "kv3 $ct exit=$?"
done
echo "--- end5: $(date -u +%FT%TZ) load $(sysctl -n vm.loadavg) $(sysctl -n vm.swapusage)" >> $S/conditions2.log
log done5
