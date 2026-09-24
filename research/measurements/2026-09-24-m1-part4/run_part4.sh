#!/bin/bash
# Part IV probes, one at a time. Each waits (up to 10 min) for a one-minute load
# average below 6, then runs regardless and records the conditions: the machine's
# load comes from the owner's own work (a Docker VM), and every comparison here
# alternates or pairs its baseline and speculative runs.
D=<scratch>
OUT=$D/out
B=$HOME/.ollama/models/blobs
M3=$B/sha256-dde5aa3fc5ffc17176b5e8bdc82f587b24b2678c6c66101bf7da77af9f7ccdff
M8=$B/sha256-e6a7edc1a4d7d9b2de136a221a57336b76316cfe53a252aeba814496c5ae439d
MD=$B/sha256-4a188102020e9c9530b687fd6400f775c45e90a0d7baafe65bd0a36963fbb7ba
mkdir -p $OUT
cd $D
log(){ echo "$(date -u +%FT%TZ) $*" >> $OUT/progress.log; }
gate(){ for i in $(seq 1 20); do l=$(sysctl -n vm.loadavg | awk '{print $2}'); awk -v l="$l" 'BEGIN{exit !(l < 6)}' && return; log "waiting: load1=$l before $1"; sleep 30; done; log "gate timed out before $1; running anyway"; }
cond(){ echo "--- $1: $(date -u +%FT%TZ) load $(sysctl -n vm.loadavg) $(sysctl -n vm.swapusage)" >> $OUT/conditions.log; }
gate verify; cond verify
llama-bench -m $M3 -p 1,2,3,4,5,6,8,12,16,32 -n 0 -r 5 -o json > $OUT/verify-3b.json 2> $OUT/verify-3b.err; log "verify-3b exit=$?"
llama-bench -m $M8 -p 1,2,3,4,5,6,8,12,16,32 -n 0 -r 5 -o json > $OUT/verify-8b.json 2> $OUT/verify-8b.err; log "verify-8b exit=$?"
llama-bench -m $MD -p 1 -n 0 -r 5 -o json > $OUT/verify-draft.json 2> $OUT/verify-draft.err; log "verify-draft exit=$?"
for e in ngram exact concurrent draftmodel; do
  gate $e; cond $e
  python3 spec_probe.py $e $OUT > $OUT/$e.stdout 2> $OUT/$e.stderr; log "$e exit=$?"
done
cond end; log done
