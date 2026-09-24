#!/bin/bash
# Part VI repeat of the RPC split, after run_part6.sh reports "done": the three
# configurations interleaved (local, split, remote, local, split, remote) so that
# load drift affects all of them alike.
D=<scratch>
M=$D/meas6
OUT=$M/out
BR=$D/llama-build-rpc
BL=$HOME/.ollama/models/blobs
L3=$BL/sha256-dde5aa3fc5ffc17176b5e8bdc82f587b24b2678c6c66101bf7da77af9f7ccdff
Q8=$BL/sha256-e6a7edc1a4d7d9b2de136a221a57336b76316cfe53a252aeba814496c5ae439d
cd $M
log(){ echo "$(date -u +%FT%TZ) $*" >> $OUT/progress.log; }
gate(){ for i in $(seq 1 20); do l=$(sysctl -n vm.loadavg | awk '{print $2}'); awk -v l="$l" 'BEGIN{exit !(l < 6)}' && return; log "waiting: load1=$l before $1"; sleep 30; done; log "gate timed out before $1; running anyway"; }
cond(){ echo "--- $1: $(date -u +%FT%TZ) load $(sysctl -n vm.loadavg) $(sysctl -n vm.swapusage)" >> $OUT/conditions.log; }
until grep -q " done$" $OUT/progress.log 2>/dev/null; do sleep 30; done
$BR/bin/rpc-server -p 50052 -d MTL0 > $OUT/rpc-server-2.log 2>&1 &
RPCPID=$!
sleep 5
for m in L3 Q8; do
  eval MP=\$$m
  for round in 1 2; do
    for cfg in local split remote; do
      case $cfg in
        local)  ARGS="" ;;
        split)  ARGS="--rpc localhost:50052 -ts 1/1" ;;
        remote) ARGS="--rpc localhost:50052 -ts 1/0" ;;
      esac
      gate rpc2-$m-$cfg-$round; cond rpc2-$m-$cfg-$round
      $BR/bin/llama-bench -m $MP -fa 1 -p 512 -n 64 -r 2 $ARGS -o json > $OUT/rpc2-$m-$cfg-$round.json 2> $OUT/rpc2-$m-$cfg-$round.err
      log "rpc2 $m $cfg $round exit=$?"
    done
  done
done
kill $RPCPID; wait $RPCPID 2>/dev/null
cond end6b; log "done6b"
