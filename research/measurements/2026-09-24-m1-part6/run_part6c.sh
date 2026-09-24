#!/bin/bash
# Part VI placement check, after run_part6b.sh reports "done6b": one short verbose
# llama-bench run per configuration, to record which device holds how many bytes.
D=<scratch>
M=$D/meas6
OUT=$M/out
BR=$D/llama-build-rpc
BL=$HOME/.ollama/models/blobs
L3=$BL/sha256-dde5aa3fc5ffc17176b5e8bdc82f587b24b2678c6c66101bf7da77af9f7ccdff
cd $M
log(){ echo "$(date -u +%FT%TZ) $*" >> $OUT/progress.log; }
until grep -q "done6b" $OUT/progress.log 2>/dev/null; do sleep 30; done
$BR/bin/rpc-server -p 50052 -d MTL0 > $OUT/rpc-server-3.log 2>&1 &
RPCPID=$!
sleep 5
for cfg in local split remote; do
  case $cfg in
    local)  ARGS="" ;;
    split)  ARGS="--rpc localhost:50052 -ts 1/1" ;;
    remote) ARGS="--rpc localhost:50052 -ts 1/0" ;;
  esac
  $BR/bin/llama-bench -v -m $L3 -fa 1 -p 16 -n 4 -r 1 $ARGS > $OUT/placement-$cfg.out 2> $OUT/placement-$cfg.err
  log "placement $cfg exit=$?"
done
kill $RPCPID; wait $RPCPID 2>/dev/null
log "done6c"
