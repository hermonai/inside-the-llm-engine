#!/bin/bash
# Part VI probes, one at a time; each waits (up to 10 min) for a one-minute load
# average below 6, then runs and records the conditions.
D=<scratch>
M=$D/meas6
OUT=$M/out
BR=$D/llama-build-rpc
BL=$HOME/.ollama/models/blobs
L3=$BL/sha256-dde5aa3fc5ffc17176b5e8bdc82f587b24b2678c6c66101bf7da77af9f7ccdff
Q8=$BL/sha256-e6a7edc1a4d7d9b2de136a221a57336b76316cfe53a252aeba814496c5ae439d
mkdir -p $OUT
cd $M
log(){ echo "$(date -u +%FT%TZ) $*" >> $OUT/progress.log; }
gate(){ for i in $(seq 1 20); do l=$(sysctl -n vm.loadavg | awk '{print $2}'); awk -v l="$l" 'BEGIN{exit !(l < 6)}' && return; log "waiting: load1=$l before $1"; sleep 30; done; log "gate timed out before $1; running anyway"; }
cond(){ echo "--- $1: $(date -u +%FT%TZ) load $(sysctl -n vm.loadavg) $(sysctl -n vm.swapusage)" >> $OUT/conditions.log; }

gate ssd; cond ssd
python3 part6_probe.py ssd $OUT > $OUT/ssd.stdout 2> $OUT/ssd.stderr; log "ssd exit=$?"

log "building rpc-server and llama-bench"
cmake -S $D/llama-src -B $BR -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DGGML_RPC=ON -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF > $OUT/build-rpc.log 2>&1
cmake --build $BR --target rpc-server llama-bench -j 4 >> $OUT/build-rpc.log 2>&1; log "build exit=$?"

$BR/bin/rpc-server -p 50052 -d MTL0 > $OUT/rpc-server.log 2>&1 &
RPCPID=$!
sleep 5
$BR/bin/llama-bench --rpc localhost:50052 --list-devices > $OUT/rpc-devices.txt 2>&1; log "list-devices exit=$?"
for m in L3 Q8; do
  eval MP=\$$m
  for cfg in local split remote; do
    case $cfg in
      local)  ARGS="" ;;
      split)  ARGS="--rpc localhost:50052 -ts 1/1" ;;
      remote) ARGS="--rpc localhost:50052 -ts 1/0" ;;
    esac
    gate rpc-$m-$cfg; cond rpc-$m-$cfg
    $BR/bin/llama-bench -m $MP -fa 1 -p 512 -n 64 -r 3 $ARGS -o json > $OUT/rpc-$m-$cfg.json 2> $OUT/rpc-$m-$cfg.err
    log "rpc $m $cfg exit=$?"
  done
done
kill $RPCPID; wait $RPCPID 2>/dev/null

for e in kvmove disagg; do
  gate $e; cond $e
  python3 part6_probe.py $e $OUT > $OUT/$e.stdout 2> $OUT/$e.stderr; log "$e exit=$?"
done
cond end; log done
