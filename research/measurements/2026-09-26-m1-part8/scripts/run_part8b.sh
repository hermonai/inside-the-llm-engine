#!/bin/bash
# Part VIII-IX runs after run_part8a.sh: repeated benchmark invocations, load tests, and
# KL divergence of runtime choices. Each step starts when the one-minute load is below 6.
cd "$(dirname "$0")"
until grep -q "all done" out/run.log 2>/dev/null; do sleep 10; done
gate() { local n=0; while [ "$(sysctl -n vm.loadavg | awk '{print ($2 < 6.0)}')" != "1" ] && [ $n -lt 30 ]; do sleep 10; n=$((n+1)); done
  echo "$(date -u +%FT%TZ) start $1 load=$(sysctl -n vm.loadavg) swap=$(sysctl -n vm.swapusage | awk '{print $6}')" >> out/run.log; }
L=$HOME/.ollama/models/blobs/sha256-dde5aa3fc5ffc17176b5e8bdc82f587b24b2678c6c66101bf7da77af9f7ccdff
# 1. the same benchmark twelve times (Ch 39)
gate repeat
mkdir -p out/repeat
for i in $(seq -w 1 12); do
  echo "{\"i\": $i, \"utc\": \"$(date -u +%FT%TZ)\", \"load\": \"$(sysctl -n vm.loadavg)\"}" > out/repeat/meta-$i.json
  llama-bench -m "$L" -fa 1 -p 0 -n 128 -r 3 -o json > out/repeat/run-$i.json 2> out/repeat/run-$i.stderr.txt
  sleep 15
done
echo "$(date -u +%FT%TZ) end repeat" >> out/run.log
# 2. closed vs open loop through a 5-second stall (Ch 39)
gate stall; python3 loadgen.py stall out/loadgen > out/loadgen-stall.log 2>&1; echo "$(date -u +%FT%TZ) end stall rc=$?" >> out/run.log
# 3. open-loop rate sweep with /metrics (Ch 40)
gate sweep; python3 loadgen.py sweep out/loadgen > out/loadgen-sweep.log 2>&1; echo "$(date -u +%FT%TZ) end sweep rc=$?" >> out/run.log
# 4. KL divergence of runtime choices against the default (Ch 38)
gate kld
mkdir -p out/kld
tr -s '[:space:]' ' ' < doc.txt > out/kld/text.txt
P="llama-perplexity -m $L -f out/kld/text.txt -c 512 --chunks 8 -ngl 99"
$P --kl-divergence-base out/kld/base.kld > out/kld/base.txt 2>&1
$P --kl-divergence-base out/kld/base.kld --kl-divergence > out/kld/repeat.txt 2>&1
$P --kl-divergence-base out/kld/base.kld --kl-divergence -ub 64 > out/kld/ubatch64.txt 2>&1
$P --kl-divergence-base out/kld/base.kld --kl-divergence -fa off > out/kld/fa-off.txt 2>&1
$P --kl-divergence-base out/kld/base.kld --kl-divergence -ctk q8_0 -ctv q8_0 > out/kld/kv-q8_0.txt 2>&1
$P --kl-divergence-base out/kld/base.kld --kl-divergence -ctk q4_0 -ctv q4_0 > out/kld/kv-q4_0.txt 2>&1
llama-quantize --allow-requantize "$L" out/kld/requant-q2_k.gguf Q2_K > out/kld/quantize-q2_k.txt 2>&1
llama-perplexity -m out/kld/requant-q2_k.gguf -f out/kld/text.txt -c 512 --chunks 8 -ngl 99 --kl-divergence-base out/kld/base.kld --kl-divergence > out/kld/weights-q2_k-requant.txt 2>&1
shasum -a 256 out/kld/requant-q2_k.gguf > out/kld/requant-q2_k.sha256
rm -f out/kld/requant-q2_k.gguf out/kld/base.kld
echo "$(date -u +%FT%TZ) end kld" >> out/run.log
echo "$(date -u +%FT%TZ) part8b done" >> out/run.log
