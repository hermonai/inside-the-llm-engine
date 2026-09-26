#!/bin/zsh
# Chapter 35: expert placement on the M1 with Granite 3.1 3B-A800M (granite3.1-moe:3b).
# All layers on the GPU; experts of the first 16 of 32 layers on the CPU; all experts on
# the CPU. --no-repack everywhere, so the CPU reads the same mapped pages the GPU uses and
# no second copy is made (repacking caused heavy swapping in Part VII). Interleaved,
# two rounds. Swap-outs are recorded around every run; the script stops if one run
# swaps out more than 1 GiB.
set -u
HERE=${0:A:h}; OUT=$HERE/out/cpumoe; mkdir -p $OUT
G=$HOME/.ollama/models/blobs/sha256-4cbc52994d8ce56d58f3ecadcd451a5dbb2a4f1142098c6b9f030d18ee5e052b
swapouts() { vm_stat | awk '/Swapouts/ {gsub("\\.",""); print $2}'; }
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $OUT/run.log; }
log "start load=$(sysctl -n vm.loadavg) swap=$(sysctl -n vm.swapusage | awk '{print $6}')"
for round in 1 2; do
  for cfg in gpu ncmoe16 cpumoe; do
    case $cfg in
      gpu) extra=() ;;
      ncmoe16) extra=(--n-cpu-moe 16) ;;
      cpumoe) extra=(--cpu-moe) ;;
    esac
    s0=$(swapouts)
    log "run $cfg round $round load=$(sysctl -n vm.loadavg)"
    llama-batched-bench -m $G -c 2048 -b 512 -ub 512 -npp 128 -ntg 64 -npl 1,4 -ngl 99 --no-repack $extra > $OUT/$cfg-r$round.txt 2>&1
    rc=$?
    s1=$(swapouts); d=$(( (s1 - s0) * 16384 / 1048576 ))
    log "end $cfg round $round rc=$rc swapout_MiB=$d"
    grep -E "^\|" $OUT/$cfg-r$round.txt | tail -3
    grep -E "CPU_Mapped|MTL0_Mapped|CPU_REPACK|model buffer size" $OUT/$cfg-r$round.txt | head -4
    if [ $d -gt 1024 ]; then log "stopping: swap-out above 1 GiB"; exit 1; fi
    sleep 5
  done
done
log "done"
