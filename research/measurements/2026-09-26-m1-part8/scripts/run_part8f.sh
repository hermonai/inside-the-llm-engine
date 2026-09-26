#!/bin/zsh
# Chapter 42 roofline, fourth version: + ceilings at d=1024, L1 sweep, spawn cost;
# both kernels per thread count, 1/2/4 threads. Writes out/capstone-roofline4.{json,txt}.
set -u
HERE=${0:A:h}
REPO=$HOME/ClaudeProjects/inside-the-llm-engine/code/mini-engine
OUT=$HERE/out
{
  echo "== start $(date -u +%Y-%m-%dT%H:%M:%SZ) load: $(sysctl -n vm.loadavg)"
  (cd $REPO && git rev-parse --short HEAD && git status --short crates/capstone | head)
  (cd $REPO && cargo build --release -p capstone --example capstone_roofline 2>&1 | tail -2)
  for rep in 1 2 3; do
    echo "== run $rep load: $(sysctl -n vm.loadavg)"
    $REPO/target/release/examples/capstone_roofline $OUT/capstone-roofline4-run$rep.json 1 2 4
  done
  echo "== end $(date -u +%Y-%m-%dT%H:%M:%SZ) load: $(sysctl -n vm.loadavg)"
} 2>&1 | tee $OUT/capstone-roofline4.txt
