#!/bin/bash
cd "$(dirname "$0")"
python3 offload3.py out/offload2 > out/offload2/attempt2.log 2>&1
python3 repack.py out/repack > out/repack.log 2>&1
echo done >> out/repack.log
