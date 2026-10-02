#!/usr/bin/env bash
# Launcher: one OpenMP thread per core; on Linux also tcmalloc and core pinning. Works on macOS (MPS) and Linux.
HERE="$(cd "$(dirname "$0")" && pwd)"
NCPU=$(nproc 2>/dev/null || sysctl -n hw.ncpu)
export OMP_NUM_THREADS=$NCPU MKL_NUM_THREADS=$NCPU
if [ "$(uname)" = "Linux" ]; then
  TC=$(ls /usr/lib/x86_64-linux-gnu/libtcmalloc.so.4 /usr/lib/aarch64-linux-gnu/libtcmalloc.so.4 2>/dev/null | head -1)
  [ -n "$TC" ] && export LD_PRELOAD="${TC}${LD_PRELOAD:+:$LD_PRELOAD}"
  export OMP_PROC_BIND=close OMP_PLACES=cores
fi
export PYTORCH_ENABLE_MPS_FALLBACK=1  # run any op MPS lacks on the CPU instead of failing
export TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_TELEMETRY=1 TRANSFORMERS_VERBOSITY=error
exec python "$HERE/run_setfit.py" "$@"
