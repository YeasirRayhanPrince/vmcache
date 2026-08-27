#!/usr/bin/env bash
# Figure "vmcache vs vmcache^n", panel 2 (random read).
#
# Reproduces the 6 series: vmcache (2-tier baseline) and vmcache^n at
# REMOTEGB = 8/16/32/64/128, with conservative migration ratios (0.1).
#
# Parameters below were transcribed from the committed sweep records
#   bench_results/20260214_031931_summary.jsonl  (tiered arms)
#   bench_results/20260213_235605_summary.jsonl  (baseline arm)
#
# Runtime: 6 runs x 900 s  ~= 1.5 h, plus load time for 1e9 records per run.
#
# Prerequisites: 2 NUMA nodes, BLOCK device, passwordless sudo, and a kernel
# providing the move_pages2 syscall (462) for NUMA_MIGRATE_METHOD=3.
#
# Usage:  ./repro_fig2.sh
#         BLOCK=/dev/nvme1n1 ./repro_fig2.sh      # override the device

set -euo pipefail
cd "$(dirname "$0")"

export BLOCK=${BLOCK:-/dev/nvme0n1}
export VIRTGB=${VIRTGB:-894}
export EXMAP=${EXMAP:-0}
export DRAM_NODE=${DRAM_NODE:-0}
export REMOTE_NODE=${REMOTE_NODE:-1}
export EVICT_BATCH_SSD=${EVICT_BATCH_SSD:-64}
export PROMOTE_BATCH_SCAN_MULTIPLIER=${PROMOTE_BATCH_SCAN_MULTIPLIER:-2}

make vmcache-n vmcache-leis

echo "=== [1/2] vmcache^n arms (REMOTEGB = 8 16 32 64 128) ==="
PHYSGB_LIST="32" \
REMOTEGB_LIST="8 16 32 64 128" \
RATIO_LIST="0.1" \
THREADS_LIST="32" \
DATASIZE_LIST="1000000000" \
RUNFOR_LIST="900" \
PROMOTE_BATCH_LIST="1" \
EVICT_BATCH_LIST="512" \
RNDREAD_LIST="1" \
NUMA_MIGRATE_METHOD_LIST="3" \
MOVE_PAGES2_MODE_LIST="0" \
  ./bench_sweep_n.sh

echo "=== [2/2] vmcache baseline arm ==="
PHYSGB_LIST="32" \
THREADS_LIST="32" \
DATASIZE_LIST="1000000000" \
RUNFOR_LIST="900" \
BATCH_LIST="64" \
RNDREAD_LIST="1" \
  ./bench_sweep_leis.sh

echo
echo "Done. Note the two sweep IDs printed above, then plot with:"
echo "  python3 plots/plot_paper.py --sweep <TIERED_ID> <BASELINE_ID> \\"
echo "      --remo-label --logy --no-legend"
