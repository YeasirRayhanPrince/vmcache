#!/usr/bin/env bash
#
# ══════════════════════════════════════════════════════════════════════
#  bench_sweep_leis.sh — parameter sweep for vmcache-leis (2-tier)
# ══════════════════════════════════════════════════════════════════════
#
# WHAT IT RUNS
#   ./vmcache-leis — the original 2-tier buffer manager (DRAM -> SSD),
#   byte-identical to upstream viktorleis/vmcache. This is the baseline
#   plotted as "vmcache" against vmcache^n.
#   For the 3-tier version use ./bench_sweep_n.sh.
#
# WHAT IT DOES
#   Runs the Cartesian product of the SWEEP_* arrays below, one benchmark
#   run of RUNFOR seconds each. Far fewer knobs than the 3-tier sweeper:
#   vmcache-leis has no tiers, so no migration ratios or methods.
#
# WHAT IT WRITES  (under bench_results/, same layout as bench_sweep_n.sh)
#   <sweep_id>_summary.jsonl   one JSON line per run; the plotting index.
#   <tag>.json / <tag>.log     per-run config and output.
#   Tags here carry no _remote/_nm/_mp2 fields, e.g.
#     20260213_235605_phys32_t32_data1000000000_run900_batch64_rnd1
#   That absence is load-bearing: plot_paper.py labels a run with no
#   REMOTEGB key as "vmcache", which is how the baseline curve is named.
#
# HOW TO USE IT
#   Edit the defaults, or override from the environment:
#     DATASIZE_LIST=1000 RNDREAD_LIST=0 ./bench_sweep_leis.sh
#   The repro_fig*.sh presets drive this script that way.
#
# REQUIREMENTS
#   A raw block device and passwordless sudo. No NUMA topology or custom
#   kernel needed — this binary contains no NUMA code at all.
#
set -euo pipefail

# ── Swept parameters (Cartesian product; override via *_LIST env vars) ─

# DRAM buffer-pool size in GB.
SWEEP_PHYSGB=(${PHYSGB_LIST:-32})
# Worker threads.
SWEEP_THREADS=(${THREADS_LIST:-32})
# TPC-C warehouses when RNDREAD=0; record count when RNDREAD=1.
SWEEP_DATASIZE=(${DATASIZE_LIST:-1000000000})
# Measurement duration per run, in seconds (excludes load time).
SWEEP_RUNFOR=(${RUNFOR_LIST:-900})
# Pages per eviction batch. Note the name: vmcache-leis reads BATCH,
# whereas vmcache-n reads EVICT_BATCH for the equivalent knob.
SWEEP_BATCH=(${BATCH_LIST:-64})
# Workload: 0 = TPC-C, 1 = random read.
SWEEP_RNDREAD=(${RNDREAD_LIST:-1})

# ── Fixed for the whole sweep (override from the environment) ──────────

# Raw block device backing the buffer pool. NOTE: it is written to.
export BLOCK=${BLOCK:-/dev/nvme0n1}
# Use the exmap kernel module instead of plain mmap (needs the module).
export EXMAP=${EXMAP:-0}

# Virtual address space reserved, in GB. Must be >= the dataset size, and
# needs vm.overcommit_memory=1. Not physical memory.
export VIRTGB=${VIRTGB:-894}

# ── Disable NUMA balancing ────────────────────────────────────────────
# Not strictly needed for this binary (it never migrates pages), but kept
# so baseline and vmcache-n runs are measured under identical kernel
# settings.
sudo sh -c 'echo 0 > /proc/sys/kernel/numa_balancing'

# ── Run loop ──────────────────────────────────────────────────────────
mkdir -p bench_results

# Create timestamped summary file
sweep_start_time=$(date +%Y%m%d_%H%M%S)
summary_file="bench_results/${sweep_start_time}_summary.jsonl"
> "$summary_file"

for phys in "${SWEEP_PHYSGB[@]}"; do
for threads in "${SWEEP_THREADS[@]}"; do
for datasize in "${SWEEP_DATASIZE[@]}"; do
for runfor in "${SWEEP_RUNFOR[@]}"; do
for batch in "${SWEEP_BATCH[@]}"; do
for rndread in "${SWEEP_RNDREAD[@]}"; do

    # Record actual run timestamp
    run_timestamp=$(date +%Y%m%d_%H%M%S)

    # Use sweep start time as experiment ID for filenames (prefix)
    tag="${sweep_start_time}_phys${phys}_t${threads}_data${datasize}_run${runfor}_batch${batch}_rnd${rndread}"
    logfile="bench_results/${tag}.log"
    jsonfile="bench_results/${tag}.json"

    export PHYSGB="$phys"
    export THREADS="$threads"
    export DATASIZE="$datasize"
    export RUNFOR="$runfor"
    export BATCH="$batch"
    export RNDREAD="$rndread"

    # Create JSON summary for this run
    cat > "$jsonfile" <<EOF
{
  "timestamp": "$run_timestamp",
  "sweep_id": "$sweep_start_time",
  "tag": "$tag",
  "logfile": "$logfile",
  "config": {
    "PHYSGB": $phys,
    "THREADS": $threads,
    "DATASIZE": $datasize,
    "RUNFOR": $runfor,
    "BATCH": $batch,
    "RNDREAD": $rndread,
    "BLOCK": "$BLOCK",
    "EXMAP": $EXMAP,
    "VIRTGB": $VIRTGB
  }
}
EOF

    # Append to summary file (one line per run)
    echo "{\"timestamp\":\"$run_timestamp\",\"sweep_id\":\"$sweep_start_time\",\"tag\":\"$tag\",\"logfile\":\"$logfile\",\"PHYSGB\":$phys,\"THREADS\":$threads,\"DATASIZE\":$datasize,\"RUNFOR\":$runfor,\"BATCH\":$batch,\"RNDREAD\":$rndread,\"BLOCK\":\"$BLOCK\",\"EXMAP\":$EXMAP,\"VIRTGB\":$VIRTGB}" >> "$summary_file"

    echo "=== Running: $tag ==="
    sudo -E numactl --cpubind=0 ./vmcache-leis &> "$logfile" || true

done; done; done; done; done; done
