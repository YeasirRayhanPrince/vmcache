#!/usr/bin/env bash
#
# ══════════════════════════════════════════════════════════════════════
#  bench_sweep_n_ycsb.sh — YCSB parameter sweep for vmcache-n (3-tier)
# ══════════════════════════════════════════════════════════════════════
#
# WHAT IT RUNS
#   ./vmcache-n under the YCSB workload. Same binary as bench_sweep_n.sh,
#   different workload: that script runs TPC-C / random read, this one
#   runs YCSB A-F with a zipfian key distribution.
#   The 2-tier baseline cannot run this at all — vmcache-leis has no YCSB.
#
# WHAT IT WRITES
#   bench_results_ycsb/  — note the SEPARATE directory, not bench_results/.
#   Plot these with `plot_paper.py --ycsb`, which switches the input dir.
#   Tags additionally carry _ycsb/_zipf/_ts/_sel fields.
#
# STATUS
#   No results from this script are currently committed (there is no
#   bench_results_ycsb/ in the repo), and no paper figure depends on it.
#
# HOW TO USE IT
#   Edit the arrays below. Unlike bench_sweep_n.sh, these are not yet
#   environment-overridable.
#
# REQUIREMENTS
#   Same as bench_sweep_n.sh: two NUMA nodes, a block device, passwordless
#   sudo, and a kernel with move_pages2 (462) for NUMA_MIGRATE_METHOD=3.
#
set -euo pipefail

# ── Swept parameters (Cartesian product) ──────────────────────────────

# DRAM buffer-pool size in GB — the local/fast tier.
SWEEP_PHYSGB=(32)
# Remote NUMA tier size in GB. 0 disables the tier.
SWEEP_REMOTEGB=(64)

# Page-migration probabilities, 0.0-1.0. Unlike bench_sweep_n.sh (which
# ties all four together via SWEEP_RATIO), these are swept individually.
# The fourth, NUMA_WRITE_RATIO, is fixed below.
SWEEP_DRAM_READ_RATIO=(1)    # promote REMOTE->DRAM on read
SWEEP_DRAM_WRITE_RATIO=(1)   # promote REMOTE->DRAM on write
SWEEP_NUMA_READ_RATIO=(1)    # demote DRAM->REMOTE on read eviction


# Worker threads.
SWEEP_THREADS=(32)
# Record count for the YCSB table.
SWEEP_DATASIZE=(1000)
# Measurement duration per run, in seconds (excludes load time).
SWEEP_RUNFOR=(900)

# Minimum pages per REMOTE->DRAM promotion batch (1 = no batching).
SWEEP_PROMOTE_BATCH=(1)
# Pages per DRAM->REMOTE demotion batch; also caps move_pages2 batches.
SWEEP_EVICT_BATCH=(1 128 256 512 1024 2048)

# YCSB workload mix: A=50/50 r/w, B=95/5, C=read-only, D=read-latest,
# E=short scans, F=read-modify-write.
SWEEP_YCSB=(A)
# Zipfian skew. 0 = uniform; higher = more skewed (0.99 is YCSB default).
SWEEP_ZIPF_THETA=(0.90)
# Tuple size in bytes. Key is 8 B and the total must stay <= 993.
SWEEP_YCSB_TUPLE_SIZE=(112)
# Fraction of the table touched per scan, for workload E.
SWEEP_YCSB_SCAN_SELECTIVITY=(1e-7)

# How pages are physically moved between NUMA nodes:
#   0 = mbind() single, 1 = move_pages() single,
#   2 = move_pages() batched, 3 = move_pages2() custom syscall.
SWEEP_NUMA_MIGRATE_METHOD=(0 1 2 3)
# Flags passed to move_pages2; only meaningful for method 3.
SWEEP_MOVE_PAGES2_MODE=(0 1 2)

# ── Fixed for the whole sweep (override from the environment) ──────────

# Raw block device backing the buffer pool. NOTE: it is written to.
export BLOCK=${BLOCK:-/dev/nvme0n1}
# Use the exmap kernel module instead of plain mmap (needs the module).
export EXMAP=${EXMAP:-0}

# Virtual address space reserved, in GB. Needs vm.overcommit_memory=1.
export VIRTGB=${VIRTGB:-894}

# Which NUMA node is the DRAM tier and which is the remote tier.
export DRAM_NODE=${DRAM_NODE:-0}
export REMOTE_NODE=${REMOTE_NODE:-1}

# The fourth migration ratio: demote DRAM->REMOTE on write eviction.
# Fixed here rather than swept.
export NUMA_WRITE_RATIO=${NUMA_WRITE_RATIO:-1}

# Pages per write-back batch when evicting all the way out to SSD.
export EVICT_BATCH_SSD=${EVICT_BATCH_SSD:-64}

# Scan window for gathering a promotion batch, as a multiple of
# PROMOTE_BATCH.
export PROMOTE_BATCH_SCAN_MULTIPLIER=${PROMOTE_BATCH_SCAN_MULTIPLIER:-2}

# ── Disable NUMA balancing ────────────────────────────────────────────
# The kernel's automatic NUMA balancing would migrate pages behind the
# buffer manager's back and corrupt the measurements.
sudo sh -c 'echo 0 > /proc/sys/kernel/numa_balancing'

# ── Run loop ──────────────────────────────────────────────────────────
mkdir -p bench_results_ycsb

# Create timestamped summary file
sweep_start_time=$(date +%Y%m%d_%H%M%S)
summary_file="bench_results_ycsb/${sweep_start_time}_summary.jsonl"
> "$summary_file"

for phys in "${SWEEP_PHYSGB[@]}"; do
for remote in "${SWEEP_REMOTEGB[@]}"; do
for dram_r in "${SWEEP_DRAM_READ_RATIO[@]}"; do
for dram_w in "${SWEEP_DRAM_WRITE_RATIO[@]}"; do
for numa_r in "${SWEEP_NUMA_READ_RATIO[@]}"; do
for threads in "${SWEEP_THREADS[@]}"; do
for datasize in "${SWEEP_DATASIZE[@]}"; do
for runfor in "${SWEEP_RUNFOR[@]}"; do
for pbatch in "${SWEEP_PROMOTE_BATCH[@]}"; do
for ebatch in "${SWEEP_EVICT_BATCH[@]}"; do
for ycsb in "${SWEEP_YCSB[@]}"; do
for zipf in "${SWEEP_ZIPF_THETA[@]}"; do
for tuple_size in "${SWEEP_YCSB_TUPLE_SIZE[@]}"; do
for selectivity in "${SWEEP_YCSB_SCAN_SELECTIVITY[@]}"; do
for numa_method in "${SWEEP_NUMA_MIGRATE_METHOD[@]}"; do

  # Only sweep MOVE_PAGES2_MODE when NUMA_MIGRATE_METHOD=3
  if [ "$numa_method" -eq 3 ]; then
    mp2_modes=("${SWEEP_MOVE_PAGES2_MODE[@]}")
  else
    mp2_modes=(0)
  fi

  for mp2_mode in "${mp2_modes[@]}"; do

    # Skip batch size 1 when NUMA_MIGRATE_METHOD >= 2
    if [ "$numa_method" -ge 2 ] && [ "$ebatch" -eq 1 ]; then
      continue
    fi

    # Record actual run timestamp
    run_timestamp=$(date +%Y%m%d_%H%M%S)

    # Use sweep start time as experiment ID for filenames (prefix)
    tag="${sweep_start_time}_phys${phys}_remote${remote}_dr${dram_r}_dw${dram_w}_nr${numa_r}_t${threads}_data${datasize}_run${runfor}_pb${pbatch}_eb${ebatch}_ycsb${ycsb}_zipf${zipf}_ts${tuple_size}_sel${selectivity}_nm${numa_method}_mp2${mp2_mode}"
    logfile="bench_results_ycsb/${tag}.log"
    jsonfile="bench_results_ycsb/${tag}.json"

    export PHYSGB="$phys"
    export REMOTEGB="$remote"
    export DRAM_READ_RATIO="$dram_r"
    export DRAM_WRITE_RATIO="$dram_w"
    export NUMA_READ_RATIO="$numa_r"
    export THREADS="$threads"
    export DATASIZE="$datasize"
    export RUNFOR="$runfor"
    export PROMOTE_BATCH="$pbatch"
    export EVICT_BATCH="$ebatch"
    export YCSB="$ycsb"
    export ZIPF_THETA="$zipf"
    export YCSB_TUPLE_SIZE="$tuple_size"
    export YCSB_SCAN_SELECTIVITY="$selectivity"
    export NUMA_MIGRATE_METHOD="$numa_method"
    export MOVE_PAGES2_MODE="$mp2_mode"
    export MOVE_PAGES2_MAX_BATCH_SIZE="$ebatch"

    # Create JSON summary for this run
    cat > "$jsonfile" <<EOF
{
  "timestamp": "$run_timestamp",
  "sweep_id": "$sweep_start_time",
  "tag": "$tag",
  "logfile": "$logfile",
  "config": {
    "PHYSGB": $phys,
    "REMOTEGB": $remote,
    "DRAM_READ_RATIO": $dram_r,
    "DRAM_WRITE_RATIO": $dram_w,
    "NUMA_READ_RATIO": $numa_r,
    "THREADS": $threads,
    "DATASIZE": $datasize,
    "RUNFOR": $runfor,
    "PROMOTE_BATCH": $pbatch,
    "EVICT_BATCH": $ebatch,
    "YCSB": "$ycsb",
    "ZIPF_THETA": $zipf,
    "YCSB_TUPLE_SIZE": $tuple_size,
    "YCSB_SCAN_SELECTIVITY": "$selectivity",
    "NUMA_MIGRATE_METHOD": $numa_method,
    "MOVE_PAGES2_MODE": $mp2_mode,
    "MOVE_PAGES2_MAX_BATCH_SIZE": $ebatch,
    "BLOCK": "$BLOCK",
    "EXMAP": $EXMAP,
    "VIRTGB": $VIRTGB,
    "DRAM_NODE": $DRAM_NODE,
    "REMOTE_NODE": $REMOTE_NODE,
    "NUMA_WRITE_RATIO": $NUMA_WRITE_RATIO,
    "EVICT_BATCH_SSD": $EVICT_BATCH_SSD,
    "PROMOTE_BATCH_SCAN_MULTIPLIER": $PROMOTE_BATCH_SCAN_MULTIPLIER
  }
}
EOF

    # Append to summary file (one line per run)
    echo "{\"timestamp\":\"$run_timestamp\",\"sweep_id\":\"$sweep_start_time\",\"tag\":\"$tag\",\"logfile\":\"$logfile\",\"PHYSGB\":$phys,\"REMOTEGB\":$remote,\"DRAM_READ_RATIO\":$dram_r,\"DRAM_WRITE_RATIO\":$dram_w,\"NUMA_READ_RATIO\":$numa_r,\"THREADS\":$threads,\"DATASIZE\":$datasize,\"RUNFOR\":$runfor,\"PROMOTE_BATCH\":$pbatch,\"EVICT_BATCH\":$ebatch,\"YCSB\":\"$ycsb\",\"ZIPF_THETA\":$zipf,\"YCSB_TUPLE_SIZE\":$tuple_size,\"YCSB_SCAN_SELECTIVITY\":\"$selectivity\",\"NUMA_MIGRATE_METHOD\":$numa_method,\"MOVE_PAGES2_MODE\":$mp2_mode,\"MOVE_PAGES2_MAX_BATCH_SIZE\":$ebatch,\"BLOCK\":\"$BLOCK\",\"EXMAP\":$EXMAP,\"VIRTGB\":$VIRTGB,\"DRAM_NODE\":$DRAM_NODE,\"REMOTE_NODE\":$REMOTE_NODE,\"NUMA_WRITE_RATIO\":$NUMA_WRITE_RATIO,\"EVICT_BATCH_SSD\":$EVICT_BATCH_SSD,\"PROMOTE_BATCH_SCAN_MULTIPLIER\":$PROMOTE_BATCH_SCAN_MULTIPLIER}" >> "$summary_file"

    echo "=== Running: $tag ==="
    sudo -E numactl --cpubind=0 ./vmcache-n &> "$logfile" || true

  done

done; done; done; done; done; done; done; done; done; done; done; done; done; done; done
