#!/usr/bin/env bash
set -euo pipefail

# ── Sweep configuration (edit these arrays) ──────────────────────────

SWEEP_PHYSGB=(32)
SWEEP_REMOTEGB=(64)

SWEEP_DRAM_READ_RATIO=(1)
SWEEP_DRAM_WRITE_RATIO=(1)
SWEEP_NUMA_READ_RATIO=(1)


SWEEP_THREADS=(32)
SWEEP_DATASIZE=(1000)
SWEEP_RUNFOR=(900)

SWEEP_PROMOTE_BATCH=(1)
SWEEP_EVICT_BATCH=(1 64 128 256 512 1024 2048 4096)

SWEEP_RNDREAD=(1)

SWEEP_NUMA_MIGRATE_METHOD=(0 1 2 3)
SWEEP_MOVE_PAGES2_MODE=(0 1 2)

# ── Fixed defaults (inherited unless overridden by sweep) ─────────────
# SSD Blocks 
export BLOCK=${BLOCK:-/dev/nvme0n1}
export EXMAP=${EXMAP:-0}

# Virtual Memory
export VIRTGB=${VIRTGB:-894}

# Memory Tiers
export DRAM_NODE=${DRAM_NODE:-0}
export REMOTE_NODE=${REMOTE_NODE:-1}

# Multi-tier migration policy (Hyrise-style)
export NUMA_WRITE_RATIO=${NUMA_WRITE_RATIO:-1}

# Memory -> SSD 
export EVICT_BATCH_SSD=${EVICT_BATCH_SSD:-64}

# Promotion details
export PROMOTE_BATCH_SCAN_MULTIPLIER=${PROMOTE_BATCH_SCAN_MULTIPLIER:-2}

# ── Disable NUMA balancing ────────────────────────────────────────────
sudo sh -c 'echo 0 > /proc/sys/kernel/numa_balancing'

# ── Run loop ──────────────────────────────────────────────────────────
mkdir -p bench_results

# Create timestamped summary file
sweep_start_time=$(date +%Y%m%d_%H%M%S)
summary_file="bench_results/${sweep_start_time}_summary.jsonl"
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
for rndread in "${SWEEP_RNDREAD[@]}"; do
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
    tag="${sweep_start_time}_phys${phys}_remote${remote}_dr${dram_r}_dw${dram_w}_nr${numa_r}_t${threads}_data${datasize}_run${runfor}_pb${pbatch}_eb${ebatch}_rnd${rndread}_nm${numa_method}_mp2${mp2_mode}"
    logfile="bench_results/${tag}.log"
    jsonfile="bench_results/${tag}.json"

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
    export RNDREAD="$rndread"
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
    "RNDREAD": $rndread,
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
    echo "{\"timestamp\":\"$run_timestamp\",\"sweep_id\":\"$sweep_start_time\",\"tag\":\"$tag\",\"logfile\":\"$logfile\",\"PHYSGB\":$phys,\"REMOTEGB\":$remote,\"DRAM_READ_RATIO\":$dram_r,\"DRAM_WRITE_RATIO\":$dram_w,\"NUMA_READ_RATIO\":$numa_r,\"THREADS\":$threads,\"DATASIZE\":$datasize,\"RUNFOR\":$runfor,\"PROMOTE_BATCH\":$pbatch,\"EVICT_BATCH\":$ebatch,\"RNDREAD\":$rndread,\"NUMA_MIGRATE_METHOD\":$numa_method,\"MOVE_PAGES2_MODE\":$mp2_mode,\"MOVE_PAGES2_MAX_BATCH_SIZE\":$ebatch,\"BLOCK\":\"$BLOCK\",\"EXMAP\":$EXMAP,\"VIRTGB\":$VIRTGB,\"DRAM_NODE\":$DRAM_NODE,\"REMOTE_NODE\":$REMOTE_NODE,\"NUMA_WRITE_RATIO\":$NUMA_WRITE_RATIO,\"EVICT_BATCH_SSD\":$EVICT_BATCH_SSD,\"PROMOTE_BATCH_SCAN_MULTIPLIER\":$PROMOTE_BATCH_SCAN_MULTIPLIER}" >> "$summary_file"

    echo "=== Running: $tag ==="
    sudo -E numactl --cpubind=0 ./vmcache &> "$logfile" || true

  done

done; done; done; done; done; done; done; done; done; done; done; done
