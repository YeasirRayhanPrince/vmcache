#!/usr/bin/env bash
set -euo pipefail

# ── Sweep configuration (edit these arrays) ──────────────────────────
# This script sweeps only the env vars that vmcache-leis.cpp understands
# (the original 2-tier baseline: DRAM -> SSD, no NUMA tier).

# Override any of these from the environment with a space-separated list,
# e.g.  DATASIZE_LIST=1000 RNDREAD_LIST=0 ./bench_sweep_leis.sh
SWEEP_PHYSGB=(${PHYSGB_LIST:-32})
SWEEP_THREADS=(${THREADS_LIST:-32})
SWEEP_DATASIZE=(${DATASIZE_LIST:-1000000000})
SWEEP_RUNFOR=(${RUNFOR_LIST:-900})
SWEEP_BATCH=(${BATCH_LIST:-64})
SWEEP_RNDREAD=(${RNDREAD_LIST:-1})

# ── Fixed defaults (inherited unless overridden by sweep) ─────────────
# SSD Blocks
export BLOCK=${BLOCK:-/dev/nvme0n1}
export EXMAP=${EXMAP:-0}

# Virtual Memory
export VIRTGB=${VIRTGB:-894}

# ── Disable NUMA balancing ────────────────────────────────────────────
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
