#!/usr/bin/env bash
set -euo pipefail

# ── Sweep configuration (edit these arrays) ──────────────────────────
SWEEP_YCSB=(A B C E F)
SWEEP_YCSB_TUPLE_SIZE=(56 100 1024 4096)
SWEEP_ZIPF_THETA=(0.99)
SWEEP_PHYSGB=(4)

SWEEP_REMOTEGB=(8)
SWEEP_THREADS=(16)
SWEEP_DATASIZE=(50)
SWEEP_RUNFOR=(100)
SWEEP_PROMOTE_BATCH=(1024)
SWEEP_EVICT_BATCH_SSD=(64)

# ── Fixed defaults (inherited unless overridden by sweep) ─────────────
export BLOCK=${BLOCK:-/dev/nvme0n1}
export VIRTGB=${VIRTGB:-894}
export EXMAP=${EXMAP:-0}

export RNDREAD=${RNDREAD:-0}
export DRAM_READ_RATIO=${DRAM_READ_RATIO:-0.2}
export DRAM_WRITE_RATIO=${DRAM_WRITE_RATIO:-0.2}
export NUMA_READ_RATIO=${NUMA_READ_RATIO:-0.2}
export NUMA_WRITE_RATIO=${NUMA_WRITE_RATIO:-0.2}

export PROMOTE_BATCH_SCAN_MULTIPLIER=${PROMOTE_BATCH_SCAN_MULTIPLIER:-2}

export NUMA_MIGRATE_METHOD=${NUMA_MIGRATE_METHOD:-3}
export MOVE_PAGES2_MODE=${MOVE_PAGES2_MODE:-0}
export DRAM_NODE=${DRAM_NODE:-0}
export REMOTE_NODE=${REMOTE_NODE:-1}


# ── Disable NUMA balancing ────────────────────────────────────────────
sudo sh -c 'echo 0 > /proc/sys/kernel/numa_balancing'

# ── Run loop ──────────────────────────────────────────────────────────
mkdir -p bench_results

for ycsb in "${SWEEP_YCSB[@]}"; do
for tuple in "${SWEEP_YCSB_TUPLE_SIZE[@]}"; do
for phys in "${SWEEP_PHYSGB[@]}"; do
for remote in "${SWEEP_REMOTEGB[@]}"; do
for threads in "${SWEEP_THREADS[@]}"; do
for theta in "${SWEEP_ZIPF_THETA[@]}"; do
for datasize in "${SWEEP_DATASIZE[@]}"; do
for runfor in "${SWEEP_RUNFOR[@]}"; do
for pbatch in "${SWEEP_PROMOTE_BATCH[@]}"; do
for ebatch_ssd in "${SWEEP_EVICT_BATCH_SSD[@]}"; do

  tag="ycsb${ycsb}_tuple${tuple}_phys${phys}_remote${remote}_threads${threads}_theta${theta}_data${datasize}_run${runfor}_pbatch${pbatch}_essd${ebatch_ssd}"
  logfile="bench_results/${tag}.log"

  export YCSB="$ycsb"
  export YCSB_TUPLE_SIZE="$tuple"
  export PHYSGB="$phys"
  export REMOTEGB="$remote"
  export THREADS="$threads"
  export ZIPF_THETA="$theta"
  export DATASIZE="$datasize"
  export RUNFOR="$runfor"
  export PROMOTE_BATCH="$pbatch"
  export EVICT_BATCH="$pbatch"
  export MOVE_PAGES2_MAX_BATCH_SIZE="$pbatch"
  export EVICT_BATCH_SSD="$ebatch_ssd"

  echo "=== Running: $tag ==="
  sudo -E numactl --cpubind=0 ./vmcache &> "$logfile" || true

done; done; done; done; done; done; done; done; done; done
