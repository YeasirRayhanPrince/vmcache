#!/usr/bin/env bash
set -euo pipefail
sudo sh -c 'echo 0 > /proc/sys/kernel/numa_balancing'

# Core configuration
export BLOCK=${BLOCK:-/dev/nvme0n1}
export VIRTGB=${VIRTGB:-894}
export PHYSGB=${PHYSGB:-8}
export REMOTEGB=${REMOTEGB:-16}
export EXMAP=${EXMAP:-0}

# Multi-tier migration policy (Hyrise-style)
export DRAM_READ_RATIO=${DRAM_READ_RATIO:-0.2}
export DRAM_WRITE_RATIO=${DRAM_WRITE_RATIO:-0.2}
export NUMA_READ_RATIO=${NUMA_READ_RATIO:-0.2}
export NUMA_WRITE_RATIO=${NUMA_WRITE_RATIO:-0.2}

# Inline batched promotion / eviction
export PROMOTE_BATCH=${PROMOTE_BATCH:-64}
export PROMOTE_BATCH_SCAN_MULTIPLIER=${PROMOTE_BATCH_SCAN_MULTIPLIER:-2}
export EVICT_BATCH=${EVICT_BATCH:-256}
export EVICT_BATCH_SSD=${EVICT_BATCH_SSD:-64}

# NUMA migration methods
export NUMA_MIGRATE_METHOD=${NUMA_MIGRATE_METHOD:-2}
export MOVE_PAGES2_MAX_BATCH_SIZE=${MOVE_PAGES2_MAX_BATCH_SIZE:-${PROMOTE_BATCH}}
export MOVE_PAGES2_MODE=${MOVE_PAGES2_MODE:-0}

# NUMA node configuration
export DRAM_NODE=${DRAM_NODE:-0}
export REMOTE_NODE=${REMOTE_NODE:-1}

# Benchmark and eviction configuration
export RUNFOR=${RUNFOR:-200}
export RNDREAD=${RNDREAD:-0}
export THREADS=${THREADS:-16}
export DATASIZE=${DATASIZE:-100}

# YCSB configuration (set YCSB to A-F to enable, unset for TPC-C/rndread)
# export YCSB=E
export ZIPF_THETA=${ZIPF_THETA:-0.99}
export YCSB_TUPLE_SIZE=${YCSB_TUPLE_SIZE:-100}

# Execute
sudo -E numactl --cpubind=0 ./vmcache

