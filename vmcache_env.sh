#!/usr/bin/env bash
set -euo pipefail
sudo sh -c 'echo 0 > /proc/sys/kernel/numa_balancing'

# Core configuration
export BLOCK=${BLOCK:-/dev/nvme0n1}
export VIRTGB=${VIRTGB:-894}
export PHYSGB=${PHYSGB:-32}
export REMOTEGB=${REMOTEGB:-32}
export EXMAP=${EXMAP:-0}

# Multi-tier migration policy (Hyrise-style)
export DRAM_READ_RATIO=${DRAM_READ_RATIO:-0.5}
export DRAM_WRITE_RATIO=${DRAM_WRITE_RATIO:-0.5}
export NUMA_READ_RATIO=${NUMA_READ_RATIO:-1}
export NUMA_WRITE_RATIO=${NUMA_WRITE_RATIO:-1}

# Inline batched promotion / eviction
export PROMOTE_BATCH=${PROMOTE_BATCH:-1}
export PROMOTE_BATCH_SCAN_MULTIPLIER=${PROMOTE_BATCH_SCAN_MULTIPLIER:-2}
export EVICT_BATCH=${EVICT_BATCH:-2048}
export EVICT_BATCH_SSD=${EVICT_BATCH_SSD:-64}

# NUMA migration methods
export NUMA_MIGRATE_METHOD=${NUMA_MIGRATE_METHOD:-2}
export MOVE_PAGES2_MAX_BATCH_SIZE=${MOVE_PAGES2_MAX_BATCH_SIZE:-${EVICT_BATCH}}
export MOVE_PAGES2_MODE=${MOVE_PAGES2_MODE:-0}

# NUMA node configuration
export DRAM_NODE=${DRAM_NODE:-0}
export REMOTE_NODE=${REMOTE_NODE:-1}

# Benchmark and eviction configuration
export RUNFOR=${RUNFOR:-500}
export RNDREAD=${RNDREAD:-0}
export THREADS=${THREADS:-32}
export DATASIZE=${DATASIZE:-1000}

# YCSB configuration (set YCSB to A-F to enable, unset for TPC-C/rndread)
export YCSB=C
export ZIPF_THETA=${ZIPF_THETA:-0.90}
# Key = 8 B and total size <= 993
export YCSB_TUPLE_SIZE=${YCSB_TUPLE_SIZE:-56}
# Scan selectivity: fraction of dataset per scan (e.g., 0.0001=0.01%, 0.001=0.1%, 0.01=1%, 0.1=10%)
export YCSB_SCAN_SELECTIVITY=${YCSB_SCAN_SELECTIVITY:-1e-7}

# Execute
sudo -E numactl --cpubind=0 ./vmcache

# sudo swapoff -a && sudo swapon -a

