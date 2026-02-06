#!/usr/bin/env bash
set -euo pipefail

# Core configuration
export BLOCK=${BLOCK:-/dev/nvme0n1}
export VIRTGB=${VIRTGB:-894}
export PHYSGB=${PHYSGB:-4}
export REMOTEGB=${REMOTEGB:-4}
export EXMAP=${EXMAP:-0}

# Multi-tier migration policy (Hyrise-style)
export DRAM_READ_RATIO=${DRAM_READ_RATIO:-0.5}
export DRAM_WRITE_RATIO=${DRAM_WRITE_RATIO:-0.5}
export NUMA_READ_RATIO=${NUMA_READ_RATIO:-0.5}
export NUMA_WRITE_RATIO=${NUMA_WRITE_RATIO:-0.5}

# Inline batched promotion / eviction
export PROMOTE_BATCH=${PROMOTE_BATCH:-32}
export PROMOTE_BATCH_SCAN_MULTIPLIER=${PROMOTE_BATCH_SCAN_MULTIPLIER:-8}
export EVICT_BATCH=${EVICT_BATCH:-${PROMOTE_BATCH}}
export EVICT_BATCH_SSD=${EVICT_BATCH_SSD:-64}

# NUMA migration methods
export NUMA_MIGRATE_METHOD=${NUMA_MIGRATE_METHOD:-2}
export MOVE_PAGES2_MAX_BATCH_SIZE=${MOVE_PAGES2_MAX_BATCH_SIZE:-${PROMOTE_BATCH}}
export MOVE_PAGES2_MODE=${MOVE_PAGES2_MODE:-0}

# NUMA node configuration
export DRAM_NODE=${DRAM_NODE:-0}
export REMOTE_NODE=${REMOTE_NODE:-1}

# Benchmark and eviction configuration
export RUNFOR=${RUNFOR:-60}
export RNDREAD=${RNDREAD:-0}
export THREADS=${THREADS:-16}
export DATASIZE=${DATASIZE:-64}

# Execute
sudo -E ./vmcache
