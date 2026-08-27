#!/usr/bin/env bash
#
# ══════════════════════════════════════════════════════════════════════
#  bench_sweep_n.sh — parameter sweep for vmcache-n (3-tier)
# ══════════════════════════════════════════════════════════════════════
#
# WHAT IT RUNS
#   ./vmcache-n — the 3-tier buffer manager (DRAM -> remote NUMA -> SSD).
#   The 2-tier baseline has its own sweeper: ./bench_sweep_leis.sh.
#   For the YCSB workload use ./bench_sweep_n_ycsb.sh.
#
# WHAT IT DOES
#   Runs the full Cartesian product of the SWEEP_* arrays below. Each
#   combination is one benchmark run of RUNFOR seconds. Beware the
#   multiplication: 2 tier sizes x 3 evict batches x 4 methods = 24 runs.
#
# WHAT IT WRITES  (all under bench_results/, one sweep = one timestamped ID)
#   <sweep_id>_summary.jsonl   one JSON line per run: config + log path.
#                              This is the index the plotting scripts read.
#   <tag>.json                 same config, one file per run.
#   <tag>.log                  stdout/stderr of the run (the measurements).
#   where <tag> encodes every swept parameter, e.g.
#     20260208_220148_phys32_remote64_dr1_dw1_nr1_nw1_t32_data1000
#       _run900_pb1_eb256_rnd0_nm3_mp20
#
# HOW TO USE IT
#   Either edit the defaults below, or override any array from the
#   environment with a space-separated list:
#     REMOTEGB_LIST="8 16 32 64 128" EVICT_BATCH_LIST=256 ./bench_sweep_n.sh
#   The repro_fig*.sh presets drive this script exactly that way.
#
# REQUIREMENTS
#   Two NUMA nodes, a raw block device, passwordless sudo (the script
#   writes /proc/sys/kernel/numa_balancing and runs the binary via sudo),
#   and — for NUMA_MIGRATE_METHOD=3 — a kernel providing the move_pages2
#   syscall (462).
#
# THEN PLOT WITH
#   python3 plots/plot_paper.py --sweep <sweep_id> ...   (run from repo root)
#
set -euo pipefail

# ── Swept parameters (Cartesian product; override via *_LIST env vars) ─

# DRAM buffer-pool size in GB — the local/fast tier.
SWEEP_PHYSGB=(${PHYSGB_LIST:-32})
# Remote NUMA tier size in GB. 0 disables the tier, making vmcache-n
# behave as a 2-tier (DRAM->SSD) buffer manager.
SWEEP_REMOTEGB=(${REMOTEGB_LIST:-96})

# Page-migration probabilities, 0.0-1.0. One value here is applied to all
# four ratios at once: DRAM_READ/WRITE (promote REMOTE->DRAM on access)
# and NUMA_READ/WRITE (demote DRAM->REMOTE on eviction). 1 = always
# migrate, 0.1 = migrate 10% of the time.
SWEEP_RATIO=(${RATIO_LIST:-1})


# Worker threads.
SWEEP_THREADS=(${THREADS_LIST:-32})
# Meaning depends on SWEEP_RNDREAD:
#   RNDREAD=0 (TPC-C)      -> number of warehouses, e.g. 1000
#   RNDREAD=1 (rnd read)   -> number of records,    e.g. 1000000000
SWEEP_DATASIZE=(${DATASIZE_LIST:-1000})
# Measurement duration per run, in seconds (excludes dataset load time).
SWEEP_RUNFOR=(${RUNFOR_LIST:-900})

# Minimum pages gathered per inline REMOTE->DRAM promotion batch.
# 1 disables batching (promote one page at a time).
SWEEP_PROMOTE_BATCH=(${PROMOTE_BATCH_LIST:-1})
# Pages per DRAM->REMOTE demotion batch. Also used as
# MOVE_PAGES2_MAX_BATCH_SIZE, the cap on pages per move_pages2 call.
SWEEP_EVICT_BATCH=(${EVICT_BATCH_LIST:-1 256 1024})

# Workload: 0 = TPC-C, 1 = random read.
SWEEP_RNDREAD=(${RNDREAD_LIST:-0})

# How pages are physically moved between NUMA nodes:
#   0 = mbind(), one page at a time      (memory-policy testing)
#   1 = move_pages(), one page at a time (baseline / debugging)
#   2 = move_pages(), batched            (production, recommended)
#   3 = move_pages2(), custom syscall    (needs a patched kernel)
SWEEP_NUMA_MIGRATE_METHOD=(${NUMA_MIGRATE_METHOD_LIST:-0 1 2 3})
# Flags passed to move_pages2. Only swept when method 3 is in use;
# for every other method the loop below pins it to 0.
SWEEP_MOVE_PAGES2_MODE=(${MOVE_PAGES2_MODE_LIST:-0})

# ── Fixed for the whole sweep (override from the environment) ──────────

# Raw block device backing the buffer pool. NOTE: it is written to.
export BLOCK=${BLOCK:-/dev/nvme0n1}
# Use the exmap kernel module instead of plain mmap (needs the module).
export EXMAP=${EXMAP:-0}

# Virtual address space reserved, in GB. Must be >= the dataset size, and
# needs vm.overcommit_memory=1. Not physical memory.
export VIRTGB=${VIRTGB:-894}

# Which NUMA node is the DRAM tier and which is the remote tier.
# Check yours with `numactl --hardware`.
export DRAM_NODE=${DRAM_NODE:-0}
export REMOTE_NODE=${REMOTE_NODE:-1}

# Pages per write-back batch when evicting all the way out to SSD.
# Distinct from EVICT_BATCH, which is the DRAM->REMOTE tier demotion.
export EVICT_BATCH_SSD=${EVICT_BATCH_SSD:-64}

# Scan window for gathering a promotion batch, as a multiple of
# PROMOTE_BATCH. Higher = look further for neighbouring hot pages.
export PROMOTE_BATCH_SCAN_MULTIPLIER=${PROMOTE_BATCH_SCAN_MULTIPLIER:-2}

# ── Disable NUMA balancing ────────────────────────────────────────────
# The kernel's automatic NUMA balancing would migrate pages behind the
# buffer manager's back and corrupt the measurements.
sudo sh -c 'echo 0 > /proc/sys/kernel/numa_balancing'

# ── Run loop ──────────────────────────────────────────────────────────
mkdir -p bench_results

# One sweep = one timestamped ID, shared by every run's filename. This is
# the ID passed to plot_paper.py --sweep.
sweep_start_time=$(date +%Y%m%d_%H%M%S)
summary_file="bench_results/${sweep_start_time}_summary.jsonl"
> "$summary_file"

for phys in "${SWEEP_PHYSGB[@]}"; do
for remote in "${SWEEP_REMOTEGB[@]}"; do
for ratio in "${SWEEP_RATIO[@]}"; do
  # One swept value drives all four migration ratios.
  dram_r=$ratio; dram_w=$ratio; numa_r=$ratio; numa_w=$ratio
for threads in "${SWEEP_THREADS[@]}"; do
for datasize in "${SWEEP_DATASIZE[@]}"; do
for runfor in "${SWEEP_RUNFOR[@]}"; do
for pbatch in "${SWEEP_PROMOTE_BATCH[@]}"; do
for ebatch in "${SWEEP_EVICT_BATCH[@]}"; do
for rndread in "${SWEEP_RNDREAD[@]}"; do
for numa_method in "${SWEEP_NUMA_MIGRATE_METHOD[@]}"; do

  # MOVE_PAGES2_MODE only means anything to the move_pages2 syscall, so
  # for methods 0-2 collapse it to a single value instead of running
  # duplicate configurations.
  if [ "$numa_method" -eq 3 ]; then
    mp2_modes=("${SWEEP_MOVE_PAGES2_MODE[@]}")
  else
    mp2_modes=(0)
  fi

  for mp2_mode in "${mp2_modes[@]}"; do

    # Batch size 1 is only meaningful for the single-page methods (0, 1);
    # pairing it with a batched method would just re-measure method 1.
    if [ "$numa_method" -ge 2 ] && [ "$ebatch" -eq 1 ]; then
      continue
    fi

    # When this individual run started (the sweep ID is the prefix below).
    run_timestamp=$(date +%Y%m%d_%H%M%S)

    # Filename encodes every swept parameter, so a run is identifiable
    # from its name alone. The plotting scripts parse the summary file
    # rather than this, but it keeps bench_results/ browsable.
    tag="${sweep_start_time}_phys${phys}_remote${remote}_dr${dram_r}_dw${dram_w}_nr${numa_r}_nw${numa_w}_t${threads}_data${datasize}_run${runfor}_pb${pbatch}_eb${ebatch}_rnd${rndread}_nm${numa_method}_mp2${mp2_mode}"
    logfile="bench_results/${tag}.log"
    jsonfile="bench_results/${tag}.json"

    # These exports are what vmcache-n actually reads; everything above
    # is just the loop that chooses their values.
    export PHYSGB="$phys"
    export REMOTEGB="$remote"
    export DRAM_READ_RATIO="$dram_r"
    export DRAM_WRITE_RATIO="$dram_w"
    export NUMA_READ_RATIO="$numa_r"
    export NUMA_WRITE_RATIO="$numa_w"
    export THREADS="$threads"
    export DATASIZE="$datasize"
    export RUNFOR="$runfor"
    export PROMOTE_BATCH="$pbatch"
    export EVICT_BATCH="$ebatch"
    export RNDREAD="$rndread"
    export NUMA_MIGRATE_METHOD="$numa_method"
    export MOVE_PAGES2_MODE="$mp2_mode"
    # Deliberately tied to EVICT_BATCH: the demotion batch size is the
    # natural cap on how many pages one move_pages2 call handles.
    export MOVE_PAGES2_MAX_BATCH_SIZE="$ebatch"

    # Per-run config file (human-readable; nested under "config").
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

    # Flat one-line-per-run index. THIS is what plot_paper.py reads, so
    # keys here must stay flat (not nested under "config").
    echo "{\"timestamp\":\"$run_timestamp\",\"sweep_id\":\"$sweep_start_time\",\"tag\":\"$tag\",\"logfile\":\"$logfile\",\"PHYSGB\":$phys,\"REMOTEGB\":$remote,\"DRAM_READ_RATIO\":$dram_r,\"DRAM_WRITE_RATIO\":$dram_w,\"NUMA_READ_RATIO\":$numa_r,\"THREADS\":$threads,\"DATASIZE\":$datasize,\"RUNFOR\":$runfor,\"PROMOTE_BATCH\":$pbatch,\"EVICT_BATCH\":$ebatch,\"RNDREAD\":$rndread,\"NUMA_MIGRATE_METHOD\":$numa_method,\"MOVE_PAGES2_MODE\":$mp2_mode,\"MOVE_PAGES2_MAX_BATCH_SIZE\":$ebatch,\"BLOCK\":\"$BLOCK\",\"EXMAP\":$EXMAP,\"VIRTGB\":$VIRTGB,\"DRAM_NODE\":$DRAM_NODE,\"REMOTE_NODE\":$REMOTE_NODE,\"NUMA_WRITE_RATIO\":$NUMA_WRITE_RATIO,\"EVICT_BATCH_SSD\":$EVICT_BATCH_SSD,\"PROMOTE_BATCH_SCAN_MULTIPLIER\":$PROMOTE_BATCH_SCAN_MULTIPLIER}" >> "$summary_file"

    echo "=== Running: $tag ==="
    # `|| true` so one crashed configuration does not abort the sweep
    # (set -e is on). Check the .log if a series is missing from a plot.
    sudo -E numactl --cpubind=0 ./vmcache-n &> "$logfile" || true

  done

done; done; done; done; done; done; done; done; done; done
