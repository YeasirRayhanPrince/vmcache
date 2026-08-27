# vmcache<sup>n</sup>

Implementation of **vmcache<sup>n</sup>**, an *n*-tier virtual-memory-assisted buffer
pool (DRAM → remote memory → disk), from the DaMoN'26 paper
[Virtual-Memory Assisted Buffer Management In Tiered Memory](https://doi.org/10.1145/3789237.3809129)
(also on [arXiv](https://arxiv.org/abs/2603.03271)).

Tiered memory architectures pair the host's local DRAM with slower but
byte-addressable *remote memory* (RMem) — NUMA memory on a remote socket,
chiplet-attached memory, or memory reached over RDMA/CXL. vmcache<sup>n</sup>
generalizes two-tier (DRAM–Disk) virtual-memory assisted buffer management to
an *n*-tier (DRAM–RMem–Disk) setting, using the virtual memory subsystem and OS
calls to migrate pages across tiers. Because page migration becomes the
bottleneck in this setup, the paper also introduces a `move_pages2` system call
giving the buffer pool fine-grained control over migration
(`NUMA_MIGRATE_METHOD=3`, see below).

This repository also contains the original two-tier `vmcache` from the SIGMOD'23
paper [Virtual-Memory Assisted Buffer Management](https://www.cs.cit.tum.de/fileadmin/w00cfj/dis/_my_direct_uploads/vmcache.pdf),
which serves as the baseline in our experiments. [exmap](https://github.com/tuhhosg/exmap)
is in a separate repository.

## Source Files

| Source | Binary | Design | Paper label |
| --- | --- | --- | --- |
| `vmcache-leis.cpp` | `vmcache-leis` | **2-tier**: DRAM → SSD. No NUMA code. Original SIGMOD'23 implementation. | `vmcache` |
| `vmcache-n.cpp` | `vmcache-n` | **3-tier**: DRAM → remote NUMA → SSD. Set `REMOTEGB>0` to enable the remote tier (`REMOTEGB=0` falls back to 2-tier). | `vmcache`<sup>`n`</sup> |
| `vmcache-n-memtrk.cpp` | `vmcache-n-memtrk` | `vmcache-n` plus rdtsc instrumentation for disk-I/O and page-migration cycles. Used for the time-breakdown plots. | — |

Build any of them with `make <binary>`, e.g. `make vmcache-n`.

Only `vmcache-n` (and `vmcache-n-memtrk`) support the YCSB workload and the
multi-tier environment variables below; `vmcache-leis` understands only the
storage, benchmark, and eviction settings.

## Environment Variables

### Storage and Memory Configuration
* BLOCK: storage block device (e.g. /dev/nvme0n1 or /dev/md0); default=/tmp/bm
* VIRTGB: virtual memory allocation in GB (e.g., 1024), should be at least device size; default=16
* PHYSGB: physical memory allocation in GB = DRAM buffer pool size, should be less than available RAM; default=4
* REMOTEGB: remote NUMA memory allocation in GB (0 = no remote tier); default=0
* EXMAP: if non-zero, use exmap interface, requires exmap kernel module; default=0

### Multi-Tier Migration Policy (Hyrise-style)
Probabilistic decision ratios for tier selection (0.0 = never use, 1.0 = always use):
* DRAM_READ_RATIO: promote REMOTE→DRAM on read access (0.0-1.0); default=1.0
* DRAM_WRITE_RATIO: promote REMOTE→DRAM on write access (0.0-1.0); default=1.0
* NUMA_READ_RATIO: demote DRAM→REMOTE on eviction during read (0.0-1.0); default=1.0
* NUMA_WRITE_RATIO: demote DRAM→REMOTE on eviction during write (0.0-1.0); default=1.0

### Inline Batched Promotion (NEW)
Batch nearby hot pages together when promoting from REMOTE to DRAM
(`vmcache-n` only):
* PROMOTE_BATCH: target number of pages to collect per inline promotion batch; default=64
  - Set to 1 to disable batching (single-page promotion)
  - Set to higher values (e.g., 128, 256) for aggressive batching
* PROMOTE_BATCH_SCAN_MULTIPLIER: how far to scan the remote clock looking for
  batch candidates, as a multiple of PROMOTE_BATCH (scan window =
  `PROMOTE_BATCH * PROMOTE_BATCH_SCAN_MULTIPLIER`); default=8

### NUMA Migration Methods (NEW)
Control how pages are migrated between NUMA nodes (`vmcache-n` only):
* NUMA_MIGRATE_METHOD: migration method to use; default=2
  - 0 = mbind() single page (memory policy testing)
  - 1 = move_pages() single page (baseline/debugging)
  - 2 = move_pages() batched (production, recommended)
  - 3 = move_pages2() custom syscall (research/custom kernel)
* MOVE_PAGES2_MAX_BATCH_SIZE: `nr_max_batched_migration` argument passed to
  move_pages2; only used by NUMA_MIGRATE_METHOD=3; default=64
* MOVE_PAGES2_MODE: flags/mode for move_pages2 custom syscall; default=0

  Methods 0-2 have no batch-size knob of their own: they migrate exactly the
  batch handed to them by the promotion/eviction path (see PROMOTE_BATCH and
  EVICT_BATCH).

### NUMA Node Configuration
* DRAM_NODE: NUMA node for DRAM tier; default=0
* REMOTE_NODE: NUMA node for remote tier; default=1

### Benchmark and Eviction Configuration
* RUNFOR: benchmark run duration in seconds; default=30
* RNDREAD: if non-zero, run random read benchmark, otherwise TPC-C; default=0
* THREADS: number of threads; default=1
* DATASIZE: number of warehouses for TPC-C, number of tuples for random read benchmark; default=10

Eviction batch size — the two binaries use different variables here:
* BATCH (`vmcache-leis` only): batch size for eviction in pages; default=64
* EVICT_BATCH (`vmcache-n` only): batch size for demotion to the remote tier; default=64
* EVICT_BATCH_SSD (`vmcache-n` only): batch size for eviction to SSD; default=64
  - `vmcache-n` ignores BATCH; `vmcache-leis` ignores both EVICT_BATCH variables.

### YCSB Workload — `vmcache-n` only
* YCSB: if set, run YCSB instead of TPC-C/random read; the first character
  selects the workload (A-F), e.g. `YCSB=A`. Unset = not a YCSB run.
  With YCSB, DATASIZE is the number of records.
* ZIPF_THETA: zipfian skew of the key distribution (0 = uniform, higher = more skewed); default=0.99
* YCSB_TUPLE_SIZE: record size in bytes; default=100
* YCSB_SCAN_SELECTIVITY: fraction of the table touched by a scan; default=1e-7

## Example Command Lines

### Basic Usage (2-tier baseline)
* TPC-C, 4 threads, 2 warehouses: `BLOCK=/dev/nvme0n1 THREADS=4 DATASIZE=2 ./vmcache-leis`
* random read, 10 threads, 1 million tuples: `BLOCK=/dev/md0 THREADS=10 DATASIZE=1e6 ./vmcache-leis`

`./vmcache-n` accepts these same settings and behaves as a 2-tier buffer
manager when `REMOTEGB` is unset or `0`.

### Multi-Tier (DRAM + REMOTE NUMA) — `vmcache-n` only
* With remote NUMA tier, aggressive batched promotion:
  ```bash
  BLOCK=/dev/nvme0n1 PHYSGB=4 REMOTEGB=16 THREADS=8 DATASIZE=10 \
    PROMOTE_BATCH=128 EVICT_BATCH=128 NUMA_MIGRATE_METHOD=2 ./vmcache-n
  ```

* Conservative promotion (10% of reads, 30% of writes):
  ```bash
  BLOCK=/dev/nvme0n1 PHYSGB=4 REMOTEGB=8 THREADS=4 \
    DRAM_READ_RATIO=0.1 DRAM_WRITE_RATIO=0.3 \
    PROMOTE_BATCH=32 NUMA_MIGRATE_METHOD=2 ./vmcache-n
  ```

* Disable promotion batching (single-page, baseline):
  ```bash
  BLOCK=/dev/nvme0n1 PHYSGB=4 REMOTEGB=8 \
    PROMOTE_BATCH=1 NUMA_MIGRATE_METHOD=1 ./vmcache-n
  ```

* Custom syscall (move_pages2) with research flags:
  ```bash
  BLOCK=/dev/nvme0n1 PHYSGB=4 REMOTEGB=16 \
    PROMOTE_BATCH=256 NUMA_MIGRATE_METHOD=3 MOVE_PAGES2_MODE=0x01 \
    MOVE_PAGES2_MAX_BATCH_SIZE=256 ./vmcache-n
  ```

* YCSB A (50/50 read/write), 1e9 records, skewed:
  ```bash
  BLOCK=/dev/nvme0n1 PHYSGB=32 REMOTEGB=32 THREADS=32 DATASIZE=1000000000 \
    YCSB=A ZIPF_THETA=0.99 YCSB_TUPLE_SIZE=112 ./vmcache-n
  ```

## Reproducing the Paper Figures

`make` builds the two binaries the figures need, `vmcache-n` and `vmcache-leis`.

* `./repro_fig1.sh` (TPC-C) and `./repro_fig2.sh` (random read) each run the
  6 arms of the corresponding figure — 5 vmcache<sup>n</sup> arms at
  `REMOTEGB` = 8/16/32/64/128 plus the 2-tier `vmcache` baseline. Each script
  takes roughly 1.5 h (6 x 900 s, plus load time per run) and prints the exact
  plotting command, with the sweep IDs it produced, when it finishes.
  Override the device with e.g. `BLOCK=/dev/nvme1n1 ./repro_fig1.sh`.
* Parameter sweeps (Cartesian product of `*_LIST` env vars, see the header
  comment in each script):
  - `./bench_sweep_n.sh` — `vmcache-n`, TPC-C / random read
  - `./bench_sweep_leis.sh` — `vmcache-leis` baseline
  - `./bench_sweep_n_ycsb.sh` — `vmcache-n`, YCSB
  Each sweep writes `bench_results/<sweep_id>_summary.jsonl` (the index the
  plotting scripts read) alongside a `.json` and `.log` per run.
* Plotting: `python3 plots/plot_paper.py --sweep <ID> [<ID> ...] ...`
  **must be run from the repository root** — it resolves `bench_results/` and
  `bench_plots/` relative to the current working directory.
* `bench_results/___summary.txt` is the hand-maintained log of which sweep ID
  produced which figure.

## Dependencies and Configuration

### Required Libraries
* libaio: For asynchronous I/O. On Ubuntu: `sudo apt install libaio-dev`
* libnuma: For NUMA memory management. On Ubuntu: `sudo apt install libnuma-dev`
  - Required for multi-tier memory support (REMOTEGB > 0) and NUMA migration methods
* numactl: Every sweep script pins the binary with `numactl --cpubind=0`, and
  `numactl --hardware` is how you inspect the NUMA topology.
  On Ubuntu: `sudo apt install numactl`
* matplotlib: Required by the plotting scripts under `plots/` (there is no
  `requirements.txt`). On Ubuntu: `pip3 install matplotlib`

### Environment Prerequisites
Needed to run the benchmarks (not to build):
* **Two NUMA nodes** for the 3-tier configuration: the DRAM tier lives on
  `DRAM_NODE` (default 0) and the remote tier on `REMOTE_NODE` (default 1).
  Check your topology with `numactl --hardware`.
* **A raw block device** for `BLOCK` (e.g. `/dev/nvme0n1`). **It will be
  written to** — the benchmark writes pages directly to the device, destroying
  anything already on it. Point `BLOCK` at a scratch device, or leave it at the
  default `/tmp/bm` file for small local runs.
* **Passwordless sudo**: the sweep scripts disable automatic NUMA balancing
  (`/proc/sys/kernel/numa_balancing`) and launch the binary under
  `sudo -E numactl`.
* `vm.overcommit_memory = 1` (see below).
* For `NUMA_MIGRATE_METHOD=3`: a **patched kernel** providing the `move_pages2`
  system call (syscall number 462). Without it, method 3 will not work; use
  method 2 instead.

### System Configuration
You will probably also need to set `vm.overcommit_memory = 1` in `/etc/sysctl.conf`. Otherwise larger values of VIRTGB will not work.
```bash
  echo 'vm.overcommit_memory = 1' | sudo tee -a /etc/sysctl.conf                                            
  sudo sysctl -p
```

### Optional: EXMAP Support
If you want to use EXMAP (EXMAP=1), you need the [exmap kernel module](https://github.com/tuhhosg/exmap).

### Optional: Custom move_pages2 Syscall
For NUMA_MIGRATE_METHOD=3, a patched kernel providing the `move_pages2` system
call (number 462) is required. This is for research/experimentation only.

## Citation

If you use this code, please cite the vmcache<sup>n</sup> paper:

```
@inproceedings{vmcachen,
  author    = {Yeasir Rayhan and Walid G. Aref},
  title     = {Virtual-Memory Assisted Buffer Management In Tiered Memory},
  booktitle = {International Workshop on Data Management on New Hardware (DaMoN)},
  year      = {2026},
  publisher = {ACM},
  doi       = {10.1145/3789237.3809129},
}
```

The two-tier baseline (`vmcache-leis`) is from the original SIGMOD'23 paper:

```
@inproceedings{vmcache,
  author    = {Viktor Leis and Adnan Alhomssi and Tobias Ziegler and Yannick Loeck and Christian Dietrich},
  title     = {Virtual-Memory Assisted Buffer Management},
  booktitle = {SIGMOD},
  year      = {2023},
}
```

## Low-Hanging Fruit (TODO)

* use C++ wait/notify to handle lock contention instead of spinning
* implement free space management for storage
