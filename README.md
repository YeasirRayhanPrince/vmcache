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
Batch nearby hot pages together when promoting from REMOTE to DRAM:
* PROMOTE_BATCH: pages to collect per inline promotion batch; default=64
  - Set to 1 to disable batching (single-page promotion)
  - Set to higher values (e.g., 128, 256) for aggressive batching
* PROMOTE_BATCH_SIZE_MAX: hard maximum limit on promotion batch size; default=256

### NUMA Migration Methods (NEW)
Control how pages are migrated between NUMA nodes:
* NUMA_MIGRATE_METHOD: migration method to use; default=2
  - 0 = mbind() single page (memory policy testing)
  - 1 = move_pages() single page (baseline/debugging)
  - 2 = move_pages() batched (production, recommended)
  - 3 = move_pages2() custom syscall (research/custom kernel)
* NUMA_MIGRATE_BATCH_SIZE: batch size for methods 2-3; default=64
* MOVE_PAGES2_MODE: flags/mode for move_pages2 custom syscall; default=0

### NUMA Node Configuration
* DRAM_NODE: NUMA node for DRAM tier; default=0
* REMOTE_NODE: NUMA node for remote tier; default=1

### Benchmark and Eviction Configuration
* BATCH: batch size for demotion/eviction in pages; default=64
* RUNFOR: benchmark run duration in seconds; default=30
* RNDREAD: if non-zero, run random read benchmark, otherwise TPC-C; default=0
* THREADS: number of threads; default=1
* DATASIZE: number of warehouses for TPC-C, number of tuples for random read benchmark; default=10

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
    PROMOTE_BATCH=128 NUMA_MIGRATE_METHOD=2 NUMA_MIGRATE_BATCH_SIZE=128 ./vmcache-n
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
    PROMOTE_BATCH=256 NUMA_MIGRATE_METHOD=3 MOVE_PAGES2_MODE=0x01 ./vmcache-n
  ```

## Dependencies and Configuration

### Required Libraries
* libaio: For asynchronous I/O. On Ubuntu: `sudo apt install libaio-dev`
* libnuma: For NUMA memory management. On Ubuntu: `sudo apt install libnuma-dev`
  - Required for multi-tier memory support (REMOTEGB > 0) and NUMA migration methods

### System Configuration
You will probably also need to set `vm.overcommit_memory = 1` in `/etc/sysctl.conf`. Otherwise larger values of VIRTGB will not work.
```bash
  echo 'vm.overcommit_memory = 1' | sudo tee -a /etc/sysctl.conf                                            
  sudo sysctl -p
```

### Optional: EXMAP Support
If you want to use EXMAP (EXMAP=1), you need the [exmap kernel module](https://github.com/tuhhosg/exmap).

### Optional: Custom move_pages2 Syscall
For NUMA_MIGRATE_METHOD=3, a custom kernel module implementing the move_pages2 syscall is required. This is for research/experimentation only.

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
