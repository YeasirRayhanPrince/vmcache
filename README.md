# vmcache

Code repository for SIGMOD'23 paper [Virtual-Memory Assisted Buffer Management](https://www.cs.cit.tum.de/fileadmin/w00cfj/dis/_my_direct_uploads/vmcache.pdf). This is the buffer manager implementation. [exmap](https://github.com/tuhhosg/exmap) is in a separate repository.

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

### Basic Usage
* TPC-C, 4 threads, 2 warehouses: `BLOCK=/dev/nvme0n1 THREADS=4 DATASIZE=2 ./vmcache`
* random read, 10 threads, 1 million tuples: `BLOCK=/dev/md0 THREADS=10 DATASIZE=1e6 ./vmcache`

### Multi-Tier (DRAM + REMOTE NUMA)
* With remote NUMA tier, aggressive batched promotion:
  ```bash
  BLOCK=/dev/nvme0n1 PHYSGB=4 REMOTEGB=16 THREADS=8 DATASIZE=10 \
    PROMOTE_BATCH=128 NUMA_MIGRATE_METHOD=2 NUMA_MIGRATE_BATCH_SIZE=128 ./vmcache
  ```

* Conservative promotion (10% of reads, 30% of writes):
  ```bash
  BLOCK=/dev/nvme0n1 PHYSGB=4 REMOTEGB=8 THREADS=4 \
    DRAM_READ_RATIO=0.1 DRAM_WRITE_RATIO=0.3 \
    PROMOTE_BATCH=32 NUMA_MIGRATE_METHOD=2 ./vmcache
  ```

* Disable promotion batching (single-page, baseline):
  ```bash
  BLOCK=/dev/nvme0n1 PHYSGB=4 REMOTEGB=8 \
    PROMOTE_BATCH=1 NUMA_MIGRATE_METHOD=1 ./vmcache
  ```

* Custom syscall (move_pages2) with research flags:
  ```bash
  BLOCK=/dev/nvme0n1 PHYSGB=4 REMOTEGB=16 \
    PROMOTE_BATCH=256 NUMA_MIGRATE_METHOD=3 MOVE_PAGES2_MODE=0x01 ./vmcache
  ```

## Dependencies and Configuration

### Required Libraries
* libaio: For asynchronous I/O. On Ubuntu: `sudo apt install libaio-dev`
* libnuma: For NUMA memory management. On Ubuntu: `sudo apt install libnuma-dev`
  - Required for multi-tier memory support (REMOTEGB > 0) and NUMA migration methods

### System Configuration
You will probably also need to set `vm.overcommit_memory = 1` in `/etc/sysctl.conf`. Otherwise larger values of VIRTGB will not work.

### Optional: EXMAP Support
If you want to use EXMAP (EXMAP=1), you need the [exmap kernel module](https://github.com/tuhhosg/exmap).

### Optional: Custom move_pages2 Syscall
For NUMA_MIGRATE_METHOD=3, a custom kernel module implementing the move_pages2 syscall is required. This is for research/experimentation only.

## Citation

Please cite the paper as follows:

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
