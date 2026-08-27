# CLAUDE.md

## What this is

Implementation of **vmcache^n**, an *n*-tier (DRAM → remote NUMA → SSD)
virtual-memory-assisted buffer pool, from the DaMoN'26 paper *Virtual-Memory
Assisted Buffer Management In Tiered Memory*
([10.1145/3789237.3809129](https://doi.org/10.1145/3789237.3809129)).

## Which file is which

| Source | Binary | Design | Paper label |
| --- | --- | --- | --- |
| `vmcache-n.cpp` | `vmcache-n` | 3-tier: DRAM → remote NUMA → SSD | `vmcache^n` |
| `vmcache-leis.cpp` | `vmcache-leis` | 2-tier: DRAM → SSD, no NUMA code | `vmcache` |
| `vmcache-n-memtrk.cpp` | `vmcache-n-memtrk` | `vmcache-n` + rdtsc timing | — |

`vmcache-leis.cpp` is identical to upstream `viktorleis/vmcache` HEAD; keep it
that way, it is the paper's baseline.

`vmcache-n` with `REMOTEGB=0` degrades to two tiers but is **not** the same as
`vmcache-leis` — it still carries the tier-bit checks. The baseline curve comes
from `vmcache-leis`.

## Build

`make` builds `vmcache-n` and `vmcache-leis`.
`make vmcache-n-debug` / `make vmcache-n-memtrk` are opt-in.

## Running benchmarks

Each sweeper drives exactly one binary — the name says which:

| Script | Binary | Workload | Output dir |
| --- | --- | --- | --- |
| `bench_sweep_n.sh` | `vmcache-n` | TPC-C / random read | `bench_results/` |
| `bench_sweep_n_ycsb.sh` | `vmcache-n` | YCSB | `bench_results_ycsb/` (absent) |
| `bench_sweep_leis.sh` | `vmcache-leis` | TPC-C / random read | `bench_results/` |

Sweep arrays take `*_LIST` environment overrides:
`REMOTEGB_LIST="8 16 32" ./bench_sweep_n.sh`

Headline figure: `./repro_fig1.sh` (TPC-C) and `./repro_fig2.sh` (random read).
Each runs 5 tiered arms + 1 baseline arm (~1.5 h) and prints the plot command.

## Plotting

`plots/plot_paper.py` (main), `plots/plot_time_breakdown.py` (memtrk data).

**Run from the repo root** — both resolve `bench_results/` and `bench_plots/`
from the CWD, not from `__file__`.

`bench_results/___summary.txt` records which command produced which figure. It
is the only such record; keep it updated.

## Live labs (onboarding)

Interactive walkthroughs of the buffer manager — the fastest way to get someone
new oriented before they read the C++.

| Lab | Link | Source |
| --- | --- | --- |
| `vmcache-leis` — 2-tier | [live lab](https://claude.ai/code/artifact/099ad77d-fa73-4138-b11f-62f998864039) | `_scratch/vmcache_leis_lab.html` |
| `vmcache-n` — 3-tier | [live lab](https://claude.ai/code/artifact/dbc7bc76-e558-4faa-a0c7-fc0d8cd55e2b) | `_scratch/vmcache_n_lab.html` |

Both step a transaction through the B-tree descent (metadata → root → inner →
leaf) and show, per page: check state → make room → fetch → use it.

Identical controls in both, so you learn the interface once:

- **Step ▸** runs one transaction operation; **Run** plays the whole thing;
  **Auto** draws transactions against the real TPC-C mix.
- **❚❚ Pause** (or `Space`) stops mid-flight — mid-descent, mid-eviction, even
  mid page-animation — and resumes exactly there.
- **◀ ▶** (or `←` `→`) step through every recorded state; the **scrubber** slides
  through them; **clicking a numbered plan chip jumps straight to that step**.
  The readout shows `23/51 · op 4` — position in the recording, and which
  transaction step it belongs to.
- **Speed** is Slow / Normal / Fast. Shrink the pools to force eviction.

The 2-tier lab has three bands (virtual address space, DRAM frames, device). The
3-tier lab adds a remote-memory band and exposes `DRAM_READ_RATIO`,
`NUMA_READ_RATIO`, `PROMOTE_BATCH` and `NUMA_MIGRATE_METHOD` as live controls —
switching the migration method between per-page (0, 1) and batched (2, 3) shows
the syscall count collapse for the same number of pages moved, which is the
`move_pages2` argument in one number. Set REMOTE to 0 to collapse it to 2-tier.

Two caveats: artifacts are **private until shared** from the page's share menu,
and the lab sources currently live in the gitignored `_scratch/`, so they do not
travel with a clone. Move them to a tracked `labs/` directory if they are meant
to be part of the handoff.

## Gotchas

- **The baseline label comes from an absent key.** `plot_paper.py:140` labels a
  run `vmcache` when it has no `REMOTEGB` field, `vmcache^n(N)` when it does.
  `bench_sweep_leis.sh` emits no such field. The filter at `plot_paper.py:125`
  is `k not in r or ...`, so runs missing a key pass every filter — the only
  reason the baseline survives `--nm 3 --eb 256`. Tighten that predicate and the
  baseline silently disappears from the figure.
- **`EVICT_BATCH` sets two things**: the DRAM→REMOTE demotion batch and
  `MOVE_PAGES2_MAX_BATCH_SIZE`.
- **`DATASIZE` is overloaded**: TPC-C warehouses when `RNDREAD=0`, record count
  when `RNDREAD=1` or under YCSB.
- **`BATCH` vs `EVICT_BATCH`**: `vmcache-leis` reads `BATCH`, `vmcache-n` reads
  `EVICT_BATCH`. Each ignores the other's.
- **A missing series in a plot is a crashed run, not a plotting bug.**
  `bench_sweep_n.sh:226` runs the binary with `|| true` so one bad config cannot
  abort a multi-hour sweep. Check the `.log`.
- **`move_pages2` is syscall 462** (`vmcache-n.cpp:38`), used by
  `NUMA_MIGRATE_METHOD=3`; needs a patched kernel, returns `-ENOSYS` otherwise.
  The `#define ... 451` at `vmcache-n.cpp:863` is dead — the first `#ifndef`
  wins. Both panels of the headline figure use method 3.
- **Tier lives in the page-state word**, bits [55:54]
  (`vmcache-n.cpp:182-206`). Anything touching the version counter must
  preserve them.
- **`_scratch/` is gitignored** and holds retired files plus a 14.5 GB `vm`
  backing file. Never `git add -f` anything there.
- **`.git` is ~5.5 GB** from one orphaned blob. A network clone is ~16 MB; only
  local copies suffer. `git gc --prune=now` reclaims it.

## Known gaps

- `vmcache-n-memtrk` has **no sweeper script**, so the time-breakdown figures
  (sweeps `20260213_130458`, `20260213_184542`) have no reproducible recipe.
- `bench_sweep_n_ycsb.sh` has no committed results and its arrays are not
  `*_LIST`-overridable.
- No `requirements.txt`; plotting needs matplotlib.

## Conventions

- Being prepared for handoff: prefer changes that make the repo
  self-explanatory over changes needing verbal context.
- `bench_results/` and `bench_plots/` are tracked **on purpose**.
- `upstream` push URL is `no_push`. Push to `origin` only.
- `promoteBatch` is the *minimum* batch size we try to promote.
