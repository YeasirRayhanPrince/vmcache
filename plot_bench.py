#!/usr/bin/env python3
import argparse
import json
import csv
import os
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Plot benchmark sweep results")
    parser.add_argument("--sweep", required=True, nargs="+", help="Sweep ID timestamp(s) (e.g., 20260208_193523)")
    parser.add_argument("--outdir", default="bench_plots", help="Output directory for figures")
    parser.add_argument("--skip", type=int, default=0, help="Seconds to skip from start (warmup)")
    parser.add_argument("--ycsb", action="store_true", help="Plot YCSB results (from bench_results_ycsb/)")
    parser.add_argument("--eb", nargs="+", type=int, help="Filter EVICT_BATCH values")
    parser.add_argument("--pb", nargs="+", type=int, help="Filter PROMOTE_BATCH values")
    parser.add_argument("--nm", nargs="+", type=int, help="Filter NUMA_MIGRATE_METHOD values")
    parser.add_argument("--mp2m", nargs="+", type=int, help="Filter MOVE_PAGES2_MODE values")
    parser.add_argument("--mp2b", nargs="+", type=int, help="Filter MOVE_PAGES2_MAX_BATCH_SIZE values")
    parser.add_argument("--t", nargs="+", type=int, help="Filter THREADS values")
    parser.add_argument("--ebs", nargs="+", type=int, help="Filter EVICT_BATCH_SSD values")
    parser.add_argument("--pbsm", nargs="+", type=int, help="Filter PROMOTE_BATCH_SCAN_MULTIPLIER values")
    parser.add_argument("--workload", nargs="+", help="Filter YCSB workload letters (e.g., A C)")
    parser.add_argument("--zipf", nargs="+", type=float, help="Filter ZIPF_THETA values")
    parser.add_argument("--ts", nargs="+", type=int, help="Filter YCSB_TUPLE_SIZE values")
    parser.add_argument("--sel", nargs="+", help="Filter YCSB_SCAN_SELECTIVITY values")
    parser.add_argument("--remo", nargs="+", type=int, help="Filter REMOTEGB values")
    args = parser.parse_args()

    # Map CLI filter flags to config keys
    filters = {
        "EVICT_BATCH": args.eb,
        "PROMOTE_BATCH": args.pb,
        "NUMA_MIGRATE_METHOD": args.nm,
        "MOVE_PAGES2_MODE": args.mp2m,
        "MOVE_PAGES2_MAX_BATCH_SIZE": args.mp2b,
        "THREADS": args.t,
        "EVICT_BATCH_SSD": args.ebs,
        "PROMOTE_BATCH_SCAN_MULTIPLIER": args.pbsm,
        "YCSB": args.workload,
        "ZIPF_THETA": args.zipf,
        "YCSB_TUPLE_SIZE": args.ts,
        "YCSB_SCAN_SELECTIVITY": args.sel,
        "REMOTEGB": args.remo,
    }
    filters = {k: v for k, v in filters.items() if v is not None}

    os.makedirs(args.outdir, exist_ok=True)
    runs = []
    for sweep_id in args.sweep:
        results_dir = "bench_results_ycsb" if args.ycsb else "bench_results"
        summary_path = os.path.join(results_dir, f"{sweep_id}_summary.jsonl")
        with open(summary_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    runs.append(json.loads(line))

    # Apply filters
    if filters:
        runs = [r for r in runs if all(k not in r or str(r[k]) in [str(x) for x in v] for k, v in filters.items())]
        if not runs:
            print("No runs match the given filters.")
            return

    # Auto-detect varying parameters
    skip_keys = {"timestamp", "sweep_id", "tag", "logfile"}
    all_keys = dict.fromkeys(k for r in runs for k in r if k not in skip_keys)
    config_keys = list(all_keys)
    varying = []
    for k in config_keys:
        vals = set(str(r.get(k)) for r in runs)
        if len(vals) > 1:
            varying.append(k)

    # Short name mapping for labels
    short = {
        "NUMA_MIGRATE_METHOD": "nm",
        "EVICT_BATCH": "eb",
        "PROMOTE_BATCH": "pb",
        "THREADS": "t",
        "DRAM_READ_RATIO": "dr",
        "DRAM_WRITE_RATIO": "dw",
        "NUMA_READ_RATIO": "nr",
        "MOVE_PAGES2_MODE": "mp2m",
        "MOVE_PAGES2_MAX_BATCH_SIZE": "mp2b",
        "PROMOTE_BATCH_SCAN_MULTIPLIER": "pbsm",
        "EVICT_BATCH_SSD": "ebs",
        "YCSB": "ycsb",
        "ZIPF_THETA": "zipf",
        "YCSB_TUPLE_SIZE": "ts",
        "YCSB_SCAN_SELECTIVITY": "sel",
        "REMOTEGB": "remo",
    }

    # Build subtitle from constant parameters
    constant = [k for k in config_keys if k not in varying]
    const_parts = []
    for k in constant:
        name = short.get(k, k.lower()[:4])
        const_parts.append(f"{name}={runs[0].get(k, '?')}")
    subtitle = ", ".join(const_parts)

    def make_label(run):
        parts = []
        for k in varying:
            name = short.get(k, k.lower()[:4])
            parts.append(f"{name}={run.get(k, '?')}")
        return ", ".join(parts) if parts else run["tag"]

    # Parse log data for each run
    run_data = []
    for run in runs:
        logfile = run["logfile"]
        ts_list, tx_list, rmb_list, wmb_list = [], [], [], []
        with open(logfile) as f:
            lines = f.readlines()
        # Find CSV header
        header_idx = None
        for i, line in enumerate(lines):
            if line.startswith("ts,"):
                header_idx = i
                break
        if header_idx is None:
            continue
        reader = csv.DictReader(lines[header_idx:])
        for row in reader:
            if row.get("ts", "").startswith("="):
                break
            try:
                t = int(row["ts"])
                if t < args.skip:
                    continue
                ts_list.append(t - args.skip)
                tx_list.append(float(row["tx"]))
                rmb_list.append(float(row["rmb"]))
                wmb_list.append(float(row["wmb"]))
            except (ValueError, KeyError):
                break
        run_data.append({
            "label": make_label(run),
            "ts": ts_list, "tx": tx_list, "rmb": rmb_list, "wmb": wmb_list,
        })

    sweep_label = "+".join(args.sweep)

    # Figure 1: Transactions/sec vs Time
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    for rd in run_data:
        ax1.plot(rd["ts"], rd["tx"], label=rd["label"])
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Transactions/sec")
    ax1.set_title(f"Transactions/sec — sweep {sweep_label}\n{subtitle}", fontsize=10)
    # if len(run_data) <= 10:
    ax1.legend(fontsize="small")
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(os.path.join(args.outdir, f"{sweep_label}_tx.png"), dpi=150)
    print(f"Saved: {args.outdir}/{sweep_label}_tx.png")

    # Figure 2: IO MB/s vs Time (2 lines per run: read + write)
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    for rd in run_data:
        ax2.plot(rd["ts"], rd["rmb"], label=f'{rd["label"]} (read)', linestyle="-")
        ax2.plot(rd["ts"], rd["wmb"], label=f'{rd["label"]} (write)', linestyle="--")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("MB/s")
    ax2.set_title(f"IO MB/s — sweep {sweep_label}\n{subtitle}", fontsize=10)
    if 2 * len(run_data) <= 10:
        ax2.legend(fontsize="small")
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(os.path.join(args.outdir, f"{sweep_label}_io.png"), dpi=150)
    print(f"Saved: {args.outdir}/{sweep_label}_io.png")

    plt.show()


if __name__ == "__main__":
    main()
