#!/usr/bin/env python3
import argparse
import json
import csv
import os
import re
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def main():
    parser = argparse.ArgumentParser(description="Plot benchmark results: tx/sec and total I/O (2-row figure)")
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
    parser.add_argument("--out", help="Output filename (without extension); defaults to sweep label")
    parser.add_argument("--logy", action="store_true", help="Use log scale on y-axis for both plots")
    parser.add_argument("--smooth", type=int, default=1,
                        help="Rolling-average window for data movement plot (default: 1 = no smoothing)")
    parser.add_argument("--remo-label", action="store_true", dest="remo_label",
                        help="Remap remo=N labels to vmcache / vmcache+(N) display names")
    parser.add_argument("--nm-label", action="store_true", dest="nm_label",
                        help="Remap nm=N labels to descriptive migration method display names")
    parser.add_argument("--no-legend", action="store_true", dest="no_legend",
                        help="Hide the legend")
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

    def remo_display_label(run):
        remotegb = run.get("REMOTEGB")
        try:
            n = int(remotegb)
            if n > 0:
                return r"vmcache$^+$(" + str(n) + ")"
        except (TypeError, ValueError):
            pass
        return "vmcache"

    def nm_display_label(run):
        nm = str(run.get("NUMA_MIGRATE_METHOD", "?"))
        bs = run.get("EVICT_BATCH", "")
        if nm == "0":
            return f"mbind ({bs})"
        elif nm == "1":
            return f"move_pages ({bs})"
        elif nm == "2":
            return f"move_pages ({bs})"
        elif nm == "3":
            return f"move_pages2 ({bs})"
        return f"nm={nm}({bs})"

    # Fixed color per REMOTEGB value so colors are stable across plots
    remo_colors = {0: "C0", 8: "C1", 16: "C2", 32: "C3", 64: "C4", 128: "C5"}

    # Parse log data for each run
    run_data = []
    for run in runs:
        logfile = run["logfile"]
        ts_list, tx_list, rmb_list, wmb_list, prom_list, dem_list = [], [], [], [], [], []
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
                prom_list.append(float(row.get("promotions", 0)))
                dem_list.append(float(row.get("demotions", 0)))
            except (ValueError, KeyError):
                break
        # Total I/O = read + write MB/s
        total_io = [r + w for r, w in zip(rmb_list, wmb_list)]
        # Total data movement = promotions + demotions
        total_movement = [p + d for p, d in zip(prom_list, dem_list)]
        raw_label = make_label(run)
        if args.remo_label:
            display_label = remo_display_label(run)
        elif args.nm_label:
            display_label = nm_display_label(run)
        else:
            display_label = raw_label
        # Determine color from REMOTEGB config key; fall back to parsing the display label
        remo_key = None
        try:
            val = run.get("REMOTEGB")
            if val is not None:
                remo_key = int(val)
        except (TypeError, ValueError):
            pass
        if remo_key is None:
            if display_label == "vmcache":
                remo_key = 0
            else:
                m = re.search(r'\((\d+)\)', display_label)
                if m:
                    remo_key = int(m.group(1))
        remo_color = remo_colors.get(remo_key, f"C{len(run_data)}") if remo_key is not None else f"C{len(run_data)}"
        run_data.append({
            "label": display_label,
            "ts": ts_list,
            "tx": tx_list,
            "rmb": rmb_list,
            "wmb": wmb_list,
            "total_io": total_io,
            "total_movement": total_movement,
            "color": remo_color,
        })

    def rolling_mean(data, w):
        if w <= 1:
            return data
        result = []
        for i in range(len(data)):
            lo = max(0, i - w // 2)
            hi = min(len(data), i + w // 2 + 1)
            result.append(sum(data[lo:hi]) / (hi - lo))
        return result

    print("Average transactions/sec:")
    for rd in run_data:
        if rd["tx"]:
            print(f"  {rd['label']}: {sum(rd['tx']) / len(rd['tx']):.0f}")

    sweep_label = "+".join(args.sweep)
    out_name = args.out if args.out else sweep_label

    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<"]
    # Space markers evenly; aim for ~15 visible marks across the longest series
    max_len = max((len(rd["ts"]) for rd in run_data), default=1)
    markevery = max(1, max_len // 15)

    # 2-column figure (horizontally stacked, square subplots)
    fig = plt.figure(figsize=(4, 2))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.4)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax1.set_box_aspect(1)
    ax2.set_box_aspect(1)

    # Row 1: Transactions / sec
    for i, rd in enumerate(run_data):
        ax1.plot(rd["ts"], rd["tx"], label=rd["label"], linewidth=0.7,
                 color=rd["color"], marker=markers[i % len(markers)], markevery=markevery, markersize=4)
        # ax1.plot(rd["ts"], rd["tx"], label=rd["label"], linewidth=0.7, color="black",
        #          marker=markers[i % len(markers)], markevery=markevery, markersize=4)
    ax1.set_xlabel("Time [seconds]")
    ax1.set_ylabel("Transactions / sec", labelpad=0)
    if args.logy:
        ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)

    if args.nm_label:
        # Row 2: Total data movement (promotions + demotions) between memory tiers
        for i, rd in enumerate(run_data):
            smoothed = rolling_mean(rd["total_movement"], args.smooth)
            ax2.plot(rd["ts"], smoothed, label=rd["label"], linewidth=0.7,
                     color=rd["color"], marker=markers[i % len(markers)], markevery=markevery, markersize=4)
            # ax2.plot(rd["ts"], smoothed, label=rd["label"], linewidth=0.7, color="black",
            #          marker=markers[i % len(markers)], markevery=markevery, markersize=4)
        ax2.set_xlabel("Time [seconds]")
        ax2.set_ylabel("Data Movement (pages/s)", labelpad=2)
    else:
        # Row 2: Total I/O (read + write) MB/s
        for i, rd in enumerate(run_data):
            ax2.plot(rd["ts"], rd["total_io"], label=rd["label"], linewidth=0.7,
                     color=rd["color"], marker=markers[i % len(markers)], markevery=markevery, markersize=4)
            # ax2.plot(rd["ts"], rd["total_io"], label=rd["label"], linewidth=0.7, color="black",
            #          marker=markers[i % len(markers)], markevery=markevery, markersize=4)
        ax2.set_xlabel("Time [seconds]")
        ax2.set_ylabel("Total I/O (MB/s)", labelpad=0)
        ax2.set_ylim(100)
    if args.logy:
        ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3)

    # Single shared legend in one horizontal line above the first subplot
    if not args.no_legend:
        handles, labels = ax1.get_legend_handles_labels()

        legend_pos = (1.0, 1.05)
        if args.nm_label:
            legend_pos = (1.0, 1.05)
        elif args.remo_label:
            legend_pos = (1.15, 1.4)

        ax1.legend(handles, labels, loc="upper center",
                   bbox_to_anchor=legend_pos,
                   ncol=len(run_data) if args.nm_label else len(run_data)/2,
                   fontsize="small",
                   frameon=True,
                   columnspacing=0.1,
                   handletextpad=0.1)

    out_path = os.path.join(args.outdir, f"{out_name}_combined.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    out_path = os.path.join(args.outdir, f"{out_name}_combined.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
