#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import re
import shlex
from collections import defaultdict

import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


def build_parser(require_sweep=True, include_panel_spec=True):
    parser = argparse.ArgumentParser(description="Plot benchmark results: tx/sec and total I/O (2-row figure)")
    parser.add_argument("--sweep", required=require_sweep, nargs="+", help="Sweep ID timestamp(s) (e.g., 20260208_193523)")
    if include_panel_spec:
        parser.add_argument("--panel-spec", action="append",
                            help="Quoted plot argument fragment for one 2-column panel; repeat to combine panels")
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
    parser.add_argument("--subplot-w", type=float, default=1.35,
                        help="Width of each standard subplot in inches")
    parser.add_argument("--subplot-h", type=float, default=1.35,
                        help="Height of each standard subplot in inches")
    parser.add_argument("--smooth", type=int, default=1,
                        help="Rolling-average window for data movement plot (default: 1 = no smoothing)")
    parser.add_argument("--remo-label", action="store_true", dest="remo_label",
                        help="Remap remo=N labels to vmcache / vmcache+(N) display names")
    parser.add_argument("--nm-label", action="store_true", dest="nm_label",
                        help="Remap nm=N labels to descriptive migration method display names")
    parser.add_argument("--no-legend", action="store_true", dest="no_legend",
                        help="Hide the legend")
    parser.add_argument("--batch-label", action="store_true", dest="batch_label",
                        help="Scatter plot: avg tx/sec vs nr_max_batched_migration (MOVE_PAGES2_MAX_BATCH_SIZE), colored by MOVE_PAGES2_MODE")
    parser.add_argument("--mig-flag", action="store_true", dest="mig_flag",
                        help="Line plot: avg tx/sec vs ratio (DRAM/NUMA read/write), one line per NR_MAX_BATCHED_MIGRATION, colored by RNDREAD workload")
    return parser


SHORT_NAMES = {
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

REMO_COLORS = {0: "C0", 8: "C1", 16: "C2", 32: "C3", 64: "C4", 128: "C5"}
NM_COLORS = {0: "C0", 1: "C1", 2: "C2", 3: "C3"}
MP2M_MARKERS = {0: "o", 1: "s", 2: "^", 3: "D"}
RNDREAD_COLORS = {0: "green", 1: "red"}
MP2M_LABELS = {0: "MIGRATE_ASYNC", 1: "MIGRATE_SYNC", 2: "MIGRATE_SYNC_LIGHT", 3: "mode 3"}
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<"]
AXIS_LABEL_FONTSIZE = 5
TICK_LABEL_FONTSIZE = 5
LEGEND_FONTSIZE = 4
YLABEL_PAD = 0
XLABEL_PAD = -0.25
TICK_PAD = 0
EXPORT_PAD_INCHES = 0.01


def filters_from_args(args):
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
    return {k: v for k, v in filters.items() if v is not None}


def load_runs(args):
    runs = []
    results_dir = "bench_results_ycsb" if args.ycsb else "bench_results"
    for sweep_id in args.sweep:
        summary_path = os.path.join(results_dir, f"{sweep_id}_summary.jsonl")
        with open(summary_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    runs.append(json.loads(line))
    filters = filters_from_args(args)
    if filters:
        runs = [r for r in runs if all(k not in r or str(r[k]) in [str(x) for x in v] for k, v in filters.items())]
    return runs


def rolling_mean(data, window):
    if window <= 1:
        return data
    result = []
    for i in range(len(data)):
        lo = max(0, i - window // 2)
        hi = min(len(data), i + window // 2 + 1)
        result.append(sum(data[lo:hi]) / (hi - lo))
    return result


def remo_display_label(run):
    remotegb = run.get("REMOTEGB")
    try:
        value = int(remotegb)
        if value > 0:
            return r"vmcache$^n$(" + str(value) + ")"
    except (TypeError, ValueError):
        pass
    return "vmcache"


def nm_display_label(run):
    nm = str(run.get("NUMA_MIGRATE_METHOD", "?"))
    batch = run.get("EVICT_BATCH", "")
    if nm == "0":
        return f"mbind ({batch})" if str(batch) == "1" else "mbind"
    if nm == "1":
        return f"mbind ({batch})"
    if nm == "2":
        return "move_pages"
    if nm == "3":
        return "move_pages2"
    return f"nm={nm}"


def make_label(run, varying):
    parts = []
    for key in varying:
        name = SHORT_NAMES.get(key, key.lower()[:4])
        parts.append(f"{name}={run.get(key, '?')}")
    return ", ".join(parts) if parts else run["tag"]


def build_run_data(args):
    runs = load_runs(args)
    if not runs:
        return None

    skip_keys = {"timestamp", "sweep_id", "tag", "logfile"}
    all_keys = dict.fromkeys(key for run in runs for key in run if key not in skip_keys)
    config_keys = list(all_keys)
    varying = [key for key in config_keys if len({str(run.get(key)) for run in runs}) > 1]

    run_data = []
    for run in runs:
        ts_list, tx_list, rmb_list, wmb_list, prom_list, dem_list = [], [], [], [], [], []
        with open(run["logfile"]) as f:
            lines = f.readlines()

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
                ts = int(row["ts"])
                if ts < args.skip:
                    continue
                ts_list.append(ts - args.skip)
                tx_list.append(float(row["tx"]))
                rmb_list.append(float(row["rmb"]))
                wmb_list.append(float(row["wmb"]))
                prom_list.append(float(row.get("promotions", 0)))
                dem_list.append(float(row.get("demotions", 0)))
            except (ValueError, KeyError):
                break

        raw_label = make_label(run, varying)
        if args.remo_label:
            display_label = remo_display_label(run)
        elif args.nm_label:
            display_label = nm_display_label(run)
        else:
            display_label = raw_label

        if args.nm_label:
            try:
                nm_val = int(run.get("NUMA_MIGRATE_METHOD"))
                color = NM_COLORS.get(nm_val, f"C{len(run_data)}")
            except (TypeError, ValueError):
                color = f"C{len(run_data)}"
        else:
            remo_key = None
            try:
                value = run.get("REMOTEGB")
                if value is not None:
                    remo_key = int(value)
            except (TypeError, ValueError):
                pass
            if remo_key is None:
                if display_label == "vmcache":
                    remo_key = 0
                else:
                    match = re.search(r"\((\d+)\)", display_label)
                    if match:
                        remo_key = int(match.group(1))
            color = REMO_COLORS.get(remo_key, f"C{len(run_data)}") if remo_key is not None else f"C{len(run_data)}"

        run_data.append({
            "label": display_label,
            "ts": ts_list,
            "tx": tx_list,
            "rmb": rmb_list,
            "wmb": wmb_list,
            "total_io": [r + w for r, w in zip(rmb_list, wmb_list)],
            "total_movement": [p + d for p, d in zip(prom_list, dem_list)],
            "color": color,
        })

    return {"args": args, "runs": runs, "run_data": run_data}


def print_summary(panel):
    print("Average transactions/sec:")
    for rd in panel["run_data"]:
        if rd["tx"]:
            print(f"  {rd['label']}: {sum(rd['tx']) / len(rd['tx']):.0f}")
    print("Average page transfers/sec:")
    for rd in panel["run_data"]:
        if rd["total_movement"]:
            print(f"  {rd['label']}: {sum(rd['total_movement']) / len(rd['total_movement']):.0f}")
    print("Average disk I/O MB/s:")
    for rd in panel["run_data"]:
        if rd["total_io"]:
            print(f"  {rd['label']}: {sum(rd['total_io']) / len(rd['total_io']):.1f}")


def render_batch(panel, fig, out_name, outdir):
    args = panel["args"]
    run_data = panel["run_data"]
    runs = panel["runs"]

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.set_box_aspect(1)

    seen_combos = {}
    pairs = sorted(zip(run_data, runs),
                   key=lambda pair: 0 if int(pair[1].get("MOVE_PAGES2_MODE", 0) or 0) != 0 else 1)
    for rd_run, run in pairs:
        avg_tx = sum(rd_run["tx"]) / len(rd_run["tx"]) if rd_run["tx"] else 0
        mp2b_val = run.get("MOVE_PAGES2_MAX_BATCH_SIZE", 1)
        if int(mp2b_val) == 1:
            continue
        try:
            mp2m_val = int(run.get("MOVE_PAGES2_MODE", 0))
        except (TypeError, ValueError):
            mp2m_val = 0
        try:
            rr_val = int(run.get("RNDREAD", 0))
        except (TypeError, ValueError):
            rr_val = 0
        color = RNDREAD_COLORS.get(rr_val, f"C{rr_val}")
        marker = MP2M_MARKERS.get(mp2m_val, "o")
        mode_label = MP2M_LABELS.get(mp2m_val, f"mp2m={mp2m_val}")
        rr_label = f"rr={rr_val}"
        combo_key = (mp2m_val, rr_val)
        label = f"{mode_label} / {rr_label}" if combo_key not in seen_combos else "_nolegend_"
        mp2b_x = math.log2(mp2b_val) if mp2b_val and mp2b_val > 0 else 0
        ax.scatter(mp2b_x, avg_tx, color=color, marker=marker, s=20, zorder=3, label=label)
        seen_combos[combo_key] = True

    ax.set_xlabel("NR_MAX_BATCHED_MIGRATION", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Average Tx / sec", fontsize=AXIS_LABEL_FONTSIZE)
    batch_ticks = [64, 128, 256, 512, 1024, 2048, 4096]
    ax.set_xticks([math.log2(v) for v in batch_ticks])
    ax.set_xticklabels([f"$2^{{{int(math.log2(v))}}}$" for v in batch_ticks])
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE, pad=TICK_PAD)
    ax.grid(True, alpha=0.3)
    if args.logy:
        ax.set_yscale("log")
    if not args.no_legend:
        ax.legend(fontsize=LEGEND_FONTSIZE, frameon=True)

    out_path = os.path.join(outdir, f"{out_name}_batch.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=EXPORT_PAD_INCHES)
    out_path = os.path.join(outdir, f"{out_name}_batch.pdf")
    fig.savefig(out_path, bbox_inches="tight", pad_inches=EXPORT_PAD_INCHES)
    print(f"Saved: {out_path}")
    plt.show()


def render_mig_flag(panel, fig, out_name, outdir):
    args = panel["args"]
    run_data = panel["run_data"]
    runs = panel["runs"]

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.set_box_aspect(1)

    group_points = defaultdict(list)
    for rd, run in zip(run_data, runs):
        if not rd["tx"]:
            continue
        try:
            rr = int(run.get("RNDREAD", 0) or 0)
        except (TypeError, ValueError):
            rr = 0
        try:
            mp2b = int(run.get("MOVE_PAGES2_MAX_BATCH_SIZE", 0) or 0)
        except (TypeError, ValueError):
            mp2b = 0
        try:
            ratio = float(run.get("DRAM_READ_RATIO", 0) or 0)
        except (TypeError, ValueError):
            ratio = 0.0
        avg_tx = sum(rd["tx"]) / len(rd["tx"])
        group_points[(rr, mp2b)].append((ratio, avg_tx))

    mp2b_values = sorted({mp2b for (_, mp2b) in group_points})
    rr_values = sorted({rr for (rr, _) in group_points})
    rr_colors = {0: "C0", 1: "red"}

    for (rr, mp2b), points in sorted(group_points.items()):
        points.sort()
        xs, ys = zip(*points)
        color = rr_colors.get(rr, f"C{rr}")
        marker = MARKERS[mp2b_values.index(mp2b) % len(MARKERS)]
        ax.scatter(xs, ys, color=color, marker=marker, s=20)

    ax.set_xlabel(r"$\mathtt{D_r, D_w, R_r, R_w}$", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Tx / sec", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_xlim(0, 1.05)
    ax.set_xticks([i / 10 for i in range(1, 11)])
    ax.set_xticklabels([f".{i}" if i < 10 else "1" for i in range(1, 11)])
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE, pad=TICK_PAD)
    ax.grid(True, alpha=0.3)
    if args.logy:
        ax.set_yscale("log")
    if not args.no_legend:
        mp2b_handles = [
            mlines.Line2D([], [], markerfacecolor="white", markeredgecolor="black",
                          marker=MARKERS[i % len(MARKERS)], linewidth=0, markersize=6, label=str(mp2b))
            for i, mp2b in enumerate(mp2b_values)
        ]
        rr_handles = [
            mpatches.Patch(color=rr_colors.get(rr, f"C{rr}"), label={0: "TPC-C", 1: "Read"}.get(rr, f"rndread={rr}"))
            for rr in rr_values
        ]
        ax.legend(handles=mp2b_handles + rr_handles, fontsize=LEGEND_FONTSIZE, frameon=True,
                  loc="upper center", bbox_to_anchor=(0.5, 1.28),
                  ncol=4, handletextpad=0.2, columnspacing=0.2)

    out_path = os.path.join(outdir, f"{out_name}_mig_ratio.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=EXPORT_PAD_INCHES)
    out_path = os.path.join(outdir, f"{out_name}_mig_ratio.pdf")
    fig.savefig(out_path, bbox_inches="tight", pad_inches=EXPORT_PAD_INCHES)
    print(f"Saved: {out_path}")
    plt.show()


def render_standard_panels(panels, out_name, outdir, show_legend=True):
    max_len = max((len(rd["ts"]) for panel in panels for rd in panel["run_data"]), default=1)
    markevery = max(1, max_len // 15)

    multi_panel = len(panels) > 1
    subplot_w = panels[0]["args"].subplot_w
    subplot_h = panels[0]["args"].subplot_h
    n_panels = len(panels)
    left_margin = 0.55
    right_margin = 0.15
    top_margin = 0.7 if show_legend else 0.2
    bottom_margin = 0.45 if multi_panel else 0.25
    gap_within_panel = 0.30
    gap_between_rows = 0.25 if multi_panel else 0.0
    fig_width = left_margin + right_margin + 2 * subplot_w + gap_within_panel
    fig_height = top_margin + bottom_margin + n_panels * subplot_h + max(0, n_panels - 1) * gap_between_rows
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(n_panels, 2, figure=fig,
                           left=left_margin / fig_width,
                           right=1.0 - right_margin / fig_width,
                           bottom=bottom_margin / fig_height,
                           top=1.0 - top_margin / fig_height,
                           wspace=gap_within_panel / subplot_w,
                           hspace=gap_between_rows / subplot_h if subplot_h else 0.0)

    tx_axes = []
    legend_map = {}
    for idx, panel in enumerate(panels):
        args = panel["args"]
        ax1 = fig.add_subplot(gs[idx, 0])
        ax2 = fig.add_subplot(gs[idx, 1])
        tx_axes.append(ax1)
        ax1.set_box_aspect(subplot_h / subplot_w)
        ax2.set_box_aspect(subplot_h / subplot_w)
        if multi_panel:
            workload_label = "TPC-C" if idx == 0 else "Random Read" if idx == 1 else f"Panel {idx + 1}"
            ax1.text(0.5, 0.03, workload_label,
                     transform=ax1.transAxes,
                     ha="center", va="bottom",
                     fontsize=LEGEND_FONTSIZE)
            ax2.text(0.5, 0.03, workload_label,
                     transform=ax2.transAxes,
                     ha="center", va="bottom",
                     fontsize=LEGEND_FONTSIZE)

        for run_idx, rd in enumerate(panel["run_data"]):
            handle = ax1.plot(rd["ts"], rd["tx"], label=rd["label"], linewidth=0.5,
                              color=rd["color"], marker=MARKERS[run_idx % len(MARKERS)],
                              markevery=markevery, markersize=2)[0]
            legend_map.setdefault(rd["label"], handle)

        ax1.set_xlabel("Time [seconds]", fontsize=AXIS_LABEL_FONTSIZE, labelpad=XLABEL_PAD)
        ax1.set_ylabel("Tx / sec", fontsize=AXIS_LABEL_FONTSIZE, labelpad=YLABEL_PAD)
        ax1.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE, pad=TICK_PAD)
        if args.logy:
            ax1.set_yscale("log")
        ax1.grid(True, alpha=0.3)

        if args.nm_label:
            for run_idx, rd in enumerate(panel["run_data"]):
                smoothed = rolling_mean(rd["total_movement"], args.smooth)
                ax2.plot(rd["ts"], smoothed, label=rd["label"], linewidth=0.5,
                         color=rd["color"], marker=MARKERS[run_idx % len(MARKERS)],
                         markevery=markevery, markersize=2)
            ax2.set_xlabel("Time [seconds]", fontsize=AXIS_LABEL_FONTSIZE, labelpad=XLABEL_PAD)
            ax2.set_ylabel("Page Migr / sec", fontsize=AXIS_LABEL_FONTSIZE, labelpad=YLABEL_PAD)
        else:
            for run_idx, rd in enumerate(panel["run_data"]):
                ax2.plot(rd["ts"], rd["total_io"], label=rd["label"], linewidth=0.5,
                         color=rd["color"], marker=MARKERS[run_idx % len(MARKERS)],
                         markevery=markevery, markersize=2)
            ax2.set_xlabel("Time [seconds]", fontsize=AXIS_LABEL_FONTSIZE, labelpad=XLABEL_PAD)
            ax2.set_ylabel("Total I/O (MB/s)", fontsize=AXIS_LABEL_FONTSIZE, labelpad=YLABEL_PAD)
            ax2.set_ylim(100)
        ax2.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE, pad=TICK_PAD)
        if args.logy:
            ax2.set_yscale("log")
        ax2.grid(True, alpha=0.3)

    if show_legend:
        labels = list(legend_map.keys())
        handles = [legend_map[label] for label in labels]
        fig.legend(handles, labels, loc="upper center",
                   bbox_to_anchor=(0.57, 0.8),
                   bbox_transform=fig.transFigure,
                   borderaxespad=0.0,
                   ncol=3, # max(1, len(labels)),
                   fontsize=LEGEND_FONTSIZE,
                   frameon=True,
                   columnspacing=0,
                   handletextpad=0)

    suffix = "_remo" if all(panel["args"].remo_label for panel in panels) else "_nm" if all(panel["args"].nm_label for panel in panels) else "_combined"
    out_path = os.path.join(outdir, f"{out_name}{suffix}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=EXPORT_PAD_INCHES)
    out_path = os.path.join(outdir, f"{out_name}{suffix}.pdf")
    fig.savefig(out_path, bbox_inches="tight", pad_inches=EXPORT_PAD_INCHES)
    print(f"Saved: {out_path}")
    plt.show()


def parse_panel_specs(specs):
    parser = build_parser(require_sweep=True, include_panel_spec=False)
    panels = []
    for spec in specs:
        panel_args = parser.parse_args(shlex.split(spec))
        panels.append(panel_args)
    return panels


def main():
    parser = build_parser(require_sweep=False)
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if args.panel_spec:
        panel_args_list = parse_panel_specs(args.panel_spec)
        panels = []
        for panel_args in panel_args_list:
            panel = build_run_data(panel_args)
            if panel is None:
                print(f"No runs match panel spec: {panel_args.sweep}")
                return
            print_summary(panel)
            panels.append(panel)
        out_name = args.out if args.out else "multi_panel"
        render_standard_panels(panels, out_name, args.outdir, show_legend=not args.no_legend)
        return

    if not args.sweep:
        parser.error("the following arguments are required: --sweep")

    panel = build_run_data(args)
    if panel is None:
        print("No runs match the given filters.")
        return
    print_summary(panel)

    sweep_label = "+".join(args.sweep)
    out_name = args.out if args.out else sweep_label

    if args.batch_label:
        render_batch(panel, plt.figure(), out_name, args.outdir)
        return

    if args.mig_flag:
        render_mig_flag(panel, plt.figure(), out_name, args.outdir)
        return

    render_standard_panels([panel], out_name, args.outdir, show_legend=not args.no_legend)


if __name__ == "__main__":
    main()
