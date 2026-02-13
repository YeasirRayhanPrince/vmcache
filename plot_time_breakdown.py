#!/usr/bin/env python3
import argparse
import json
import re
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt

# Migration method label ordering and mapping
# Key: (NUMA_MIGRATE_METHOD, MOVE_PAGES2_MODE)  (-1 = any/wildcard)
METHOD_ORDER = [
    (0, -1),
    (1, -1),
    (2, -1),
    (3, 0),
    (3, 1),
    (3, 2),
]

METHOD_LABELS = {
    (0, -1): "mbind",
    (1, -1): "mbind(n)",
    (2, -1): "move_pages",
    (3, 0): "move_pages2",
    (3, 1): "move_pages2\n(mode 1)",
    (3, 2): "move_pages2\n(mode 2)",
}

IO_RE  = re.compile(r'Avg IO sec/s:\s+([\d.]+)')
MIG_RE = re.compile(r'Avg Migration sec/s:\s+([\d.]+)')
CMP_RE = re.compile(r'Avg Compute sec/s:\s+([\d.]+)')


def method_key(nm, mp2):
    if nm in (0, 1, 2):
        return (nm, -1)
    return (nm, mp2)


def parse_log(path):
    """Return (io, migration, compute) floats or None if not found."""
    try:
        text = open(path).read()
    except OSError:
        return None
    m_io  = IO_RE.search(text)
    m_mig = MIG_RE.search(text)
    m_cmp = CMP_RE.search(text)
    if not (m_io and m_mig and m_cmp):
        return None
    return float(m_io.group(1)), float(m_mig.group(1)), float(m_cmp.group(1))


def load_runs(sweep_ids, results_dir="bench_results"):
    runs = []
    for sweep_id in sweep_ids:
        summary_path = os.path.join(results_dir, f"{sweep_id}_summary.jsonl")
        with open(summary_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    runs.append(json.loads(line))
    return runs


def build_accum(runs):
    """Accumulate (io, mig, cmp) lists keyed by method_key."""
    accum = defaultdict(list)
    first_entry = None
    for e in runs:
        nm  = e.get("NUMA_MIGRATE_METHOD", 0)
        mp2 = e.get("MOVE_PAGES2_MODE", 0)
        logfile = e.get("logfile", "")
        vals = parse_log(logfile)
        if vals is None:
            print(f"  [warn] no time-breakdown data in {logfile}", file=sys.stderr)
            continue
        key = method_key(nm, mp2)
        accum[key].append(vals)
        if first_entry is None:
            first_entry = e
    return accum, first_entry


def avg_triple(triples):
    n = len(triples)
    return (
        sum(t[0] for t in triples) / n,
        sum(t[1] for t in triples) / n,
        sum(t[2] for t in triples) / n,
    )


def make_subtitle(entry):
    if entry is None:
        return ""
    parts = []
    for k, label in [
        ("PHYSGB",   "PHYSGB={}"),
        ("REMOTEGB", "REMOTEGB={}"),
        ("THREADS",  "THREADS={}"),
        ("PROMOTE_BATCH", "PB={}"),
        ("EVICT_BATCH",   "EB={}"),
        ("RNDREAD",  "rnd={}"),
    ]:
        if k in entry:
            parts.append(label.format(entry[k]))
    return "  ".join(parts)


COLORS = {
    "IO":        "#3A7DC9",   # strong blue
    "Migration": "#E8622A",   # vivid orange-red
    "Compute":   "#2BA84A",   # rich green
}


def plot(accum, first_entry, out_stem, outdir, workload_title=None):
    keys = [k for k in METHOD_ORDER if k in accum]
    if not keys:
        print("No data to plot.", file=sys.stderr)
        sys.exit(1)

    labels   = [METHOD_LABELS[k] for k in keys]
    io_abs, mig_abs, cmp_abs = [], [], []
    for k in keys:
        io, mig, cmp = avg_triple(accum[k])
        io_abs.append(io)
        mig_abs.append(mig)
        cmp_abs.append(cmp)

    # Normalise to 100 %
    io_pct, mig_pct, cmp_pct = [], [], []
    for io, mig, cmp in zip(io_abs, mig_abs, cmp_abs):
        total = io + mig + cmp or 1.0
        io_pct.append(100 * io / total)
        mig_pct.append(100 * mig / total)
        cmp_pct.append(100 * cmp / total)

    x = list(range(len(keys)))
    bar_w = 0.78
    fig, ax = plt.subplots(figsize=(4,3))

    ax.bar(x, io_pct,  width=bar_w, label="IO",        color=COLORS["IO"])
    mig_bottoms = io_pct
    ax.bar(x, mig_pct, width=bar_w, bottom=mig_bottoms, label="Migration", color=COLORS["Migration"])
    cmp_bottoms = [i + m for i, m in zip(io_pct, mig_pct)]
    ax.bar(x, cmp_pct, width=bar_w, bottom=cmp_bottoms, label="Compute",   color=COLORS["Compute"])

    # Percentage annotations centred in each segment — always shown;
    # for very thin segments use an outside annotation with an arrow.
    MIN_INSIDE = 5  # % height threshold for inside label
    text_in  = dict(ha="center", va="center", fontsize=11, fontweight="bold", color="white")
    text_out = dict(ha="center", va="bottom",  fontsize=11,  fontweight="bold", color="white")

    def annotate_segment(i, bottom, height, label):
        if height <= 0:
            return
        mid = bottom + height / 2
        if height >= MIN_INSIDE:
            ax.text(i, mid, label, **text_in)
        else:
            # place text above the bar with a small offset
            ax.annotate(label, xy=(i, bottom + height), xytext=(i, bottom + height + 2),
                        ha="center", va="bottom", fontsize=11, fontweight="bold", color="white",
                        arrowprops=dict(arrowstyle="-", color="gray", lw=0.8))

    for i in range(len(keys)):
        annotate_segment(i, 0,               io_pct[i],  f"{io_pct[i]:.1f}%")
        annotate_segment(i, mig_bottoms[i],  mig_pct[i], f"{mig_pct[i]:.1f}%")
        annotate_segment(i, cmp_bottoms[i],  cmp_pct[i], f"{cmp_pct[i]:.1f}%")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Time breakdown (%)", fontsize=11)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_xlim(-0.5, len(keys) - 0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2),
              ncols=3, fontsize=10, framealpha=0.85)
    # if workload_title:
    #     fig.text(0.5, -0.01, workload_title, ha="center", va="top",
    #              fontsize=12)

    fig.tight_layout()

    os.makedirs(outdir, exist_ok=True)
    for ext in ("png", "pdf"):
        out_path = os.path.join(outdir, f"{out_stem}_time_breakdown.{ext}")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")
    try:
        plt.show()
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(
        description="Stacked bar: IO / Migration / Compute by NUMA migration method"
    )
    parser.add_argument(
        "--sweep", required=True, nargs="+",
        help="Sweep ID timestamp(s) (e.g., 20260213_130458)"
    )
    parser.add_argument(
        "--outdir", default="bench_plots",
        help="Output directory for figures (default: bench_plots)"
    )
    parser.add_argument(
        "--out",
        help="Output filename stem (default: sweep IDs joined with '+')"
    )
    parser.add_argument("--nm",   nargs="+", type=int, help="Filter NUMA_MIGRATE_METHOD values")
    parser.add_argument("--mp2m", nargs="+", type=int, help="Filter MOVE_PAGES2_MODE values")
    parser.add_argument("--eb",   nargs="+", type=int, help="Filter EVICT_BATCH values")
    parser.add_argument("--pb",   nargs="+", type=int, help="Filter PROMOTE_BATCH values")
    parser.add_argument("--t",    nargs="+", type=int, help="Filter THREADS values")
    parser.add_argument("--remo",    nargs="+", type=int, help="Filter REMOTEGB values")
    parser.add_argument("--rndread", nargs="+", type=int, help="Filter RNDREAD values (0=TPC-C, 1=Random Read)")
    args = parser.parse_args()

    filters = {
        "NUMA_MIGRATE_METHOD": args.nm,
        "MOVE_PAGES2_MODE":    args.mp2m,
        "EVICT_BATCH":         args.eb,
        "PROMOTE_BATCH":       args.pb,
        "THREADS":             args.t,
        "REMOTEGB":            args.remo,
        "RNDREAD":             args.rndread,
    }
    filters = {k: v for k, v in filters.items() if v is not None}

    print(f"Loading {len(args.sweep)} sweep(s)…")
    runs = load_runs(args.sweep)
    print(f"  {len(runs)} entries total")

    if filters:
        runs = [r for r in runs if all(
            k not in r or str(r[k]) in [str(x) for x in v]
            for k, v in filters.items()
        )]
        print(f"  {len(runs)} entries after filtering")
        if not runs:
            print("No runs match the given filters.", file=sys.stderr)
            sys.exit(1)

    accum, first_entry = build_accum(runs)

    # Determine workload title from --rndread flag or data
    if args.rndread is not None:
        rnd = args.rndread[0]
    elif first_entry is not None:
        rnd = first_entry.get("RNDREAD")
    else:
        rnd = None
    if rnd is not None:
        workload_title = "Random Read" if int(rnd) != 0 else "TPC-C"
    else:
        workload_title = None

    sweep_label = "+".join(args.sweep)
    out_stem = args.out if args.out else sweep_label
    plot(accum, first_entry, out_stem, args.outdir, workload_title=workload_title)


if __name__ == "__main__":
    main()
