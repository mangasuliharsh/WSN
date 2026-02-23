#!/usr/bin/env python3
"""
Phase 1 Results Comparator
Reads results/performance_A.csv, B.csv, C.csv and prints a comparison table.
Optionally plots if matplotlib is available.
"""

import csv, os, sys
from collections import defaultdict

SCENARIOS = ["A", "B", "C"]
LABELS = {
    "A": "Baseline AODV",
    "B": "Trust Only",
    "C": "Trust + Energy (Phase 1)"
}
RESULTS_DIR = "results"

def load_csv(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) if k not in ("scenario",) else v
                         for k, v in row.items()})
    return rows

def final_stats(rows):
    if not rows:
        return {}
    last = rows[-1]
    pdrs = [r["pdr"] for r in rows if r["pdr"] > 0]
    delays = [r["avg_delay_ms"] for r in rows if r["avg_delay_ms"] > 0]
    stddevs = [r.get("energy_stddev", 0) for r in rows]
    return {
        "pdr_mean":        sum(pdrs) / len(pdrs) * 100   if pdrs    else 0,
        "delay_mean":      sum(delays) / len(delays)     if delays  else 0,
        "energy_final":    last.get("avg_energy_J", 0),
        "alive_final":     int(last.get("alive_nodes", 0)),
        "iso_events":      int(last.get("isolation_events_cum", 0)),
        "energy_stddev":   sum(stddevs) / len(stddevs)   if stddevs else 0,
        "routing_metric":  last.get("avg_routing_metric", 0),
    }

# ── Load all scenarios ────────────────────────────────────────────────────────
data = {}
for s in SCENARIOS:
    path = os.path.join(RESULTS_DIR, f"performance_{s}.csv")
    data[s] = load_csv(path)

# ── Print comparison table ────────────────────────────────────────────────────
print()
print("╔══════════════════════════════════════════════════════════════════════╗")
print("║              Phase 1 — Scenario Comparison Table                    ║")
print("╠══════════════════════════════════════════════════════════════════════╣")
print(f"║  {'Metric':<28} {'A: Baseline':>13} {'B: Trust':>13} {'C: Full':>13}  ║")
print("╠══════════════════════════════════════════════════════════════════════╣")

metrics = [
    ("PDR (%)",               "pdr_mean",        ".2f"),
    ("Avg E2E Delay (ms)",    "delay_mean",       ".2f"),
    ("Final Avg Energy (J)",  "energy_final",     ".2f"),
    ("Alive Nodes",           "alive_final",      "d"),
    ("Isolation Events",      "iso_events",       "d"),
    ("Energy Std Dev (J)",    "energy_stddev",    ".3f"),
    ("Avg Routing Metric",    "routing_metric",   ".4f"),
]

stats = {s: final_stats(data[s]) for s in SCENARIOS}

for label, key, fmt in metrics:
    vals = []
    for s in SCENARIOS:
        v = stats[s].get(key, 0)
        vals.append(format(v, fmt) if data[s] else "  N/A  ")
    print(f"║  {label:<28} {vals[0]:>13} {vals[1]:>13} {vals[2]:>13}  ║")

print("╚══════════════════════════════════════════════════════════════════════╝")
print()

# ── Optional: generate plot ────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Phase 1 — Scenario Comparison (A: Baseline | B: Trust | C: Trust+Energy)",
                 fontsize=13, fontweight="bold")

    plot_fields = [
        ("pdr",                "PDR (fraction)",            (0, 1.05)),
        ("avg_delay_ms",       "Avg E2E Delay (ms)",        None),
        ("avg_energy_J",       "Avg Energy (J)",            (0, 160)),
        ("alive_nodes",        "Alive Nodes",               (0, 55)),
        ("energy_stddev",      "Energy Std Dev (J)",        None),
        ("avg_routing_metric", "Avg Routing Metric",        (0, 1.05)),
    ]

    colors = {"A": "#e74c3c", "B": "#3498db", "C": "#2ecc71"}

    for ax, (field, ylabel, ylim) in zip(axes.flat, plot_fields):
        for s in SCENARIOS:
            rows = data[s]
            if not rows or field not in rows[0]: continue
            xs = [r["time_s"] for r in rows]
            ys = [r[field] for r in rows]
            ax.plot(xs, ys, label=LABELS[s], color=colors[s], linewidth=1.8)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        if ylim: ax.set_ylim(ylim)

        # Shade attack windows
        for base in range(0, 601, 120):
            for start, end in [(base+20, base+40), (base+60, base+80), (base+90, base+110)]:
                if start < 600:
                    ax.axvspan(start, min(end, 600), alpha=0.08, color="red")

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "phase1_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[+] Plot saved to {out}")

except ImportError:
    print("[!] matplotlib not available — skipping plot (install with: pip install matplotlib)")
except Exception as e:
    print(f"[!] Plot error: {e}")
