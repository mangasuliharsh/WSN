#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  WSN Phase 2 — Routing Evidence Visualiser  [FIXED]                          ║
║  Generates publication-quality figures from the 4 evidence CSV files         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  BUGS FIXED vs previous version:                                              ║
║   • Fig2: broken out_path expression (os.path.join of list) — replaced       ║
║   • All figs: _fig_path() now uses the --results CLI arg consistently.       ║
║     Previously Fig1 wrote to results_dir directly while Fig2-7 used a        ║
║     global that was never updated from the default "results".                 ║
║   • Fig5: phase windows corrected — first attack is t=20-40s not t=10-20s   ║
║   • Fig5: bins crash when min==max hops — safe fallback added                ║
║   • Fig6: axvline was at x=0.5 (wrong) — sink is col 0, now at x=0          ║
║   • Fig7: twinx legend duplication fixed; empty route_changes handled        ║
║   • Attack shading: changed from t_end<=t[-1] to t_start<t_max+30 so the   ║
║     last window is never silently skipped                                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

Usage:
    python3 visualise_routing_evidence.py [--scenario D] [--results ./results]

Requires: numpy, pandas, matplotlib, networkx, scipy
    pip install numpy pandas matplotlib networkx scipy
"""

import os
import glob
import argparse
import numpy   as np
import pandas  as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot   as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches  as mpatches
import networkx            as nx
from   scipy.ndimage       import uniform_filter1d

# ── Runtime config — set in main(), used via _fig_path() everywhere ───────────
_RESULTS_DIR = "results"

def _fig_path(fname):
    """Absolute path to output figure; creates figures/ subdir if needed."""
    d = os.path.join(_RESULTS_DIR, "figures")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, fname)

# ── Simulation constants (must match C++ source) ──────────────────────────────
N_NODES     = 50
GRID_W      = 10
DX, DY      = 50.0, 50.0

# Attack windows derived from ScheduleCycle(base) logic:
#   base=0:   W1=20-40, W2=60-80, W3=90-110
#   base=120: W1=140-160, W2=180-200, W3=210-230  ...etc.
ATTACK_WINDOWS = [
    (20,  40),  (60,  80),  (90,  110),
    (140, 160), (180, 200), (210, 230),
    (260, 280), (300, 320), (330, 350),
    (380, 400), (420, 440), (450, 470),
    (500, 520), (540, 560), (570, 590),
]

PALETTE = {
    "normal":            "#27AE60",
    "malicious":         "#E67E22",
    "isolated":          "#2C3E50",
    "soft-avoid":        "#17A589",
    "dead":              "#95A5A6",
    "sink":              "#C0392B",
    "route-established": "#1ABC9C",
    "improve":           "#27AE60",
    "degrade":           "#E74C3C",
    "nh-swap":           "#3498DB",
    "cost-rise":         "#E74C3C",
    "cost-fall":         "#27AE60",
    "cost-drift-up":     "#E67E22",
    "cost-drift-down":   "#F1C40F",
    "cost-drift":        "#BDC3C7",
    "attack-activated":  "#E67E22",
    "attack-ended":      "#27AE60",
    "isolated-loss":     "#2C3E50",
    "malicious-loss":    "#8E44AD",
}

# ── Node positions matching ns-3 GridPositionAllocator ───────────────────────
def node_pos():
    pos = {}
    for i in range(N_NODES):
        row = i // GRID_W
        col = i  % GRID_W
        pos[i] = (20.0 + col * DX, 20.0 + row * DY)
    return pos

# ── Attack window shading ─────────────────────────────────────────────────────
def shade_attacks(ax, t_max, alpha=0.10):
    """Shade every attack window that overlaps the plotted time range."""
    for t_s, t_e in ATTACK_WINDOWS:
        if t_s < t_max + 30:
            ax.axvspan(t_s, min(t_e, t_max + 5), alpha=alpha, color='red', zorder=0)

# ── Data loaders ──────────────────────────────────────────────────────────────
def load_routing_matrix(results_dir, ts):
    path = os.path.join(results_dir, f"routing_matrix_{int(ts)}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col=0)
    return df.values.astype(int)

def available_matrices(results_dir):
    files = glob.glob(os.path.join(results_dir, "routing_matrix_*.csv"))
    ts_list = []
    for f in files:
        try:
            ts = int(os.path.basename(f)
                       .replace("routing_matrix_", "")
                       .replace(".csv", ""))
            ts_list.append(ts)
        except ValueError:
            pass
    return sorted(ts_list)

def load_path_traces(results_dir):
    path = os.path.join(results_dir, "path_traces.csv")
    if not os.path.exists(path):
        print(f"  WARN: {path} not found")
        return pd.DataFrame()
    df = pd.read_csv(path)
    if 'hop_count' in df.columns:
        df['hop_count'] = pd.to_numeric(df['hop_count'], errors='coerce')
    if 'path_cost' in df.columns:
        df['path_cost'] = pd.to_numeric(df['path_cost'], errors='coerce')
    return df

def load_route_changes(results_dir):
    path = os.path.join(results_dir, "route_changes.csv")
    if not os.path.exists(path):
        print(f"  WARN: {path} not found")
        return pd.DataFrame()
    return pd.read_csv(path)

def load_performance(results_dir, scenario):
    path = os.path.join(results_dir, f"performance_{scenario}.csv")
    if not os.path.exists(path):
        print(f"  WARN: {path} not found")
        return pd.DataFrame()
    return pd.read_csv(path)


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Routing graph snapshots (NetworkX)
# ═════════════════════════════════════════════════════════════════════════════
def fig1_routing_graph_snapshots(results_dir, path_traces):
    all_ts = available_matrices(results_dir)
    if not all_ts:
        print("  [Fig1] No routing matrix files — skipping"); return

    pre    = min(all_ts, key=lambda t: abs(t - 15))
    during = min(all_ts, key=lambda t: abs(t - 25))
    after  = min(all_ts, key=lambda t: abs(t - 45))
    chosen = [pre, during, after]
    labels = ["Pre-attack", "During attack", "Post-isolation"]

    pos = node_pos()
    fig, axes = plt.subplots(1, 3, figsize=(22, 8))
    fig.suptitle("Routing Topology Snapshots — Next-Hop Graph Evolution",
                 fontsize=14, fontweight='bold')

    # Build state lookup from nearest path-trace timestamp
    state_at = {}
    if not path_traces.empty:
        avail = path_traces['time_s'].unique()
        for ts in chosen:
            nearest = avail[np.argmin(np.abs(avail - ts))]
            sub = path_traces[path_traces['time_s'] == nearest]
            state_at[ts] = dict(zip(sub['node'], sub['state']))

    for ax, ts, label in zip(axes, chosen, labels):
        mat = load_routing_matrix(results_dir, ts)
        title = f"t = {ts}s ({label})"

        if mat is None:
            ax.text(0.5, 0.5, f"Matrix not found\n(t={ts}s)",
                    ha='center', va='center', transform=ax.transAxes, fontsize=11)
            ax.set_title(title); ax.axis('off'); continue

        G = nx.DiGraph()
        G.add_nodes_from(range(N_NODES))
        for i in range(N_NODES):
            for j in range(N_NODES):
                if mat[i, j] == 1:
                    G.add_edge(i, j)

        colours = []
        sizes   = []
        for n in range(N_NODES):
            if n == 0:
                colours.append(PALETTE["sink"]); sizes.append(500)
            else:
                state = state_at.get(ts, {}).get(n, "normal")
                colours.append(PALETTE.get(state, PALETTE["normal"]))
                sizes.append(180)

        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colours,
                               node_size=sizes, alpha=0.92)
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#2980B9",
                               alpha=0.55, arrows=True, arrowsize=12,
                               connectionstyle="arc3,rad=0.08", width=1.5)
        nx.draw_networkx_labels(G, pos, ax=ax,
                                labels={n: str(n) for n in range(N_NODES)},
                                font_size=5, font_color='white', font_weight='bold')

        if any(s <= ts <= e for s, e in ATTACK_WINDOWS):
            ax.text(0.02, 0.98, "⚠ ATTACK ACTIVE", transform=ax.transAxes,
                    color='red', fontsize=9, fontweight='bold', va='top')

        ax.set_title(f"{title}\n({G.number_of_edges()} active routes)", fontsize=10)
        ax.set_aspect('equal'); ax.axis('off')

    legend_elems = [
        mpatches.Patch(color=PALETTE["sink"],       label="Sink (N0)"),
        mpatches.Patch(color=PALETTE["normal"],     label="Normal"),
        mpatches.Patch(color=PALETTE["malicious"],  label="Malicious"),
        mpatches.Patch(color=PALETTE["isolated"],   label="ML-Isolated"),
        mpatches.Patch(color=PALETTE["soft-avoid"], label="Soft-avoided"),
        mpatches.Patch(color=PALETTE["dead"],       label="Dead"),
    ]
    fig.legend(handles=legend_elems, loc='lower center', ncol=6,
               bbox_to_anchor=(0.5, -0.03), fontsize=9)

    out = _fig_path("Fig1_routing_graph_snapshots.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Fig1] Saved: {out}")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Hop count evolution (FIXED: out_path was broken)
# ═════════════════════════════════════════════════════════════════════════════
def fig2_hop_count_evolution(perf_df):
    if perf_df.empty or 'avg_hop_count' not in perf_df.columns:
        print("  [Fig2] No hop count data — skipping"); return

    fig, ax = plt.subplots(figsize=(14, 5))

    t     = perf_df['time_s'].values
    avg_h = pd.to_numeric(perf_df['avg_hop_count'], errors='coerce').fillna(0).values
    std_h = pd.to_numeric(perf_df['hop_stddev'],    errors='coerce').fillna(0).values
    min_h = pd.to_numeric(perf_df['min_hops'],      errors='coerce').fillna(0).values
    max_h = pd.to_numeric(perf_df['max_hops'],      errors='coerce').fillna(0).values

    smooth_k = min(5, max(1, len(t) // 10))
    avg_sm   = uniform_filter1d(avg_h, size=smooth_k)
    t_max    = float(t[-1]) if len(t) else 600.0

    shade_attacks(ax, t_max)
    ax.fill_between(t, min_h, max_h, alpha=0.12, color='#2980B9', label='Min–Max range')
    ax.fill_between(t, np.maximum(0, avg_h - std_h), avg_h + std_h,
                    alpha=0.25, color='#2980B9', label='Avg ± Std Dev')
    ax.plot(t, avg_sm, color='#2980B9', lw=2.5, label='Avg hops (smoothed)')
    ax.plot(t, avg_h,  color='#2980B9', lw=0.8, alpha=0.35)

    ax.set_xlabel("Simulation Time (s)", fontsize=12)
    ax.set_ylabel("Hop Count to Sink",   fontsize=12)
    ax.set_title("Hop Count Evolution — Routing Adaptation Under Attack",
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, t_max)
    ax.set_ylim(bottom=0)

    out = _fig_path("Fig2_hop_count_evolution.png")   # FIX: was broken expression
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Fig2] Saved: {out}")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Route change event timeline
# ═════════════════════════════════════════════════════════════════════════════
def fig3_route_change_timeline(route_changes):
    if route_changes.empty:
        print("  [Fig3] No route change data — skipping"); return

    t_max   = float(route_changes['time_s'].max())
    reasons = sorted(route_changes['reason'].unique())

    fig, (ax_sc, ax_bar) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                         gridspec_kw={'height_ratios': [3, 1]})
    shade_attacks(ax_sc,  t_max)
    shade_attacks(ax_bar, t_max)

    legend_elems = []
    for reason in reasons:
        sub = route_changes[route_changes['reason'] == reason]
        col = PALETTE.get(reason, "#95A5A6")
        ax_sc.scatter(sub['time_s'], sub['node'],
                      c=col, s=22, alpha=0.75, zorder=3)
        legend_elems.append(mpatches.Patch(color=col, label=reason))

    ax_sc.set_ylabel("Node ID", fontsize=11)
    ax_sc.set_title("Route Change Events — Node × Time (coloured by reason)",
                    fontsize=12, fontweight='bold')
    ax_sc.legend(handles=legend_elems, loc='upper right', fontsize=7, ncol=3,
                 framealpha=0.85)
    ax_sc.set_ylim(-1, N_NODES)
    ax_sc.set_yticks(range(0, N_NODES, 5))
    ax_sc.grid(True, alpha=0.2)

    bin_w = max(10, int(t_max / 60))
    bins  = np.arange(0, t_max + bin_w + 1, bin_w)
    cnt, edges = np.histogram(route_changes['time_s'], bins=bins)
    ax_bar.bar(edges[:-1], cnt, width=bin_w * 0.9, align='edge',
               color='#2980B9', alpha=0.75)
    ax_bar.set_xlabel("Simulation Time (s)", fontsize=11)
    ax_bar.set_ylabel(f"Events\n/{bin_w}s", fontsize=9)
    ax_bar.set_title("Route Change Rate", fontsize=10)
    ax_bar.grid(True, alpha=0.2)

    out = _fig_path("Fig3_route_change_timeline.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Fig3] Saved: {out}")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Path cost heatmap (node × time)
# ═════════════════════════════════════════════════════════════════════════════
def fig4_path_cost_heatmap(path_traces):
    if path_traces.empty or 'path_cost' not in path_traces.columns:
        print("  [Fig4] No path trace data — skipping"); return

    pivot = path_traces[path_traces['node'] > 0].pivot_table(
        index='node', columns='time_s', values='path_cost', aggfunc='mean')

    times_idx = sorted(path_traces['time_s'].unique())
    pivot = pivot.reindex(index=range(1, N_NODES), columns=times_idx)

    fig, ax = plt.subplots(figsize=(16, 7))
    im = ax.imshow(pivot.values, aspect='auto',
                   cmap='RdYlGn_r', vmin=0, vmax=5,
                   interpolation='nearest',
                   extent=[times_idx[0], times_idx[-1],
                            N_NODES - 0.5, 0.5])
    plt.colorbar(im, ax=ax, label='Path Cost  Σ(1 − nodeScore)',
                 fraction=0.02, pad=0.02)

    t_max = times_idx[-1]
    for t_s, t_e in ATTACK_WINDOWS:
        if t_s < t_max:
            ax.axvline(t_s, color='red',  lw=0.9, alpha=0.55, linestyle='--')
            ax.axvline(min(t_e, t_max), color='navy', lw=0.9, alpha=0.35, linestyle=':')

    ax.set_xlabel("Simulation Time (s)", fontsize=12)
    ax.set_ylabel("Node ID",             fontsize=12)
    ax.set_title("Path Cost Heatmap — Per-Node, Per-Interval\n"
                 "(Green=healthy path | Red=detour or no-route | "
                 "Red dashed=attack start | Navy dotted=attack end)",
                 fontsize=11, fontweight='bold')
    ax.set_yticks(range(1, N_NODES, 5))

    out = _fig_path("Fig4_path_cost_heatmap.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Fig4] Saved: {out}")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Hop count distribution: pre / during / post attack
# FIXED: phase windows now match simulation schedule; safe bin fallback added
# ═════════════════════════════════════════════════════════════════════════════
def fig5_hop_distribution(path_traces):
    if path_traces.empty or 'hop_count' not in path_traces.columns:
        print("  [Fig5] No hop count in path traces — skipping"); return

    # Phase definitions — corrected to match ScheduleCycle:
    # First attack activates at base+20 = 20s
    phases = [
        ("Pre-attack\n(t = 10–19s)",    10,  19,  "#27AE60"),
        ("During attack\n(t = 20–40s)", 20,  40,  "#E74C3C"),
        ("Post-isolation\n(t = 40–60s)",40,  60,  "#2980B9"),
    ]

    def get_hops(t_lo, t_hi):
        sub = path_traces[(path_traces['time_s'] >= t_lo) &
                          (path_traces['time_s'] <= t_hi)]
        h = sub['hop_count'].dropna()
        return h[(h > 0) & (h < 20)].values

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    fig.suptitle("Hop Count Distribution — Phase Comparison\n"
                 "(Detour paths force higher hop counts during attack windows)",
                 fontsize=12, fontweight='bold')

    for ax, (title, t_lo, t_hi, colour) in zip(axes, phases):
        hops = get_hops(t_lo, t_hi)

        if len(hops) == 0:
            ax.text(0.5, 0.5, "No data\nfor this phase",
                    ha='center', va='center', transform=ax.transAxes, fontsize=11)
            ax.set_title(title, fontsize=10); continue

        lo, hi = int(min(hops)), int(max(hops))
        # FIX: safe bins when all hops are identical (lo==hi crashes range())
        bins = [lo - 0.5, lo + 0.5] if lo == hi else [x - 0.5 for x in range(lo, hi + 2)]

        ax.hist(hops, bins=bins, color=colour, alpha=0.78,
                edgecolor='white', rwidth=0.85)
        mean_h = np.mean(hops)
        ax.axvline(mean_h, color='black', lw=2.0, linestyle='--',
                   label=f'Mean = {mean_h:.1f}')
        ax.set_xlabel("Hop Count to Sink",     fontsize=11)
        ax.set_ylabel("Node-interval count",   fontsize=10)
        ax.set_title(f"{title}\nn={len(hops)},  mean={mean_h:.1f}", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

    out = _fig_path("Fig5_hop_distribution.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Fig5] Saved: {out}")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Routing adjacency matrix heatmap
# FIXED: axvline now at x=0 (sink column), not x=0.5
# ═════════════════════════════════════════════════════════════════════════════
def fig6_routing_adjacency_heatmap(results_dir):
    all_ts = available_matrices(results_dir)
    if not all_ts:
        print("  [Fig6] No routing matrix files — skipping"); return

    pre    = min(all_ts, key=lambda t: abs(t - 15))
    during = min(all_ts, key=lambda t: abs(t - 25))
    after  = min(all_ts, key=lambda t: abs(t - 45))
    chosen = [pre, during, after]
    labels = ["Pre-attack", "During attack", "Post-isolation"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Routing Adjacency Matrix — 50×50 Next-Hop Snapshots\n"
                 "(Blue cell: node i's next-hop is node j | "
                 "Red column = Sink N0 | Row=source, Col=next-hop)",
                 fontsize=11, fontweight='bold')

    for ax, ts, label in zip(axes, chosen, labels):
        mat = load_routing_matrix(results_dir, ts)
        title = f"t={ts}s ({label})"

        if mat is None:
            ax.text(0.5, 0.5, "Not found", ha='center', va='center',
                    transform=ax.transAxes); ax.set_title(title); continue

        ax.imshow(mat, cmap='Blues', vmin=0, vmax=1, aspect='auto',
                  interpolation='nearest',
                  extent=[-0.5, N_NODES - 0.5, N_NODES - 0.5, -0.5])

        ax.set_xlabel("Next-Hop Node ID", fontsize=10)
        ax.set_ylabel("Source Node ID",   fontsize=10)
        ax.set_title(f"{title}\n{int(mat.sum())} active routes", fontsize=10)
        ax.set_xticks(range(0, N_NODES, 5))
        ax.set_yticks(range(0, N_NODES, 5))
        ax.tick_params(labelsize=7)

        # FIX: sink is column 0; highlight at x=0 not x=0.5
        ax.axvline(0, color='red', lw=2.0, alpha=0.65)
        ax.text(0, N_NODES * 0.97, "Sink", color='red',
                fontsize=7, ha='center', fontweight='bold')

    out = _fig_path("Fig6_routing_adjacency_heatmap.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Fig6] Saved: {out}")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 7 — 4-panel summary (FIXED: twinx legends, empty data guards)
# ═════════════════════════════════════════════════════════════════════════════
def fig7_summary_panel(perf_df, route_changes, path_traces):
    if perf_df.empty:
        print("  [Fig7] No performance data — skipping"); return

    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.32)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    t     = perf_df['time_s'].values
    t_max = float(t[-1]) if len(t) else 600.0

    # ── Panel 1: Hop count ────────────────────────────────────────────────────
    if 'avg_hop_count' in perf_df.columns:
        h  = pd.to_numeric(perf_df['avg_hop_count'], errors='coerce').fillna(0).values
        hs = pd.to_numeric(perf_df['hop_stddev'],    errors='coerce').fillna(0).values
        shade_attacks(ax1, t_max)
        ax1.fill_between(t, np.maximum(0, h - hs), h + hs, alpha=0.25, color='#2980B9')
        ax1.plot(t, uniform_filter1d(h, size=min(5, len(t))),
                 color='#2980B9', lw=2.5, label='Avg hops')
        ax1.set_xlabel("Time (s)"); ax1.set_ylabel("Hop Count")
        ax1.set_title("Avg Hop Count to Sink\n(rises on detour during attack)",
                      fontweight='bold')
        ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3); ax1.set_ylim(bottom=0)
    else:
        ax1.text(0.5, 0.5, "No hop count data", ha='center', va='center',
                 transform=ax1.transAxes)

    # ── Panel 2: Route change rate ────────────────────────────────────────────
    if not route_changes.empty:
        bin_w = max(10, int(t_max / 60))
        bins  = np.arange(0, t_max + bin_w + 1, bin_w)
        cnt, edges = np.histogram(route_changes['time_s'], bins=bins)
        shade_attacks(ax2, t_max)
        ax2.bar(edges[:-1], cnt, width=bin_w * 0.9, align='edge',
                color='#E74C3C', alpha=0.75)
        ax2.set_xlabel("Time (s)"); ax2.set_ylabel(f"Events / {bin_w}s")
        ax2.set_title("Route Change Rate\n(spikes = attack/isolation events)",
                      fontweight='bold')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5,
                 "No route change data\n\nCheck route_changes.csv\nExists and non-empty",
                 ha='center', va='center', transform=ax2.transAxes,
                 color='red', fontsize=10)
        ax2.set_title("Route Change Rate", fontweight='bold')

    # ── Panel 3: Path cost (left) + stability (right) ─────────────────────────
    if 'avg_path_cost' in perf_df.columns:
        pc   = pd.to_numeric(perf_df['avg_path_cost'],  errors='coerce').fillna(0).values
        stab = pd.to_numeric(perf_df['path_stability'], errors='coerce').fillna(0).values * 100
        ax3r = ax3.twinx()
        shade_attacks(ax3, t_max, alpha=0.07)
        l1, = ax3.plot(t,  pc,   color='#E74C3C', lw=2.5, label='Avg path cost')
        l2, = ax3r.plot(t, stab, color='#27AE60', lw=2.0, linestyle='--',
                        label='Path stability %')
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Avg Path Cost",      color='#E74C3C', fontsize=11)
        ax3r.set_ylabel("Path Stability (%)", color='#27AE60', fontsize=11)
        ax3.tick_params(axis='y', labelcolor='#E74C3C')
        ax3r.tick_params(axis='y', labelcolor='#27AE60')
        ax3r.set_ylim(0, 105)
        ax3.set_title("Path Cost & Stability\n(cost rises, stability drops under attack)",
                      fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend(handles=[l1, l2], fontsize=9, loc='upper left')
    else:
        ax3.text(0.5, 0.5, "No path cost data", ha='center', va='center',
                 transform=ax3.transAxes)

    # ── Panel 4: PDR (left) + cumulative isolations (right) ───────────────────
    if 'pdr' in perf_df.columns:
        pdr = pd.to_numeric(perf_df['pdr'], errors='coerce').fillna(0).values * 100
        iso_series = perf_df.get('isolation_events_cum', pd.Series([0]*len(t), dtype=float))
        iso = pd.to_numeric(iso_series, errors='coerce').fillna(0).values
        ax4r = ax4.twinx()
        shade_attacks(ax4, t_max, alpha=0.07)
        l1, = ax4.plot(t,  pdr, color='#2980B9', lw=2.5, label='PDR %')
        l2, = ax4r.plot(t, iso, color='#E67E22', lw=2.0, linestyle='--',
                        label='Cumul. isolations')
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("PDR (%)",                   color='#2980B9', fontsize=11)
        ax4r.set_ylabel("Cumul. Isolation Events",  color='#E67E22', fontsize=11)
        ax4.tick_params(axis='y', labelcolor='#2980B9')
        ax4r.tick_params(axis='y', labelcolor='#E67E22')
        ax4.set_ylim(0, 105)
        ax4.set_title("PDR & Isolation Events\n(PDR sustained via ML isolation)",
                      fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend(handles=[l1, l2], fontsize=9, loc='lower left')
    else:
        ax4.text(0.5, 0.5, "No PDR data", ha='center', va='center',
                 transform=ax4.transAxes)

    fig.suptitle("WSN Phase 2 — Routing Evidence Summary Panel\n"
                 "(Red shading = attack windows | Scenario D: ML + Energy + Route Opt)",
                 fontsize=13, fontweight='bold')

    out = _fig_path("Fig7_summary_panel.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Fig7] Saved: {out}")


# ═════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ═════════════════════════════════════════════════════════════════════════════
def print_summary_stats(perf_df, path_traces, route_changes):
    print("\n" + "═" * 62)
    print("  ROUTING EVIDENCE SUMMARY STATISTICS")
    print("═" * 62)

    if not perf_df.empty:
        def col(name):
            return pd.to_numeric(perf_df.get(name, pd.Series()), errors='coerce')
        pdr_c = col('pdr');  dly_c = col('avg_delay_ms')
        h_c   = col('avg_hop_count'); mxh_c = col('max_hops'); mnh_c = col('min_hops')
        pc_c  = col('avg_path_cost'); st_c  = col('path_stability')
        iso_c = col('isolation_events_cum')

        if not pdr_c.empty:  print(f"  Avg PDR                : {pdr_c.mean()*100:.2f} %")
        if not dly_c.empty:  print(f"  Avg E2E Delay          : {dly_c.mean():.2f} ms")
        if not h_c.empty:
            print(f"  Avg Hop Count (overall): {h_c.mean():.2f}")
            print(f"  Peak Hop Count         : {mxh_c.max():.0f}")
            print(f"  Floor Hop Count        : {mnh_c.min():.0f}")
        if not pc_c.empty:   print(f"  Avg Path Cost          : {pc_c.mean():.4f}")
        if not st_c.empty:   print(f"  Avg Path Stability     : {st_c.mean()*100:.2f} %")
        if not iso_c.empty:  print(f"  Total Isolation Events : {iso_c.max():.0f}")

    if not route_changes.empty:
        print(f"\n  Total Route Changes    : {len(route_changes)}")
        print("  By reason:")
        for reason, cnt in route_changes['reason'].value_counts().items():
            print(f"    {reason:<30}: {cnt}")
    else:
        print("\n  ⚠  route_changes.csv is empty or missing.")
        print("     Expected events:")
        print("     • ~49 'route-established' events at first ML evaluation (~t=20s)")
        print("     • 'cost-rise'/'degrade' events during each attack window")
        print("     • 'isolated-loss' when ML isolates a node")
        print("     • 'route-established' again on node restore")

    if not path_traces.empty:
        h = pd.to_numeric(path_traces.get('hop_count', pd.Series()), errors='coerce')
        valid_h = h[(h > 0) & (h < 20)]
        print(f"\n  Path traces recorded   : {len(path_traces)}")
        if not valid_h.empty:
            print(f"  Mean hop count (traces): {valid_h.mean():.2f}")
        stable = path_traces.get('stable', pd.Series(dtype=float))
        if not stable.empty:
            print(f"  Stable paths fraction  : {(stable==1).mean()*100:.1f} %")
        print("  State distribution:")
        for state, cnt in path_traces['state'].value_counts().items():
            print(f"    {state:<30}: {cnt}")

    print("═" * 62 + "\n")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    global _RESULTS_DIR

    parser = argparse.ArgumentParser(
        description="WSN Phase 2 Routing Evidence Visualiser [FIXED]")
    parser.add_argument("--scenario", default="D")
    parser.add_argument("--results",  default="results",
                        help="Path to results directory")
    args = parser.parse_args()

    _RESULTS_DIR = args.results
    scenario     = args.scenario.upper()

    fig_dir = os.path.join(_RESULTS_DIR, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    print(f"\n{'═'*62}")
    print(f"  WSN Phase 2 Routing Evidence Visualiser  [FIXED]")
    print(f"  Scenario : {scenario}")
    print(f"  Results  : {os.path.abspath(_RESULTS_DIR)}")
    print(f"  Figures  : {os.path.abspath(fig_dir)}")
    print(f"{'═'*62}\n")

    print("[1/4] Loading performance log...")
    perf_df = load_performance(_RESULTS_DIR, scenario)

    print("[2/4] Loading path traces...")
    path_traces = load_path_traces(_RESULTS_DIR)

    print("[3/4] Loading route change events...")
    route_changes = load_route_changes(_RESULTS_DIR)

    print("[4/4] Checking routing matrices...")
    mats = available_matrices(_RESULTS_DIR)

    print(f"\n  Routing matrix snapshots : {len(mats)}")
    print(f"  Path trace rows          : {len(path_traces)}")
    print(f"  Route change events      : {len(route_changes)}")
    print()

    print("Generating figures...")
    fig1_routing_graph_snapshots(_RESULTS_DIR, path_traces)
    fig2_hop_count_evolution(perf_df)
    fig3_route_change_timeline(route_changes)
    fig4_path_cost_heatmap(path_traces)
    fig5_hop_distribution(path_traces)
    fig6_routing_adjacency_heatmap(_RESULTS_DIR)
    fig7_summary_panel(perf_df, route_changes, path_traces)

    print_summary_stats(perf_df, path_traces, route_changes)
    print(f"Done. Figures → {os.path.abspath(fig_dir)}/\n")


if __name__ == "__main__":
    main()
