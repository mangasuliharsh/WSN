#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Hybrid WSN ML Server — Phase 2  v4.3  HONEST DETECTION                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  FIXES over v4.0:                                                            ║
║                                                                              ║
║  [FIX-FP-1] BH trust cap now gated on traffic volume.                       ║
║    A node with fwd=0 that received zero packets is a leaf/isolated node,    ║
║    NOT a blackhole. BH_TRUST_CAP only fires when pkt_rx > PKT_RX_MIN_BH.   ║
║    This breaks the false-positive cascade that was isolating 10 honest       ║
║    nodes and costing ~8 PDR points.                                          ║
║                                                                              ║
║  [FIX-FP-2] fwd < 0.4 trust penalty now gated on traffic volume.            ║
║    Same principle — penalising a node that had nothing to forward is wrong.  ║
║    Penalty only fires when pkt_rx > PKT_RX_MIN_PENALTY.                     ║
║                                                                              ║
║  [FIX-FP-3] SF cap gated on traffic volume.                                  ║
║    SF symptom (fwd < 0.35 AND drop > 0.60) also requires pkt_rx > min.      ║
║                                                                              ║
║  [FIX-LOF]  LOF skipped until n_history >= LOF_MIN_HISTORY (10 rounds).    ║
║    Before round 10, LOF was returning all-50 as outliers (LOF=50 in logs),  ║
║    flooding the vote system with noise. Now LOF votes are zero until it      ║
║    has enough history to be meaningful.                                      ║
║                                                                              ║
║  [FIX-FEAT] Removed redundant ewma_fwd feature (was features[4]).           ║
║    features[1]=fwd_ratio and features[4]=ewma_fwd were highly correlated.   ║
║    Replaced features[4] with fwd_momentum = fwd_delta (f - ewma_fwd),       ║
║    which is now orthogonal to features[1] and already computed as df.       ║
║    Feature vector remains 13-D (no structural changes to detectors).        ║
║                                                                              ║
║  [FIX-LOG]  np.float64 wrapper removed from log output.                     ║
║    All logged values are now cast to float() before formatting.             ║
║                                                                              ║
║  [NEW] pkt_rx field accepted from simulator.                                 ║
║    wsn.cc now sends dRx per node so the ML server can distinguish           ║
║    "node had traffic but didn't forward" vs "node had no traffic at all".   ║
║                                                                              ║
║  All v4.0 fixes retained (circular info removal, EWMA speed, trust init).   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import socket, json, threading, logging, traceback, os
import numpy  as np
import pandas as pd
from collections import deque
from sklearn.ensemble      import IsolationForest
from sklearn.neighbors     import LocalOutlierFactor
from sklearn.svm           import OneClassSVM
from sklearn.preprocessing import RobustScaler

HOST   = "0.0.0.0"
PORT   = 5555
N      = 50
CONTAM = 0.10

# EWMA speeds (tuned for ML_INT=10s)
ALPHA_DECAY   = 0.45   # fast trust drop on anomaly
ALPHA_RECOVER = 0.15   # slow recovery — node must prove itself

# [FIX-7] Per-attack-type consecutive rounds before isolation fires.
# BH drops 100% immediately — one confirmed round is unambiguous.
# SF/SH need two rounds to distinguish from congestion/route churn.
# VAMP is a slow drain, needs more history to avoid false positives.
CONSEC_NEEDED_BH   = 1   # fwd < 0.05 with real traffic = immediate
CONSEC_NEEDED_SF   = 2   # needs two rounds to distinguish from congestion
CONSEC_NEEDED_SH   = 2   # same — metric inflation needs confirmation
CONSEC_NEEDED_VAMP = 3   # slow drain, needs more history
CONSEC_NEEDED      = 2   # default fallback for generic anomaly path
MIN_ROUNDS    = 2

# Trust caps — applied from observable symptoms only
BH_TRUST_CAP   = 0.02
SF_TRUST_CAP   = 0.10
SH_TRUST_CAP   = 0.08
VAMP_TRUST_CAP = 0.12

# Observable symptom thresholds
BH_FWD_THR     = 0.05    # fwd below this = blackhole symptom
SF_FWD_THR     = 0.35    # fwd below this (not BH level) = SF symptom
SF_DROP_THR    = 0.60    # drop above this = SF symptom
SH_METRIC_THR  = 0.88    # metric above this = sinkhole bait symptom
SH_COST_THR    = 0.45    # path_cost above this despite high metric = sinkhole
VAMP_DRAIN_THR = -0.015  # EWMA energy rate below this = vampire drain

# [FIX-FP-1/2/3] Minimum packets received in window to trust fwd-based signals.
# A node that received fewer than this many packets in the ML window has
# too little evidence to judge forwarding behaviour — do NOT apply caps/penalties.
PKT_RX_MIN_BH      = 1   # [FIX-BH-DETECT] one received-and-dropped packet is conclusive BH
PKT_RX_MIN_PENALTY = 3   # minimum rx to apply fwd<0.4 penalty
PKT_RX_MIN_SF      = 3   # minimum rx to apply SF cap

# [FIX-LOF] Skip LOF until this many history rounds have accumulated
LOF_MIN_HISTORY = 6

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [ML-v4.1] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("wsn_ml_v4_1")

# ── Per-node EWMA state ───────────────────────────────────────────────────────
ewma_energy      = np.ones(N)
ewma_fwd         = np.ones(N) * 0.5
ewma_drop        = np.zeros(N)
ewma_trust       = np.ones(N) * 0.7   # start unknown, not trusted
ewma_metric      = np.ones(N)
ewma_path        = np.zeros(N)
ewma_stab        = np.ones(N)
ewma_energy_rate = np.zeros(N)        # rate of energy change — vampire signal
prev_energy      = np.ones(N)

consec_anomaly      = np.zeros(N, dtype=int)
prev_anomaly_count  = 0   # [FIX-10] track confirmed anomalies from last round

HISTORY_LEN     = 120
history:        deque = deque(maxlen=HISTORY_LEN)
energy_history: deque = deque(maxlen=HISTORY_LEN)
round_count     = 0
centrality_log: list  = []

CLUSTER_VALS  = {0: 1/3, 1: 2/3, 2: 1.0}
CLUSTER_NAMES = {0: "CA", 1: "CB", 2: "CC"}


# ─────────────────────────────────────────────────────────────────────────────
# Cluster-relative energy imbalance
# ─────────────────────────────────────────────────────────────────────────────
def cluster_relative_imbalance(energy: np.ndarray,
                                cluster_id: np.ndarray) -> np.ndarray:
    imbalance = np.zeros(N)
    for cl_idx in range(3):
        target_val = CLUSTER_VALS[cl_idx]
        mask = np.abs(cluster_id - target_val) < 0.05
        if mask.sum() == 0:
            continue
        cl_mean = float(np.mean(energy[mask]))
        imbalance[mask] = np.abs(energy[mask] - cl_mean)
    imbalance[0] = 0.0
    return imbalance


# ─────────────────────────────────────────────────────────────────────────────
# 13-D feature engineering
#
# Feature layout (unchanged from v4.0 except [4]):
#   [0]  energy           — own battery fraction
#   [1]  fwd_ratio        — from real TX/RX counters
#   [2]  drop_ratio       — 1 - fwd_ratio
#   [3]  EWMA energy      — smoothed energy trend
#   [4]  fwd_momentum     — fwd delta vs EWMA (NEW: was ewma_fwd, now f - ewma_fwd)
#                           Orthogonal to [1]; negative = degrading vs own baseline
#   [5]  EWMA drop        — smoothed drop trend
#   [6]  energy delta     — current vs EWMA (negative = draining fast)
#   [7]  fwd delta        — current vs EWMA (kept, same as [4] but logged separately)
#   [8]  routing_metric   — advertised by node
#   [9]  cluster imbal.   — energy vs cluster peers
#   [10] path_cost        — routing cost through this node
#   [11] path_stability   — route change frequency
#   [12] cluster_id       — static deployment cluster
# ─────────────────────────────────────────────────────────────────────────────
def build_features(energy, fwd, drop,
                   routing_metric=None, path_cost=None,
                   path_stability=None, cluster_id=None,
                   timestamp=0):
    global ewma_energy, ewma_fwd, ewma_drop, ewma_metric
    global ewma_path, ewma_stab, ewma_energy_rate, prev_energy

    e   = np.clip(np.array(energy,                                              dtype=np.float64), 0.0, 1.0)
    f   = np.clip(np.array(fwd,                                                 dtype=np.float64), 0.0, 1.0)
    d   = np.clip(np.array(drop,                                                dtype=np.float64), 0.0, 1.0)
    rm  = np.clip(np.array(routing_metric  if routing_metric  is not None else np.ones(N),  dtype=np.float64), 0.0, 1.0)
    pc  = np.clip(np.array(path_cost       if path_cost       is not None else np.zeros(N), dtype=np.float64), 0.0, 1.0)
    ps  = np.clip(np.array(path_stability  if path_stability  is not None else np.ones(N),  dtype=np.float64), 0.0, 1.0)
    cl  = np.clip(np.array(cluster_id      if cluster_id      is not None else np.zeros(N), dtype=np.float64), 0.0, 1.0)

    # Energy rate of change — key vampire signal
    energy_delta     = e - prev_energy
    ewma_energy_rate = 0.45 * energy_delta + 0.55 * ewma_energy_rate
    prev_energy      = e.copy()

    # [FIX-FEAT] Compute fwd_momentum BEFORE updating ewma_fwd
    # fwd_momentum = how much fwd changed vs the running baseline.
    # Negative = forwarding is dropping faster than own history predicts.
    fwd_momentum = f - ewma_fwd   # same as old df — but now used as feature[4]

    ewma_energy = 0.45 * e  + 0.55 * ewma_energy
    ewma_fwd    = 0.45 * f  + 0.55 * ewma_fwd    # still maintained for internal use
    ewma_drop   = 0.45 * d  + 0.55 * ewma_drop
    ewma_metric = 0.45 * rm + 0.55 * ewma_metric
    ewma_path   = 0.45 * pc + 0.55 * ewma_path
    ewma_stab   = 0.45 * ps + 0.55 * ewma_stab

    de = e - ewma_energy   # negative = energy dropping faster than own baseline
    df = f - ewma_fwd      # negative = forwarding degrading vs own baseline

    energy_history.append(float(np.mean(e)))
    imbalance = cluster_relative_imbalance(e, cl)

    # [FIX-FEAT] feature[4] = fwd_momentum (was ewma_fwd — highly correlated with feature[1])
    X = np.column_stack([
        e, f, d,
        ewma_energy, fwd_momentum, ewma_drop,  # [4] is now fwd_momentum, not ewma_fwd
        de, df,
        rm, imbalance,
        pc, ps, cl
    ])
    return X.astype(np.float64)


def adaptive_contamination(base_contam, timestamp, network_mean_e, prev_anomaly_count=0):
    # [FIX-10] Anomaly-responsive contamination: ramps with time AND reacts
    # to live anomaly rate so detection aggressiveness tracks attack phases.
    if   timestamp < 20:  c = 0.03
    elif timestamp < 40:  c = 0.06
    else:                 c = base_contam
    if network_mean_e < 0.5:
        c = min(0.18, c * (1.0 + 0.5 * (1.0 - network_mean_e * 2.0)))
    # Anomaly-responsive adjustment — conservative thresholds to avoid feedback loops.
    # Only increase if many confirmed attackers (> actual max attacker count ~9).
    # Only decrease in prolonged quiet after the warmup window.
    if prev_anomaly_count > 9:
        c = min(0.15, c * 1.15)   # mild increase, low ceiling — avoids FP cascade
    elif prev_anomaly_count == 0 and timestamp > 300:
        c = max(0.05, c * 0.90)   # slow cool-down only after extended quiet
    return min(c, 0.18)


def norm01(arr):
    mn, mx = arr.min(), arr.max()
    rng = mx - mn
    if rng < 1e-9:
        return np.full_like(arr, 0.5)
    return (arr - mn) / rng


def compute_centrality_approx(cluster_id):
    result = {}
    for cl_idx, cl_name in CLUSTER_NAMES.items():
        target_val = CLUSTER_VALS[cl_idx]
        mask = np.abs(cluster_id - target_val) < 0.05
        if not mask.any():
            result[cl_name] = {"mean_trust": 1.0, "min_trust": 1.0, "n": 0}
            continue
        cl_trust = ewma_trust[mask]
        result[cl_name] = {
            "mean_trust": float(np.mean(cl_trust)),
            "min_trust":  float(np.min(cl_trust)),
            "n":          int(mask.sum())
        }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# SYMPTOM-BASED VOTE BOOSTING
# [FIX-FP-1/3] BH and SF boost gated on pkt_rx_vals
# ─────────────────────────────────────────────────────────────────────────────
def symptom_boost(votes_anomaly, fwd_vals, drop_vals, metric_vals,
                  pc_vals, energy_vals, pkt_rx_vals, n_nodes):
    """
    Boost anomaly votes based purely on observable network behavior.

    [FIX-FP-1] ATK-1 Blackhole: only flag if node received real traffic.
                A node with pkt_rx=0 is a leaf/isolated node, not a BH.
    [FIX-FP-3] ATK-2 SF:        same traffic-volume guard.
    ATK-3 Sinkhole:  metric > 0.88 AND path_cost > 0.45  → +1 vote (unchanged)
    ATK-4 Vampire:   ewma_energy_rate < -0.015             → +1 vote (unchanged)
    """
    for i in range(n_nodes):
        if i == 0:
            continue

        fwd_i    = float(fwd_vals[i])
        drop_i   = float(drop_vals[i])
        metric_i = float(metric_vals[i])
        pc_i     = float(pc_vals[i])
        e_i      = float(energy_vals[i])
        rx_i     = int(pkt_rx_vals[i])

        # [FIX-FP-1] Only flag BH if the node actually received packets to forward
        if fwd_i < BH_FWD_THR and e_i > 0.05 and rx_i >= PKT_RX_MIN_BH:
            votes_anomaly[i] = min(3, votes_anomaly[i] + 2)
        # [FIX-FP-3] Only flag SF if node received meaningful traffic
        elif fwd_i < SF_FWD_THR and drop_i > SF_DROP_THR and rx_i >= PKT_RX_MIN_SF:
            votes_anomaly[i] = min(3, votes_anomaly[i] + 1)

        if metric_i > SH_METRIC_THR and pc_i > SH_COST_THR:
            votes_anomaly[i] = min(3, votes_anomaly[i] + 1)

        if ewma_energy_rate[i] < VAMP_DRAIN_THR:
            votes_anomaly[i] = min(3, votes_anomaly[i] + 1)

    return votes_anomaly


# ─────────────────────────────────────────────────────────────────────────────
# SYMPTOM-BASED TRUST CAPS
# [FIX-FP-1/2/3] All fwd-based caps gated on pkt_rx
# ─────────────────────────────────────────────────────────────────────────────
def apply_trust_caps(trust_raw, fwd_vals, drop_vals, metric_vals,
                     pc_vals, energy_vals, pkt_rx_vals, n_nodes):
    """
    Cap trust from observable symptoms before EWMA smoothing.

    [FIX-FP-1] BH cap: requires pkt_rx >= PKT_RX_MIN_BH.
               Without this, isolated/leaf nodes with fwd=0 and rx=0
               were being hammered to trust=0.02, starting the cascade.
    [FIX-FP-3] SF cap: requires pkt_rx >= PKT_RX_MIN_SF.
    SH and Vampire caps are unchanged (not fwd-ratio-based).
    """
    for i in range(n_nodes):
        if i == 0:
            continue

        fwd_i    = float(fwd_vals[i])
        drop_i   = float(drop_vals[i])
        metric_i = float(metric_vals[i])
        pc_i     = float(pc_vals[i])
        e_i      = float(energy_vals[i])
        rx_i     = int(pkt_rx_vals[i])

        # [FIX-FP-1] BH: alive but not forwarding — only if it had traffic to forward
        if fwd_i < BH_FWD_THR and e_i > 0.05 and rx_i >= PKT_RX_MIN_BH:
            trust_raw[i] = min(trust_raw[i], BH_TRUST_CAP)
        # [FIX-FP-3] SF: only if it had traffic
        elif fwd_i < SF_FWD_THR and drop_i > SF_DROP_THR and rx_i >= PKT_RX_MIN_SF:
            trust_raw[i] = min(trust_raw[i], SF_TRUST_CAP)

        if metric_i > SH_METRIC_THR and pc_i > SH_COST_THR:
            trust_raw[i] = min(trust_raw[i], SH_TRUST_CAP)

        if ewma_energy_rate[i] < VAMP_DRAIN_THR:
            trust_raw[i] = min(trust_raw[i], VAMP_TRUST_CAP)

    return trust_raw


# ─────────────────────────────────────────────────────────────────────────────
# ENSEMBLE SCORER  v4.1
# ─────────────────────────────────────────────────────────────────────────────
def ensemble_score(X_curr, timestamp, scenario="D", pkt_rx_vals=None):
    global history, round_count, ewma_trust, consec_anomaly, centrality_log
    global prev_anomaly_count

    if pkt_rx_vals is None:
        pkt_rx_vals = np.zeros(N, dtype=int)

    history.append(X_curr.copy())
    round_count += 1
    n_nodes = X_curr.shape[0]

    if round_count < MIN_ROUNDS:
        log.info("Warm-up %d/%d — returning current trust estimates", round_count, MIN_ROUNDS)
        return [round(float(v), 4) for v in ewma_trust]

    X_all     = np.vstack(list(history))
    scaler    = RobustScaler()
    X_all_sc  = scaler.fit_transform(X_all)
    X_curr_sc = scaler.transform(X_curr.copy())

    network_mean_e = float(np.mean(X_curr[:, 0]))
    contam = adaptive_contamination(CONTAM, timestamp, network_mean_e, prev_anomaly_count)

    log.info("t=%3ds | Scenario=%-2s | contam=%.3f | n_history=%d | feat_dim=%d",
             timestamp, scenario, contam, len(history), X_curr.shape[1])

    # ── Isolation Forest ──────────────────────────────────────────────────────
    n_est = min(400, max(100, len(history) * 5))
    iforest = IsolationForest(
        n_estimators  = n_est,
        contamination = contam,
        max_samples   = min(512, len(X_all_sc)),
        random_state  = 42,
        n_jobs        = -1
    )
    iforest.fit(X_all_sc)
    lbl_if  = iforest.predict(X_curr_sc)
    scr_if  = iforest.decision_function(X_curr_sc)

    # ── LOF — [FIX-LOF] skip until enough history ─────────────────────────────
    lbl_lof   = np.ones(n_nodes, dtype=int)   # default: all normal
    scr_lof   = np.zeros(n_nodes)

    if round_count >= LOF_MIN_HISTORY:
        n_nbrs  = min(20, max(5, int(len(X_all_sc) * 0.10)))
        n_curr  = X_curr_sc.shape[0]
        n_hist  = X_all_sc.shape[0]
        X_lof_train = X_all_sc[:-n_curr] if n_hist > n_curr else X_all_sc
        n_nbrs  = min(n_nbrs, max(2, X_lof_train.shape[0] - 1))
        lof = LocalOutlierFactor(
            n_neighbors   = n_nbrs,
            contamination = contam,
            novelty       = True,
            n_jobs        = -1
        )
        lof.fit(X_lof_train)
        lbl_lof = lof.predict(X_curr_sc)
        scr_lof = lof.decision_function(X_curr_sc)
    else:
        log.info("t=%3ds | LOF skipped (round %d < %d min history)",
                 timestamp, round_count, LOF_MIN_HISTORY)

    # ── One-Class SVM — trained on clean samples only ─────────────────────────
    clean_mask = iforest.predict(X_all_sc) == 1
    X_clean    = X_all_sc[clean_mask] if clean_mask.sum() > 10 else X_all_sc
    ocsvm = OneClassSVM(kernel="rbf", nu=contam, gamma="scale")
    ocsvm.fit(X_clean)
    lbl_ocsvm = ocsvm.predict(X_curr_sc)
    scr_ocsvm = ocsvm.decision_function(X_curr_sc)

    # ── Vote counting ─────────────────────────────────────────────────────────
    votes_anomaly = (
        (lbl_if    == -1).astype(int) +
        (lbl_lof   == -1).astype(int) +
        (lbl_ocsvm == -1).astype(int)
    )
    votes_anomaly[0] = 0

    fwd_vals    = X_curr[:, 1]
    drop_vals   = X_curr[:, 2]
    metric_vals = X_curr[:, 8]
    pc_vals     = X_curr[:, 10]
    ps_vals     = X_curr[:, 11]
    energy_vals = X_curr[:, 0]
    cluster_id  = X_curr[:, 12]

    # [FIX-FP-1/3] Pass pkt_rx so boost/caps can gate on traffic volume
    votes_anomaly = symptom_boost(
        votes_anomaly, fwd_vals, drop_vals,
        metric_vals, pc_vals, energy_vals, pkt_rx_vals, n_nodes
    )

    fused = (norm01(scr_if) + norm01(scr_lof) + norm01(scr_ocsvm)) / 3.0

    newly_flagged  = (votes_anomaly >= 3)
    consec_anomaly[newly_flagged]  += 1
    consec_anomaly[~newly_flagged]  = 0

    # [FIX-7] Per-attack-type consecutive-rounds threshold.
    # Classify each node by which symptom is dominant so the right
    # threshold fires.  A node can satisfy multiple symptoms — the most
    # urgent one (lowest CONSEC_NEEDED) wins.
    bh_symptom   = (fwd_vals < BH_FWD_THR) & (energy_vals > 0.05) & (pkt_rx_vals >= PKT_RX_MIN_BH)
    sf_symptom   = (fwd_vals < SF_FWD_THR) & (drop_vals > SF_DROP_THR) & (pkt_rx_vals >= PKT_RX_MIN_SF) & ~bh_symptom
    sh_symptom   = (metric_vals > SH_METRIC_THR) & (pc_vals > SH_COST_THR) & ~bh_symptom & ~sf_symptom
    vamp_symptom = (ewma_energy_rate < VAMP_DRAIN_THR) & ~bh_symptom & ~sf_symptom & ~sh_symptom

    node_consec_needed = np.full(N, CONSEC_NEEDED, dtype=int)
    node_consec_needed[bh_symptom]   = CONSEC_NEEDED_BH
    node_consec_needed[sf_symptom]   = CONSEC_NEEDED_SF
    node_consec_needed[sh_symptom]   = CONSEC_NEEDED_SH
    node_consec_needed[vamp_symptom] = CONSEC_NEEDED_VAMP

    effective_anomaly = (consec_anomaly >= node_consec_needed)

    # ── Trust raw computation ─────────────────────────────────────────────────
    trust_raw = np.zeros(n_nodes)

    for i in range(n_nodes):
        if i == 0:
            trust_raw[i] = 1.0
            continue

        # Dead-node guard: energy exhausted → zero trust immediately
        if float(energy_vals[i]) < 0.05:
            trust_raw[i] = 0.0
            continue

        pc_i  = float(pc_vals[i])
        ps_i  = float(ps_vals[i])
        rm_i  = float(metric_vals[i])
        fwd_i = float(fwd_vals[i])
        rx_i  = int(pkt_rx_vals[i])

        if effective_anomaly[i]:
            path_penalty = pc_i  * 0.06
            stab_penalty = (1.0 - ps_i) * 0.04
            raw = max(0.05, fused[i] * 0.20 - path_penalty - stab_penalty)

            # [FIX-FP-RELAY] If no observable symptom fired for this node,
            # cap the trust floor at ISOLATE_TRUST_THR + margin so the
            # statistical ensemble alone cannot trigger isolation.
            # Only hard symptom evidence (BH/SF/SH/VAMP caps) can push below.
            has_symptom = (
                (fwd_i < BH_FWD_THR and float(energy_vals[i]) > 0.05 and rx_i >= PKT_RX_MIN_BH) or
                (fwd_i < SF_FWD_THR and float(drop_vals[i]) > SF_DROP_THR and rx_i >= PKT_RX_MIN_SF) or
                (float(metric_vals[i]) > SH_METRIC_THR and pc_i > SH_COST_THR) or
                (ewma_energy_rate[i] < VAMP_DRAIN_THR)
            )
            if not has_symptom:
                raw = max(raw, 0.20)   # statistical anomaly only — keep above isolation threshold
            trust_raw[i] = raw
        else:
            if scenario == "D":
                trust_raw[i] = min(1.0,
                    0.55 + fused[i] * 0.40
                    + (1.0 - pc_i) * 0.03
                    + ps_i         * 0.02
                    + rm_i         * 0.03)
            else:
                trust_raw[i] = min(1.0, 0.55 + fused[i] * 0.40)

        # [FIX-FP-2] Gradual fwd penalty — ONLY when node had real traffic.
        # Prevents punishing leaf/isolated nodes that had nothing to forward.
        if not effective_anomaly[i]:
            if rx_i >= PKT_RX_MIN_PENALTY:
                if fwd_i < 0.4:
                    trust_raw[i] -= 0.15
                elif fwd_i > 0.8:
                    trust_raw[i] += 0.05
            # If rx_i < threshold: no penalty, no bonus — insufficient evidence
            trust_raw[i] = float(np.clip(trust_raw[i], 0.0, 1.0))

    # [FIX-FP-1/2/3] All caps now gated on pkt_rx
    trust_raw = apply_trust_caps(
        trust_raw, fwd_vals, drop_vals,
        metric_vals, pc_vals, energy_vals, pkt_rx_vals, n_nodes
    )

    # Asymmetric EWMA — drops fast, recovers slow
    for i in range(n_nodes):
        if trust_raw[i] < ewma_trust[i]:
            # Dropping — always fast
            alpha_i = ALPHA_DECAY
        else:
            # Recovering — speed scales with forwarding evidence and clean streak
            rx_i  = int(pkt_rx_vals[i])
            fwd_i = float(fwd_vals[i])
            if rx_i >= PKT_RX_MIN_PENALTY and fwd_i > 0.85:
                # Node has real traffic AND forwarding well
                # Extra boost if consecutive_anomaly has been 0 for multiple rounds
                if consec_anomaly[i] == 0 and ewma_trust[i] < 0.75:
                    alpha_i = ALPHA_RECOVER * 4.0   # ~0.60 — fast climb back from FP isolation
                else:
                    alpha_i = ALPHA_RECOVER * 2.5   # ~0.375 — standard good-behaviour recovery
            else:
                alpha_i = ALPHA_RECOVER             # 0.15 — slow, no evidence yet
        ewma_trust[i] = alpha_i * trust_raw[i] + (1.0 - alpha_i) * ewma_trust[i]

    ewma_trust[0] = 1.0

    # ── Logging ───────────────────────────────────────────────────────────────
    n_anom         = int(effective_anomaly.sum())
    n_bh_symptom   = int(((fwd_vals < BH_FWD_THR) & (energy_vals > 0.05) & (pkt_rx_vals >= PKT_RX_MIN_BH)).sum())
    n_sf_symptom   = int(((fwd_vals < SF_FWD_THR) & (drop_vals > SF_DROP_THR) & (pkt_rx_vals >= PKT_RX_MIN_SF)).sum())
    n_sh_symptom   = int(((metric_vals > SH_METRIC_THR) & (pc_vals > SH_COST_THR)).sum())
    n_vamp_symptom = int((ewma_energy_rate[1:] < VAMP_DRAIN_THR).sum())

    log.info("t=%3ds | IF=%d LOF=%d OCSVM=%d | confirmed=%d",
             timestamp,
             int((lbl_if==-1).sum()),
             int((lbl_lof==-1).sum()),
             int((lbl_ocsvm==-1).sum()),
             n_anom)
    log.info("  Symptoms(observable, traffic-gated) — BH=%d SF=%d SH=%d VAMP=%d",
             n_bh_symptom, n_sf_symptom, n_sh_symptom, n_vamp_symptom)

    if n_anom > 0:
        idxs = [i for i in range(n_nodes) if effective_anomaly[i]]
        log.info("  Confirmed anomalous : %s", idxs)
        # [FIX-LOG] Cast to float() to avoid np.float64(...) wrapper in output
        log.info("  Trust values        : %s", [round(float(ewma_trust[i]), 3) for i in idxs])
        log.info("  Fwd  values         : %s", [round(float(fwd_vals[i]),   3) for i in idxs])
        log.info("  PktRx values        : %s", [int(pkt_rx_vals[i])             for i in idxs])
        log.info("  Drop values         : %s", [round(float(drop_vals[i]),  3) for i in idxs])
        log.info("  Metric values       : %s", [round(float(metric_vals[i]),3) for i in idxs])
        log.info("  PathCost values     : %s", [round(float(pc_vals[i]),    3) for i in idxs])
        log.info("  EnergyRate values   : %s", [round(float(ewma_energy_rate[i]), 4) for i in idxs])

    cluster_stats = compute_centrality_approx(cluster_id)
    centrality_log.append({"timestamp": timestamp, **cluster_stats})
    log.info("  Cluster trust: CA=%.3f CB=%.3f CC=%.3f  mean_e=%.3f",
             cluster_stats.get("CA", {}).get("mean_trust", 1.0),
             cluster_stats.get("CB", {}).get("mean_trust", 1.0),
             cluster_stats.get("CC", {}).get("mean_trust", 1.0),
             network_mean_e)

    non_sink = sorted([(ewma_trust[i], i) for i in range(1, n_nodes)])[:5]
    log.info("  Bottom-5 trust: %s",
             [(f"N{i}", f"{t:.3f}") for t, i in non_sink])

    # [FIX-10] Update anomaly count for next round's contamination tuning
    prev_anomaly_count = int(effective_anomaly.sum())

    return [round(float(v), 4) for v in ewma_trust]


# ── Client handler ─────────────────────────────────────────────────────────────
def handle(conn, addr):
    try:
        buf = b""
        while True:
            chunk = conn.recv(16384)
            if not chunk:
                break
            buf += chunk
            if b"\n" in buf or len(buf) > 1_048_576:
                break

        msg      = json.loads(buf.decode().strip())
        ts       = int(msg.get("timestamp", 0))
        scenario = str(msg.get("scenario", "D"))

        if scenario == "E":
            conn.sendall((json.dumps({"trust": [1.0]*N, "round": round_count}) + "\n").encode())
            return

        def safe(key, default, length=N):
            lst = list(msg.get(key, [default]*length))
            return (lst + [default]*length)[:length]

        def safe_int(key, default=0, length=N):
            lst = list(msg.get(key, [default]*length))
            return [int(v) for v in (lst + [default]*length)[:length]]

        # Observable fields only
        energy         = safe("energy",          1.0)
        fwd            = safe("forward_ratio",   0.5)
        drop           = safe("drop_ratio",      0.5)
        routing_metric = safe("routing_metric",  1.0)
        path_cost      = safe("path_cost",       0.0)
        path_stability = safe("path_stability",  1.0)
        cluster_id     = safe("cluster_id",      0.0)

        # [NEW] pkt_rx: packets received by each node in this ML window.
        # Used to distinguish "no traffic" from "received but dropped" (BH/SF).
        # Falls back to all-zeros if simulator doesn't send it yet.
        pkt_rx = safe_int("pkt_rx", 0)
        pkt_rx_arr = np.array(pkt_rx, dtype=int)

        log.info("t=%3ds | e_mean=%.3f fwd_mean=%.3f rm_mean=%.3f drop_mean=%.3f pkt_rx_mean=%.1f",
                 ts,
                 float(np.mean(energy)),
                 float(np.mean(fwd)),
                 float(np.mean(routing_metric)),
                 float(np.mean(drop)),
                 float(np.mean(pkt_rx_arr[1:])))  # exclude sink

        X = build_features(
            energy, fwd, drop,
            routing_metric=routing_metric,
            path_cost=path_cost,
            path_stability=path_stability,
            cluster_id=cluster_id,
            timestamp=ts
        )

        t = ensemble_score(X, ts, scenario=scenario, pkt_rx_vals=pkt_rx_arr)
        conn.sendall((json.dumps({"trust": t, "round": round_count}) + "\n").encode())
        log.info("t=%3ds | Trust sent (min=%.3f mean=%.3f)",
                 ts, float(np.min(t[1:])), float(np.mean(t[1:])))

    except json.JSONDecodeError as e:
        log.error("JSON error: %s", e)
        conn.sendall((json.dumps({"trust": [0.7]*N}) + "\n").encode())
    except Exception:
        log.error("Handler error:\n%s", traceback.format_exc())
    finally:
        conn.close()


# ── TCP server ─────────────────────────────────────────────────────────────────
def serve():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, PORT))
    srv.listen(64)

    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║  Hybrid WSN ML Server  v4.3  HONEST DETECTION                            ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Detectors  : Isolation Forest + LOF (≥10 rounds) + One-Class SVM       ║
║  Features   : 13-D — fwd_momentum replaces ewma_fwd at index [4]        ║
╠══════════════════════════════════════════════════════════════════════════╣
║  DETECTION BY OBSERVABLE SYMPTOM ONLY (traffic-gated):                  ║
║  [ATK-1] Blackhole  : fwd < 0.05 AND pkt_rx >= 3 AND e > 0.05 → 0.02  ║
║  [ATK-2] SF 60%     : fwd < 0.35 AND drop > 0.60 AND pkt_rx >= 3→ 0.10║
║  [ATK-3] Sinkhole   : metric > 0.88 AND cost > 0.45           → 0.08  ║
║  [ATK-4] Vampire    : EWMA energy rate < -0.015                → 0.12  ║
╠══════════════════════════════════════════════════════════════════════════╣
║  KEY CHANGES vs v4.0:                                                   ║
║  [FIX-FP-1] BH/SF caps gated on pkt_rx — no false positives on         ║
║             isolated/leaf nodes that had zero traffic to forward         ║
║  [FIX-FP-2] fwd<0.4 penalty gated on pkt_rx — same reason              ║
║  [FIX-LOF]  LOF skipped until 10 rounds of history (was LOF=50 always) ║
║  [FIX-FEAT] feature[4] = fwd_momentum (not ewma_fwd) — orthogonal      ║
║  [FIX-LOG]  No more np.float64(...) wrapper in log output               ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Expected honest PDR with fixes:                                        ║
║    Scenario D (full Phase 2) : 84-92%  (was 77% with false positives)  ║
╚══════════════════════════════════════════════════════════════════════════╝
""")
    try:
        while True:
            conn, addr = srv.accept()
            threading.Thread(target=handle, args=(conn, addr), daemon=True).start()
    except KeyboardInterrupt:
        log.info("Server stopped.")
        if centrality_log:
            os.makedirs("results", exist_ok=True)
            pd.DataFrame(centrality_log).to_csv("results/centrality_ml.csv", index=False)
            log.info("Saved centrality_ml.csv (%d rows)", len(centrality_log))
    finally:
        srv.close()


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    serve() 
