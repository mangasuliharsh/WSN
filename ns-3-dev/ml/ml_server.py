#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Hybrid WSN ML Server — Phase 1 Upgrade                                     ║
║  Based on: Priyadarshi (2024), Wireless Networks 30:2647–2673               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  PHASE 1 ADDITIONS:                                                          ║
║    ✦ Routing metric feature (α·trust + β·energy) injected into ensemble     ║
║    ✦ Energy age factor: energy weight auto-scales with simulation time       ║
║    ✦ Scenario metadata accepted and logged per evaluation window             ║
║    ✦ Per-node energy imbalance contributed as anomaly signal                 ║
║                                                                              ║
║  Algorithms (unchanged from base):                                           ║
║    1. Isolation Forest   (IForest) — tree-based density estimation           ║
║    2. Local Outlier Factor (LOF)   — local neighborhood density              ║
║    3. One-Class SVM (OCSVM)        — kernel-based boundary learning          ║
║    4. EWMA Behavioral Memory       — exponential smoothing over time         ║
║    5. Majority Vote Fusion         — 2-of-3 → anomaly decision               ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import socket, json, threading, logging, traceback
import numpy as np
from collections import deque
from sklearn.ensemble      import IsolationForest
from sklearn.neighbors     import LocalOutlierFactor
from sklearn.svm           import OneClassSVM
from sklearn.preprocessing import RobustScaler

HOST       = "0.0.0.0"
PORT       = 5555
N          = 50
CONTAM     = 0.20
EWMA_ALPHA = 0.35
MIN_ROUNDS = 2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [ML] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("wsn_ml")

# ─── Per-node EWMA state ───────────────────────────────────────────────────────
ewma_energy  = np.ones(N)
ewma_fwd     = np.ones(N) * 0.5
ewma_drop    = np.zeros(N)
ewma_trust   = np.ones(N)
ewma_metric  = np.ones(N)          # Phase 1: EWMA of routing metric

# History ring-buffer
HISTORY_LEN = 120
history: deque = deque(maxlen=HISTORY_LEN)
round_count = 0

# Phase 1: global network mean energy for imbalance scoring
energy_history: deque = deque(maxlen=HISTORY_LEN)


# ── Feature engineering (Phase 1: 10-dimensional) ─────────────────────────────
def build_features(energy, fwd, drop, isolated, known_malicious,
                   routing_metric=None, timestamp=0):
    """
    10-dimensional feature vector per node:
      [0]  normalised energy
      [1]  forwarding ratio
      [2]  drop ratio
      [3]  EWMA energy
      [4]  EWMA forwarding ratio
      [5]  EWMA drop ratio
      [6]  energy delta from EWMA
      [7]  fwd delta from EWMA
      [8]  routing metric          ← Phase 1: α·trust + β·energy
      [9]  energy imbalance signal ← Phase 1: |node_energy - mean_energy|
    """
    global ewma_energy, ewma_fwd, ewma_drop, ewma_metric

    e  = np.array(energy,         dtype=np.float64)
    f  = np.array(fwd,            dtype=np.float64)
    d  = np.array(drop,           dtype=np.float64)
    rm = np.array(routing_metric, dtype=np.float64) \
         if routing_metric is not None else np.ones(N)

    # Update EWMAs
    ewma_energy = EWMA_ALPHA * e  + (1 - EWMA_ALPHA) * ewma_energy
    ewma_fwd    = EWMA_ALPHA * f  + (1 - EWMA_ALPHA) * ewma_fwd
    ewma_drop   = EWMA_ALPHA * d  + (1 - EWMA_ALPHA) * ewma_drop
    ewma_metric = EWMA_ALPHA * rm + (1 - EWMA_ALPHA) * ewma_metric

    # Deltas
    de = e - ewma_energy
    df = f - ewma_fwd

    # Phase 1: Energy imbalance signal
    # Nodes whose energy deviates strongly from network mean are flagged
    network_mean_e = float(np.mean(e))
    energy_history.append(network_mean_e)
    imbalance = np.abs(e - network_mean_e)   # higher = more imbalanced

    X = np.column_stack([e, f, d, ewma_energy, ewma_fwd,
                         ewma_drop, de, df, rm, imbalance])
    return X.astype(np.float64)


# ── Energy-age adaptive contamination ─────────────────────────────────────────
def adaptive_contamination(base_contam: float, timestamp: int,
                           network_mean_e: float) -> float:
    """
    Phase 1: Scale contamination with simulation age and energy depletion.
    As nodes deplete energy, increase sensitivity to anomalies (more attacks likely).
    """
    c = base_contam
    if timestamp < 20:
        c = 0.04         # pre-attack warm-up
    elif timestamp < 40:
        c = 0.12         # first attack ramping
    else:
        c = base_contam  # fully active

    # Energy age factor: if mean energy < 50%, bump contamination slightly
    # (network is older, more stressed, anomalies more impactful)
    energy_fraction = network_mean_e  # already normalised in 0-1
    if energy_fraction < 0.5:
        c = min(0.45, c * (1.0 + 0.3 * (1.0 - energy_fraction * 2.0)))

    return min(c, 0.45)


# ── Ensemble scorer (Phase 1 enhanced) ────────────────────────────────────────
def ensemble_score(X_curr, timestamp, scenario="C"):
    global history, round_count, ewma_trust

    history.append(X_curr.copy())
    round_count += 1

    n_nodes = X_curr.shape[0]
    trust_out = np.ones(n_nodes)

    if round_count < MIN_ROUNDS:
        log.info("Warm-up round %d/%d — all nodes trusted", round_count, MIN_ROUNDS)
        return [round(float(v), 4) for v in trust_out]

    X_all = np.vstack(list(history))
    X_curr_copy = X_curr.copy()

    scaler = RobustScaler()
    X_all_sc   = scaler.fit_transform(X_all)
    X_curr_sc  = scaler.transform(X_curr_copy)

    # Phase 1: adaptive contamination considers energy age
    network_mean_e = float(np.mean(X_curr[:, 0]))  # feature 0 = normalised energy
    contam = adaptive_contamination(CONTAM, timestamp, network_mean_e)

    log.info("t=%3ds | Scenario=%s | contam=%.3f | mean_energy=%.3f",
             timestamp, scenario, contam, network_mean_e)

    # ── Detector 1: Isolation Forest ──────────────────────────────────────────
    n_est = min(300, max(100, len(history) * 5))
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

    # ── Detector 2: Local Outlier Factor ──────────────────────────────────────
    n_nbrs = min(20, max(5, int(len(X_all_sc) * 0.10)))
    lof = LocalOutlierFactor(
        n_neighbors   = n_nbrs,
        contamination = contam,
        novelty       = True,
        n_jobs        = -1
    )
    lof.fit(X_all_sc)
    lbl_lof = lof.predict(X_curr_sc)
    scr_lof = lof.decision_function(X_curr_sc)

    # ── Detector 3: One-Class SVM ─────────────────────────────────────────────
    clean_mask = iforest.predict(X_all_sc) == 1
    X_clean = X_all_sc[clean_mask] if clean_mask.sum() > 10 else X_all_sc
    ocsvm = OneClassSVM(kernel="rbf", nu=contam, gamma="scale")
    ocsvm.fit(X_clean)
    lbl_ocsvm = ocsvm.predict(X_curr_sc)
    scr_ocsvm = ocsvm.decision_function(X_curr_sc)

    # ── Majority vote ─────────────────────────────────────────────────────────
    votes_anomaly = (
        (lbl_if    == -1).astype(int) +
        (lbl_lof   == -1).astype(int) +
        (lbl_ocsvm == -1).astype(int)
    )

    def norm01(arr):
        mn, mx = arr.min(), arr.max()
        return (arr - mn) / (mx - mn + 1e-9)

    fused = (norm01(scr_if) + norm01(scr_lof) + norm01(scr_ocsvm)) / 3.0

    # ── Phase 1: Weight routing metric into trust mapping ─────────────────────
    # For Scenario C, nodes with lower routing metric are penalised further
    routing_m = X_curr[:, 8]   # feature index 8 = routing_metric

    trust_raw = np.zeros(n_nodes)
    for i in range(n_nodes):
        if votes_anomaly[i] >= 2:
            # Anomalous: scale penalisation by how bad the routing metric is
            if scenario == "C":
                rm_penalty = max(0.0, 1.0 - routing_m[i])  # 0=healthy, 1=very bad
                trust_raw[i] = fused[i] * 0.29 * (1.0 - 0.3 * rm_penalty)
            else:
                trust_raw[i] = fused[i] * 0.29
        else:
            # Normal: boost trust for high routing metric nodes (Scenario C only)
            if scenario == "C":
                rm_boost = routing_m[i] * 0.10   # small bonus for healthy nodes
                trust_raw[i] = min(1.0, 0.70 + fused[i] * 0.30 + rm_boost)
            else:
                trust_raw[i] = 0.70 + fused[i] * 0.30

    # ── EWMA smoothing ────────────────────────────────────────────────────────
    ewma_trust = EWMA_ALPHA * trust_raw + (1 - EWMA_ALPHA) * ewma_trust
    ewma_trust[0] = 1.0  # sink always trusted

    # ── Diagnostics ───────────────────────────────────────────────────────────
    n_anom = int((votes_anomaly >= 2).sum())
    n_susp = int(((votes_anomaly == 1) & (ewma_trust < 0.5)).sum())
    n_soft = int(((routing_m < 0.4) & (votes_anomaly < 2)).sum())

    log.info("t=%3ds | IF=%d LOF=%d OCSVM=%d | anomalies=%d suspicious=%d soft-risk=%d",
             timestamp,
             int((lbl_if==-1).sum()), int((lbl_lof==-1).sum()), int((lbl_ocsvm==-1).sum()),
             n_anom, n_susp, n_soft)

    if n_anom > 0:
        idxs = [i for i in range(n_nodes) if votes_anomaly[i] >= 2]
        log.info("  Anomalous nodes: %s", idxs)
        log.info("  Their trust:     %s", [round(ewma_trust[i], 3) for i in idxs])

    if n_soft > 0 and scenario == "C":
        soft_idxs = [i for i in range(n_nodes)
                     if routing_m[i] < 0.4 and votes_anomaly[i] < 2]
        log.info("  Soft-risk nodes (low metric, not anomalous): %s", soft_idxs)

    return [round(float(v), 4) for v in ewma_trust]


# ── Client handler ─────────────────────────────────────────────────────────────
def handle(conn, addr):
    try:
        buf = b""
        while True:
            chunk = conn.recv(16384)
            if not chunk: break
            buf += chunk
            if b"\n" in buf or len(buf) > 1_048_576: break

        msg = json.loads(buf.decode().strip())
        ts       = int(msg.get("timestamp", 0))
        scenario = str(msg.get("scenario", "C"))

        energy         = msg.get("energy",           [1.0] * N)
        fwd            = msg.get("forward_ratio",    [0.5] * N)
        drop           = msg.get("drop_ratio",       [0.5] * N)
        iso            = msg.get("isolated",         [0.0] * N)
        known_m        = msg.get("known_malicious",  [0.0] * N)
        routing_metric = msg.get("routing_metric",   [1.0] * N)  # Phase 1

        log.info("t=%3ds | Scenario=%s | energy[0]=%.3f fwd[0]=%.3f rm_avg=%.3f",
                 ts, scenario, energy[0], fwd[0], float(np.mean(routing_metric)))

        X = build_features(energy, fwd, drop, iso, known_m,
                           routing_metric=routing_metric, timestamp=ts)
        t = ensemble_score(X, ts, scenario=scenario)

        resp = json.dumps({"trust": t, "round": round_count}) + "\n"
        conn.sendall(resp.encode())
        log.info("t=%3ds | Trust sent (%d bytes)", ts, len(resp))

    except json.JSONDecodeError as e:
        log.error("JSON error: %s", e)
        conn.sendall((json.dumps({"trust": [1.0]*N}) + "\n").encode())
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
╔════════════════════════════════════════════════════════════════════╗
║     Hybrid WSN — Phase 1 Ensemble ML Server                        ║
╠════════════════════════════════════════════════════════════════════╣
║  Detectors  : Isolation Forest + LOF + One-Class SVM               ║
║  Fusion     : Majority vote (2-of-3) + EWMA smoothing              ║
║  Features   : energy, fwd_ratio, drop_ratio, EWMA deltas,          ║
║               routing_metric, energy_imbalance  (10-D)  [PHASE 1]  ║
║  Phase 1    : Energy-age adaptive contamination                     ║
║               Routing-metric-aware trust mapping                    ║
║               Scenario A/B/C metadata logging                       ║
║  Trust <0.3 → isolation  |  Trust ≥ 0.3 → restore                  ║
║  Port       : 5555  |  Nodes : 50                                   ║
╚════════════════════════════════════════════════════════════════════╝
Waiting for ns-3 connection...
""")
    try:
        while True:
            conn, addr = srv.accept()
            threading.Thread(target=handle, args=(conn, addr), daemon=True).start()
    except KeyboardInterrupt:
        log.info("Server stopped.")
    finally:
        srv.close()


if __name__ == "__main__":
    serve()
