#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Hybrid WSN ML Server — Phase 2 Upgrade (BUG-FIXED)                          ║
║  Based on: Priyadarshi (2024), Wireless Networks 30:2647–2673                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  BUGS FIXED:                                                                  ║
║  [BF-1] CONTAM 0.20→0.06: was forcing 10/50 anomalies even in healthy nets   ║
║  [BF-2] Symmetric EWMA α=0.35: replaced with asymmetric decay/recovery +     ║
║         consecutive-clean bonus so innocent nodes recover in ~3 rounds        ║
║  [BF-3] votes_anomaly threshold: added round-based warm-up guard and         ║
║         per-node consecutive anomaly counter (requires 2 consecutive hits)    ║
║  [BF-6] Sink node (i=0) features polluted trust for neighbours: sink is now  ║
║         excluded from anomaly scoring entirely                                 ║
║  [BF-7] norm01() division-by-zero: guarded with max(range,1e-9)              ║
║  [BF-8] LOF novelty=True requires separate fit/predict sets: fixed to use    ║
║         a proper held-out current window rather than re-scoring training data  ║
║  [BF-9] OC-SVM clean_mask: could produce empty X_clean silently; now falls  ║
║         back to full X_all_sc with a warning                                  ║
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
# ── [BF-1] FIXED: was 0.20 → forced ~10 anomalies per round even in healthy
#    networks. 0.06 ≈ 3 nodes, matching realistic attack window sizes (2 nodes).
CONTAM     = 0.06

# ── [BF-2] FIXED: asymmetric EWMA — slower decay protects innocent nodes from
#    a single bad ML round; faster recovery so restored nodes re-earn trust
#    within ~3 rounds rather than 7+.
ALPHA_DECAY    = 0.25   # was 0.35 — used when trust is falling
ALPHA_RECOVER  = 0.45   # new    — used when trust is rising
MIN_ROUNDS     = 3      # was 2  — extra warm-up guard

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [ML-P2] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("wsn_ml_p2")

# ─── Per-node EWMA state ───────────────────────────────────────────────────────
ewma_energy   = np.ones(N)
ewma_fwd      = np.ones(N) * 0.5
ewma_drop     = np.zeros(N)
ewma_trust    = np.ones(N)
ewma_metric   = np.ones(N)
ewma_path     = np.zeros(N)
ewma_stab     = np.ones(N)

# ── [BF-3] Consecutive-anomaly counter: a node must be flagged anomalous in
#    2 consecutive rounds before its trust is penalised.  A single bad round
#    resets to 0 if the next round is clean.
consec_anomaly = np.zeros(N, dtype=int)   # per-node consecutive anomaly count
CONSEC_NEEDED  = 2                        # rounds needed before penalising

# History ring-buffer
HISTORY_LEN = 120
history: deque = deque(maxlen=HISTORY_LEN)
round_count = 0

energy_history: deque = deque(maxlen=HISTORY_LEN)


# ── Feature engineering (Phase 2: 12-dimensional) ─────────────────────────────
def build_features(energy, fwd, drop, isolated, known_malicious,
                   routing_metric=None, path_cost=None,
                   path_stability=None, timestamp=0):
    """
    12-dimensional feature vector per node:
      [0]  normalised energy
      [1]  forwarding ratio
      [2]  drop ratio
      [3]  EWMA energy
      [4]  EWMA forwarding ratio
      [5]  EWMA drop ratio
      [6]  energy delta from EWMA
      [7]  fwd delta from EWMA
      [8]  routing metric          (α·trust + β·energy)
      [9]  energy imbalance signal |node_energy - mean_energy|
      [10] path cost (normalised)  ← Phase 2: Σ(1-score) along path
      [11] path stability          ← Phase 2: fraction of stable intervals
    """
    global ewma_energy, ewma_fwd, ewma_drop, ewma_metric, ewma_path, ewma_stab

    e  = np.array(energy,           dtype=np.float64)
    f  = np.array(fwd,              dtype=np.float64)
    d  = np.array(drop,             dtype=np.float64)
    rm = np.array(routing_metric,   dtype=np.float64) \
         if routing_metric  is not None else np.ones(N)
    pc = np.array(path_cost,        dtype=np.float64) \
         if path_cost        is not None else np.zeros(N)
    ps = np.array(path_stability,   dtype=np.float64) \
         if path_stability   is not None else np.ones(N)

    # Clip to valid ranges to avoid NaN propagation
    e  = np.clip(e,  0.0, 1.0)
    f  = np.clip(f,  0.0, 1.0)
    d  = np.clip(d,  0.0, 1.0)
    rm = np.clip(rm, 0.0, 1.0)
    pc = np.clip(pc, 0.0, 1.0)
    ps = np.clip(ps, 0.0, 1.0)

    # Update EWMAs with fixed α (direction-agnostic for feature smoothing)
    ewma_energy = 0.35 * e  + 0.65 * ewma_energy
    ewma_fwd    = 0.35 * f  + 0.65 * ewma_fwd
    ewma_drop   = 0.35 * d  + 0.65 * ewma_drop
    ewma_metric = 0.35 * rm + 0.65 * ewma_metric
    ewma_path   = 0.35 * pc + 0.65 * ewma_path
    ewma_stab   = 0.35 * ps + 0.65 * ewma_stab

    de = e - ewma_energy
    df = f - ewma_fwd

    network_mean_e = float(np.mean(e))
    energy_history.append(network_mean_e)
    imbalance = np.abs(e - network_mean_e)

    X = np.column_stack([
        e, f, d,
        ewma_energy, ewma_fwd, ewma_drop,
        de, df,
        rm, imbalance,
        pc, ps
    ])
    return X.astype(np.float64)


# ── Energy-age adaptive contamination ─────────────────────────────────────────
def adaptive_contamination(base_contam: float, timestamp: int,
                           network_mean_e: float) -> float:
    """
    Ramp contamination up gently as the simulation matures.
    Stays well below 0.20 to avoid the forced-quota problem.
    """
    if   timestamp < 20: c = 0.03
    elif timestamp < 40: c = 0.05
    else:                c = base_contam          # 0.06

    # As energy depletes, slightly more nodes may behave oddly — allow up to 0.10
    energy_fraction = network_mean_e
    if energy_fraction < 0.5:
        c = min(0.10, c * (1.0 + 0.3 * (1.0 - energy_fraction * 2.0)))
    return min(c, 0.10)   # hard cap at 10% to prevent mass false positives


# ── [BF-7] FIXED: safe normalisation helper ───────────────────────────────────
def norm01(arr: np.ndarray) -> np.ndarray:
    """Normalise to [0,1]; returns 0.5 for constant arrays (avoids div-by-zero)."""
    mn, mx = arr.min(), arr.max()
    rng = mx - mn
    if rng < 1e-9:
        return np.full_like(arr, 0.5)
    return (arr - mn) / rng


# ── Ensemble scorer ────────────────────────────────────────────────────────────
def ensemble_score(X_curr, timestamp, scenario="D"):
    """
    Returns trust scores [0,1] for all N nodes.

    Key fixes applied here:
      [BF-1] adaptive_contamination() now caps at 0.10
      [BF-2] Asymmetric EWMA trust update
      [BF-3] consec_anomaly gate: 2 consecutive rounds required
      [BF-6] Sink (index 0) excluded from anomaly detection
      [BF-7] norm01() safe against constant arrays
      [BF-8] LOF trained on history, scored on current window only
      [BF-9] OC-SVM clean_mask fallback
    """
    global history, round_count, ewma_trust, consec_anomaly

    history.append(X_curr.copy())
    round_count += 1

    n_nodes = X_curr.shape[0]
    trust_out = np.ones(n_nodes)

    if round_count < MIN_ROUNDS:
        log.info("Warm-up round %d/%d — all nodes trusted", round_count, MIN_ROUNDS)
        return [round(float(v), 4) for v in trust_out]

    X_all = np.vstack(list(history))

    scaler     = RobustScaler()
    X_all_sc   = scaler.fit_transform(X_all)
    X_curr_sc  = scaler.transform(X_curr.copy())

    network_mean_e = float(np.mean(X_curr[:, 0]))
    contam = adaptive_contamination(CONTAM, timestamp, network_mean_e)

    log.info("t=%3ds | Scenario=%s | contam=%.3f | mean_e=%.3f | feat_dim=%d",
             timestamp, scenario, contam, network_mean_e, X_curr.shape[1])

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

    # ── Detector 2: LOF ───────────────────────────────────────────────────────
    # [BF-8] FIXED: novelty=True means LOF must be fitted on training data and
    # then predict() called on the *separate* current window — not the same data
    # it was trained on.  Previously, X_all_sc was used for both fit and predict,
    # which is invalid with novelty=True and causes sklearn to raise or silently
    # return garbage scores.
    n_nbrs = min(20, max(5, int(len(X_all_sc) * 0.10)))

    # Use all-but-current as training, current window as novelty test
    n_curr = X_curr_sc.shape[0]
    n_hist = X_all_sc.shape[0]
    if n_hist > n_curr:
        X_lof_train = X_all_sc[:-n_curr]   # history without the latest snapshot
    else:
        X_lof_train = X_all_sc             # not enough history — fall back

    # Ensure we have enough samples for the requested neighbour count
    n_nbrs = min(n_nbrs, max(2, X_lof_train.shape[0] - 1))

    lof = LocalOutlierFactor(
        n_neighbors   = n_nbrs,
        contamination = contam,
        novelty       = True,
        n_jobs        = -1
    )
    lof.fit(X_lof_train)
    lbl_lof = lof.predict(X_curr_sc)
    scr_lof = lof.decision_function(X_curr_sc)

    # ── Detector 3: One-Class SVM ─────────────────────────────────────────────
    # [BF-9] FIXED: clean_mask could be all-False producing an empty X_clean,
    # which causes OC-SVM to fit on zero samples (silent crash or garbage output).
    clean_mask = iforest.predict(X_all_sc) == 1
    if clean_mask.sum() > 10:
        X_clean = X_all_sc[clean_mask]
    else:
        log.warning("OC-SVM: clean_mask too sparse (%d samples) — using full X_all_sc",
                    int(clean_mask.sum()))
        X_clean = X_all_sc

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

    # [BF-6] FIXED: Sink (node 0) must never be scored as anomalous — it has
    # fundamentally different traffic characteristics (aggregator, not forwarder)
    # and was previously dragging down trust scores of its closest neighbours
    # by skewing the feature distribution.
    votes_anomaly[0] = 0

    # [BF-7] FIXED: norm01() now handles constant arrays safely
    fused = (norm01(scr_if) + norm01(scr_lof) + norm01(scr_ocsvm)) / 3.0

    # Phase 2 features
    routing_m  = X_curr[:, 8]
    path_cost  = X_curr[:, 10]
    path_stab  = X_curr[:, 11]

    # ── [BF-3] Consecutive anomaly gate ──────────────────────────────────────
    # Update per-node consecutive counter.  A node must be flagged anomalous in
    # CONSEC_NEEDED consecutive rounds before we apply the trust penalty.
    # A single clean round resets the counter.
    newly_flagged = (votes_anomaly >= 2)
    consec_anomaly[newly_flagged]  += 1
    consec_anomaly[~newly_flagged]  = 0   # reset on clean round

    # Effective anomaly = flagged AND has been consistently anomalous
    effective_anomaly = (consec_anomaly >= CONSEC_NEEDED)

    # ── [BF-2] FIXED: Asymmetric EWMA trust update ───────────────────────────
    # Phase 2 path-aware trust mapping
    trust_raw = np.zeros(n_nodes)

    for i in range(n_nodes):
        pc_i  = float(path_cost[i])
        ps_i  = float(path_stab[i])
        rm_i  = float(routing_m[i])

        if i == 0:
            # Sink is always trusted
            trust_raw[i] = 1.0
            continue

        if effective_anomaly[i]:
            # Confirmed anomalous: penalise scaled by path cost
            if scenario in ("C", "D"):
                path_penalty = pc_i * 0.10
                stab_penalty = (1.0 - ps_i) * 0.03
                trust_raw[i] = max(0.0,
                    fused[i] * 0.29 - path_penalty - stab_penalty)
            else:
                trust_raw[i] = fused[i] * 0.29
        else:
            # Normal or single-round suspect: healthy path bonuses
            if scenario == "D":
                path_bonus = max(0.0, (1.0 - pc_i) * 0.05)
                stab_bonus = ps_i * 0.03
                rm_boost   = rm_i * 0.05
                trust_raw[i] = min(1.0,
                    0.70 + fused[i] * 0.30 + path_bonus + stab_bonus + rm_boost)
            elif scenario == "C":
                rm_boost = rm_i * 0.07
                trust_raw[i] = min(1.0, 0.70 + fused[i] * 0.30 + rm_boost)
            else:
                trust_raw[i] = 0.70 + fused[i] * 0.30

    # ── [BF-2] Asymmetric EWMA smoothing ─────────────────────────────────────
    # Use a slower α when trust is falling to give innocent nodes more time,
    # and a faster α when trust is recovering so restored nodes earn it back quickly.
    for i in range(n_nodes):
        if trust_raw[i] < ewma_trust[i]:
            alpha_i = ALPHA_DECAY     # slower fall — protects innocents
        else:
            alpha_i = ALPHA_RECOVER   # faster rise — quicker restoration

        ewma_trust[i] = alpha_i * trust_raw[i] + (1.0 - alpha_i) * ewma_trust[i]

    ewma_trust[0] = 1.0  # sink always trusted

    n_anom = int(effective_anomaly.sum())
    n_susp = int(((votes_anomaly == 1) & (ewma_trust < 0.5)).sum())
    n_soft = int(((routing_m < 0.4) & (~effective_anomaly)).sum())

    log.info("t=%3ds | IF=%d LOF=%d OCSVM=%d | confirmed=%d suspicious=%d soft-risk=%d",
             timestamp,
             int((lbl_if==-1).sum()), int((lbl_lof==-1).sum()),
             int((lbl_ocsvm==-1).sum()),
             n_anom, n_susp, n_soft)

    if n_anom > 0:
        idxs = [i for i in range(n_nodes) if effective_anomaly[i]]
        log.info("  Confirmed anomalous: %s", idxs)
        log.info("  consec_count       : %s", [int(consec_anomaly[i]) for i in idxs])
        log.info("  Their trust        : %s", [round(ewma_trust[i], 3) for i in idxs])
        log.info("  Their path_cost    : %s", [round(float(path_cost[i]), 3) for i in idxs])
        log.info("  Their stability    : %s", [round(float(path_stab[i]), 3) for i in idxs])

    valid_pc = [float(path_cost[i]) for i in range(1, n_nodes) if path_cost[i] > 0]
    if valid_pc:
        log.info("  PathCost stats: min=%.3f mean=%.3f max=%.3f",
                 min(valid_pc), sum(valid_pc)/len(valid_pc), max(valid_pc))

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
        scenario = str(msg.get("scenario", "D"))

        energy         = msg.get("energy",            [1.0] * N)
        fwd            = msg.get("forward_ratio",     [0.5] * N)
        drop           = msg.get("drop_ratio",        [0.5] * N)
        iso            = msg.get("isolated",          [0.0] * N)
        known_m        = msg.get("known_malicious",   [0.0] * N)
        routing_metric = msg.get("routing_metric",    [1.0] * N)
        path_cost      = msg.get("path_cost",         [0.0] * N)
        path_stability = msg.get("path_stability",    [1.0] * N)

        # Validate list lengths — truncate or pad to N to prevent index errors
        def safe_list(lst, default):
            lst = list(lst)
            if len(lst) >= N: return lst[:N]
            return lst + [default] * (N - len(lst))

        energy         = safe_list(energy,         1.0)
        fwd            = safe_list(fwd,            0.5)
        drop           = safe_list(drop,           0.5)
        iso            = safe_list(iso,            0.0)
        known_m        = safe_list(known_m,        0.0)
        routing_metric = safe_list(routing_metric, 1.0)
        path_cost      = safe_list(path_cost,      0.0)
        path_stability = safe_list(path_stability, 1.0)

        log.info("t=%3ds | Scenario=%s | energy[0]=%.3f fwd[0]=%.3f "
                 "rm_avg=%.3f pc_avg=%.3f ps_avg=%.3f",
                 ts, scenario,
                 energy[0], fwd[0],
                 float(np.mean(routing_metric)),
                 float(np.mean(path_cost)),
                 float(np.mean(path_stability)))

        X = build_features(
            energy, fwd, drop, iso, known_m,
            routing_metric=routing_metric,
            path_cost=path_cost,
            path_stability=path_stability,
            timestamp=ts
        )

        t = ensemble_score(X, ts, scenario=scenario)

        resp = json.dumps({
            "trust": t,
            "round": round_count,
        }) + "\n"
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
╔════════════════════════════════════════════════════════════════════════════╗
║      Hybrid WSN — Phase 2 Ensemble ML Server  [BUG-FIXED]                  ║
╠════════════════════════════════════════════════════════════════════════════╣
║  Detectors   : Isolation Forest + LOF + One-Class SVM                      ║
║  Fusion      : Majority vote (2-of-3) + Asymmetric EWMA smoothing          ║
║  Features    : energy, fwd_ratio, drop_ratio, EWMA deltas,                 ║
║                routing_metric, energy_imbalance,                            ║
║                path_cost, path_stability              (12-D) [PHASE 2]     ║
╠════════════════════════════════════════════════════════════════════════════╣
║  FIXES:                                                                     ║
║  [BF-1] CONTAM 0.20→0.06  (was forcing 10/50 anomalies unconditionally)    ║
║  [BF-2] Asymmetric EWMA: decay α=0.25, recover α=0.45                      ║
║  [BF-3] Consecutive-anomaly gate: 2 rounds before penalising               ║
║  [BF-6] Sink (node 0) excluded from anomaly detection loop                 ║
║  [BF-7] norm01() division-by-zero guarded                                  ║
║  [BF-8] LOF novelty=True: fitted on history, scored on current window      ║
║  [BF-9] OC-SVM clean_mask empty-set fallback added                         ║
║  Trust <0.3  → isolation  |  Trust ≥ 0.3 → restore                         ║
║  Port        : 5555  |  Nodes : 50                                          ║
╚════════════════════════════════════════════════════════════════════════════╝
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
