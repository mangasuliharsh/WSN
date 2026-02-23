# 🔐 Hybrid ML-Driven Secure Energy-Aware Routing Framework for Wireless Sensor Networks

> A Closed-Loop, Multi-Objective Secure Routing Architecture for Wireless Sensor Networks (WSNs)  
> Inspired by Priyadarshi (2024), Wireless Networks 30:2647–2673

---

## 📌 Overview

This project implements a **Hybrid Real-Time Closed-Loop Secure Wireless Sensor Network (WSN)** using:

- 📡 ns-3 network simulation
- 🤖 Python-based ML anomaly detection server
- 🔋 Energy-aware adaptive routing
- 🛡 Trust-based malicious node isolation
- 📊 Multi-scenario evaluation framework

The system integrates **security, energy balancing, and adaptive routing** into a unified architecture for resilient WSN operation under repeated adversarial conditions.

---

## 🎯 Project Objectives

1. Detect malicious behavior in WSNs using unsupervised ML.
2. Reduce false positive isolation events.
3. Improve energy load balancing across nodes.
4. Preserve routing topology under sustained attacks.
5. Maintain acceptable Packet Delivery Ratio (PDR) under adversarial conditions.

---

# 🧠 System Architecture

          ┌──────────────────────┐
          │   ns-3 Simulation    │
          │  (WSN + AODV + PHY)  │
          └──────────┬───────────┘
                     │
                     │ Feature Extraction
                     │ (Energy, Fwd Ratio, Drop Ratio)
                     ▼
          ┌──────────────────────┐
          │   Python ML Server   │
          │ Ensemble Anomaly ML  │
          └──────────┬───────────┘
                     │
                     │ Trust Scores
                     ▼
          ┌──────────────────────┐
          │  Adaptive Routing    │
          │ Trust + Energy Score │
          └──────────────────────┘


---

# ⚙️ Technologies Used

| Component | Technology |
|------------|------------|
| Network Simulation | ns-3 |
| Routing Protocol | AODV |
| ML Framework | scikit-learn |
| ML Algorithms | Isolation Forest, LOF, One-Class SVM |
| IPC | TCP Socket (ns-3 ↔ Python) |
| Visualization | NetAnim |
| Metrics Logging | CSV + FlowMonitor |

---

# 🚀 Implemented Phases

---

## ✅ Phase 0 – Core Simulation Infrastructure

- 50-node 802.11b Ad Hoc network  
- Grid topology  
- 150J per node energy model  
- Periodic attack injection cycle  
- FlowMonitor integration  
- NetAnim visualization  
- Performance logging (600s simulation window)  

---

## ✅ Phase 1 – Trust-Based Secure Routing

- Ensemble anomaly detection:
  - Isolation Forest  
  - Local Outlier Factor  
  - One-Class SVM  
- Majority vote fusion  
- EWMA trust smoothing  
- Hard isolation at trust < 0.3  
- Dynamic node restoration  
- Isolation event tracking  

---

## ✅ Phase 1.5 – Energy-Aware Adaptive Routing

Composite routing metric:
routingMetric = 0.7 * trust + 0.3 * energy_fraction

- Soft avoidance of low-score nodes  
- Energy imbalance index (standard deviation)  
- Reduced false positive isolations  
- Improved load balancing  
- Routing metric preservation  

---

# 📊 Experimental Evaluation (600s Simulation)

Three scenarios were tested:

| Scenario | Description |
|----------|-------------|
| A | Pure AODV (No Security) |
| B | Trust-Only Isolation |
| C | Trust + Energy Weighted Routing |

### 🔥 Key Results

- ✅ 51.8% reduction in energy imbalance  
- ✅ 53.6% reduction in isolation events  
- ✅ Improved routing survivability  
- ✅ +1.22% PDR recovery over trust-only  

---

# 📈 Performance Metrics Collected

- Packet Delivery Ratio (PDR)  
- Average End-to-End Delay  
- Average Residual Energy  
- Energy Standard Deviation (Load Balancing Index)  
- Isolation Events  
- Alive Nodes Count  
- Routing Metric Trend  

---

# 🔄 Upcoming Work (Phase 2)

Phase 2 will integrate the composite routing metric **directly into AODV route selection**, replacing hop-count-only logic with a multi-objective cost function.

Planned features:

- Modify AODV RREQ/RREP handling  
- Extend routing table entries  
- Multi-objective path cost computation  
- Route re-evaluation trigger  
- Control overhead measurement  

---

# 🛠 How to Run

## 1️⃣ Start ML Server
python3 ml_server.py

## 2️⃣ Run ns-3 Simulation
./ns3 run hybrid_wsn_secure


## 3️⃣ View Results

- `results/performance.csv`
- `results/flowmonitor.xml`
- `results/animation.xml` (open with NetAnim)

---

# 📚 Research Context

This work extends:

> Priyadarshi (2024), *Secure Routing in Wireless Sensor Networks*, Wireless Networks 30:2647–2673

By incorporating:

- Multi-objective routing metrics
- Energy-aware load balancing
- Reduced false positive anomaly detection
- Closed-loop adaptive routing control

---

# 🏆 Project Status

| Phase | Status |
|-------|--------|
| Phase 0 | ✅ Completed |
| Phase 1 | ✅ Completed |
| Phase 1.5 | ✅ Completed |
| Phase 2 | 🚧 In Progress |
| Phase 3 | 🔮 Planned |

---

# 👨‍💻 Author

Harshavardhan Balaraj Mangasuli  
Information Science & Engineering  
B.E. Final Year Project  

---

# 📜 License

This project is for academic and research purposes.
