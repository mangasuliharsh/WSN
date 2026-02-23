#!/usr/bin/env bash
# =============================================================================
# Phase 1 Evaluation Runner
# Runs Scenarios A, B, C sequentially and collects results
# =============================================================================
set -e

echo "======================================================"
echo "  Phase 1: Trust + Energy Adaptive Routing Evaluation"
echo "======================================================"

mkdir -p results

# Start ML server in background (used by Scenarios B and C)
echo ""
echo "[*] Starting ML server on port 5555..."
python3 ml/ml_server.py &
ML_PID=$!
sleep 2
echo "[*] ML server PID: $ML_PID"

run_scenario() {
    local S=$1
    echo ""
    echo "------------------------------------------------------"
    echo "  Running Scenario $S..."
    echo "------------------------------------------------------"
    # Compile (adjust path to your ns-3 build as needed)
    ./ns3 run "scratch/hybrid_wsn_secure" -- --scenario="$S"
    echo "  → Results saved to results/performance_${S}.csv"
    echo "  → Animation saved to results/animation_${S}.xml"
    echo "  → FlowMonitor saved to results/flowmonitor_${S}.xml"
}

# Scenario A: Baseline AODV (ML not used, but server is running — harmless)
run_scenario A

# Scenario B: Trust only
run_scenario B

# Scenario C: Trust + Energy (Phase 1 full system)
run_scenario C

# Stop ML server
echo ""
echo "[*] Stopping ML server..."
kill $ML_PID 2>/dev/null || true

echo ""
echo "======================================================"
echo "  All scenarios complete. Results in ./results/"
echo "======================================================"
echo ""
echo "Files:"
ls -lh results/performance_*.csv 2>/dev/null || true
echo ""
echo "To compare results:"
python3 ml/compare_results.py
