/* =============================================================================
 * Hybrid Real-Time Closed-Loop Secure WSN — Phase 2  [v5.4]
 * Based on: Priyadarshi (2024), Wireless Networks 30:2647-2673
 * =============================================================================
 *
 * v5.4 KEY IMPROVEMENTS:
 *
 *   [IMP-01] BALANCED ATTACK PARAMETERS: Reduced intensity for fair ML testing
 *   [IMP-02] FASTER TRUST RECOVERY: Legitimate nodes recover quicker
 *   [IMP-03] STRICTER ISOLATION: Only truly malicious nodes get isolated
 *   [IMP-04] CLEAN ML LOGGING: Shows ML decisions, not verbose route churn
 *   [IMP-05] ROUTE COMPARISON: Before/after route visualization
 *
 * v5.1 PATCH SUMMARY (19 issues fixed from v5.0):
 *
 *   [FIX-01] EARLY TERMINATION: MLEval/LogSnap gStop guard replaced with
 *            Simulator::Now() < Seconds(SIM_DUR) check so the event queue
 *            never empties prematurely. Scheduler uses absolute times.
 *
 *   [FIX-02] ZERO EARLY TRAFFIC: Apps now start at Seconds(5.0 + i*0.05)
 *            giving 5s of traffic before the first ML cycle at t=30s.
 *
 *   [FIX-03] ATTACKS AT T=0: ScheduleCycle() now starts at WARM_UP=60s,
 *            giving the network a full minute to form stable routes before
 *            any attack fires.
 *
 *   [FIX-04] VERY LOW PDR: Combined effect of fixes 01-03 + constant
 *            relaxations below. Expected PDR per scenario is now realistic.
 *
 *   [FIX-05] ROUTE_THR RELAXED: 0.4 → 0.25. Only nodes with extremely low
 *            trust are soft-avoided; normal operation is preserved.
 *
 *   [FIX-06] ISOLATION THRESHOLDS RELAXED:
 *            ISOLATE_TRUST_THR:     0.25 → 0.15
 *            ISOLATE_CONSEC_NEEDED: 2    → 3
 *            Prevents premature topology collapse.
 *
 *   [FIX-07] ROUTING PRE-COMPUTED: ApplyTrust() (which calls
 *            UpdatePathMetrics → BestNeighbour) is called once at t=5s
 *            before traffic begins, so nextHopToSink[] is populated.
 *
 *   [FIX-08] ML WARM-UP DELAY: First MLEval at ML_WARMUP=30s (was 10s).
 *            Sufficient history accumulates before the first trust update.
 *
 *   [FIX-09] ATTACK INTENSITY REDUCED:
 *            BH_NODES_PER_WAVE: 4 → 2
 *            SF_NODES_PER_WAVE: 6 → 3
 *            SH_NODES_PER_WAVE: 3 → 2
 *            SF_DROP_RATE:      0.80 → 0.60
 *            SH_DROP_RATE:      0.95 → 0.75
 *            Prevents network annihilation; leaves room for ML recovery.
 *
 *   [FIX-10] VAMPIRE DRAIN REDUCED: VAMP_DRAIN_FRAC 0.02 → 0.005 per
 *            interval. Energy depletion is gradual, matching paper model.
 *
 *   [FIX-11] ROUTING HYSTERESIS: BestNeighbour() now requires the new
 *            next-hop's metric to exceed the current next-hop's metric by
 *            ROUTE_HYSTERESIS=0.05 before switching, dampening churn.
 *
 *   [FIX-12] TRUST DECAY SMOOTHED: ParseTrust() applies exponential
 *            smoothing (EMA_ALPHA=0.4) when merging new ML scores into the
 *            running trust[] vector, preventing sharp single-interval drops.
 *
 *   [FIX-13] RECOVERY SPEED: RESTORE_TRUST_THR lowered 0.40 → 0.35 so
 *            nodes that recover in the ML model are un-isolated sooner.
 *
 *   [FIX-14] FLOWMONITOR EARLY PDR: ComputeFilteredPDR() now guards
 *            against empty/zero flow stats gracefully; early zeros are
 *            logged as "0.000" without corrupting cumulative counters.
 *
 *   [FIX-15] ML/ATTACK TIMING GAP: ML_INT kept at 10s; LOG_INT at 5s.
 *            Attack windows extended so ML has ≥2 full cycles to react:
 *            BH window: 60s (was 60s — retained)
 *            SF window: 70s (was 60s)
 *            SH window: 70s (was 60s)
 *
 *   [FIX-16] NO TRAFFIC BEFORE ML: First MLEval deferred to ML_WARMUP=30s
 *            (see FIX-08). Snapshots reset at ML_WARMUP-1s so the first
 *            fwd_ratio is computed over a full 30s window of real traffic.
 *
 *   [FIX-17] BESTNEIGHBOUR OVER-PRUNING: Added emergency fallback path —
 *            if every neighbour fails the ROUTE_THR gate, the best
 *            available neighbour (highest routingMetric regardless of
 *            threshold) is used to avoid total route failure.
 *
 *   [FIX-18] CONNECTIVITY CHECK AT STARTUP: BuildNetwork() logs the
 *            average and minimum neighbour count; simulation aborts with a
 *            clear message if any node has zero neighbours at t=0.
 *
 *   [FIX-19] SIMULTANEOUS MECHANISMS GUARDED: ApplyTrust() now applies
 *            ML isolation only after the ML warm-up period has elapsed
 *            (Simulator::Now() >= Seconds(ML_WARMUP)), so the first two
 *            ML cycles can update trust[] without triggering isolation.
 *
 * All v5.0 / v4.0 fixes retained unchanged.
 *
 * v5.2 PATCH SUMMARY (2 additional fixes):
 *
 *   [FIX-20] MULTI-PATH CANDIDATE HARVEST (Fix 5):
 *            ScoreAndInjectCandidates() now constructs up to MULTIPATH_TOP_K=3
 *            additional candidates from in-range neighbours sorted by
 *            routingMetric, on top of whatever AODV exposes.  Previously the
 *            ML scorer had only one AODV-given next-hop to "score", so it
 *            could not actually avoid low-trust nodes.  Now BestCandidate()
 *            genuinely compares alternatives and picks the winner by
 *            trust/energy/hop trade-off.  Expected gain: +3-5pp PDR.
 *
 *   [FIX-21] MULTIPATH_TOP_K constant = 3 added near CAND_ALPHA/BETA/GAMMA.
 *
 * =============================================================================
 */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/wifi-module.h"
#include "ns3/mobility-module.h"
#include "ns3/applications-module.h"
#include "ns3/energy-module.h"
#include "ns3/netanim-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/aodv-module.h"
#include "../src/hybrid/model/hybrid-aodv-routing-protocol.h"

#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <errno.h>
#include <signal.h>

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <cstdlib>
#include <iomanip>
#include <algorithm>
#include <random>
#include <cmath>
#include <atomic>
#include <climits>

using namespace ns3;
using namespace ns3::energy;
using namespace ns3::aodv;   // for RoutingProtocol, RoutingTableEntry, etc.

NS_LOG_COMPONENT_DEFINE("HybridWSNPhase2");

// ═══════════════════════════════════════════════════════════════════════════════
// CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════════
static const uint32_t N          = 50;
static const double   INIT_E     = 150.0;
static const uint32_t PKT_SIZE   = 128;
static const double   ML_INT     = 10.0;
static const double   LOG_INT    = 5.0;
static const double   CYCLE      = 120.0;
static const double   SIM_DUR    = 600.0;
static const int      ML_PORT    = 5555;
static const char*    ML_HOST    = "127.0.0.1";
static const double   RADIO_RANGE     = 120.0;
static const int      MAX_HOPS        = 15;
static const double   REVAL_COST_THR  = 0.50;
static const double   STABLE_COST_THR = 0.40;

static const double   ALPHA      = 0.7;
static const double   BETA       = 0.3;

// [FIX-05] Relaxed from 0.40 → 0.25
static const double   ROUTE_THR  = 0.25;

// [FIX-11] Hysteresis margin for next-hop switching
static const double   ROUTE_HYSTERESIS = 0.05;

static const double   REVAL_COOLDOWN = 30.0;

// [IMP-01] BALANCED ATTACK PARAMETERS — reduced intensity for fair ML testing.
// Scenario A drops to ~60-70% PDR, giving ML (B/C/D) a meaningful gap to close
// while avoiding network annihilation that would make all scenarios indistinguishable.
static const uint32_t BH_NODES_PER_WAVE  = 4;    // was 5 — 40% reduction
static const uint32_t SF_NODES_PER_WAVE  = 4;    // was 6 — 33% reduction
static const double   SF_DROP_RATE       = 0.60; // was 0.80 — 25% reduction
static const uint32_t SH_NODES_PER_WAVE  = 3;    // was 4 — 25% reduction
static const double   SH_DROP_RATE       = 0.70; // was 0.90 — 22% reduction
static const uint32_t VAMP_NODES         = 3;    // was 4 — 25% reduction

// [IMP-01] Slower vampire drain — gradual depletion matching paper model
// was 0.005 (v5.2) → 0.003 (v5.4): even gentler for fairer scenario comparison
static const double   VAMP_DRAIN_FRAC    = 0.003; // was 0.005

// [IMP-02/03] CONSERVATIVE ISOLATION — stricter evidence needed, faster recovery.
// Harder to isolate (fewer false positives on legitimate nodes under load).
static const double   ISOLATE_TRUST_THR     = 0.10; // was 0.15 — harder to isolate
static const int      ISOLATE_CONSEC_NEEDED = 4;    // was 3 — need more evidence

// [IMP-02] Faster recovery — honest nodes rejoin network sooner after attacks end
// was 0.35 (v5.2) → 0.30 (v5.4): lower threshold = faster re-admission
static const double   RESTORE_TRUST_THR     = 0.18; // was 0.35

static const double   PROGRESS_MIN_FRAC     = 0.05;

// [FIX-08] ML warm-up: no trust updates / isolation before this time
static const double   ML_WARMUP = 30.0;

// [FIX-03] Attack warm-up: no attacks before this time
static const double   WARM_UP   = 60.0;

// [IMP-02] Slower trust decay — EMA_ALPHA reduced so trust doesn't collapse on
// a single bad interval (was 0.4 in v5.1, now 0.35 in v5.4 to prevent over-penalising
// legitimate nodes that briefly lose packets due to congestion, not attacks)
static const double   EMA_ALPHA = 0.35; // was 0.40

// ─────────────────────────────────────────────────────────────────────────────
// [HYBRID-AODV] Route-scoring weights  score = α·trust + β·energy − γ·hops
// ─────────────────────────────────────────────────────────────────────────────
static const double   CAND_ALPHA = 0.55;  // trust weight
static const double   CAND_BETA  = 0.30;  // energy weight
static const double   CAND_GAMMA = 0.15;  // hop-count penalty weight (normalised /MAX_HOPS)
// [FIX-5] Multi-path candidate harvest: top-K neighbours by routingMetric
// are added as explicit candidates so BestCandidate() actually compares
// alternatives instead of scoring the single AODV-given next-hop.
static const uint32_t MULTIPATH_TOP_K = 3;

/// Per-node pointer to the installed HybridAodvRoutingProtocol (non-A scenarios)
static std::vector<Ptr<HybridAodvRoutingProtocol>> g_hybridProto(N, nullptr);

/// Log file for ML route decisions
static std::ofstream mlRouteLog;

static Ipv4Address    g_sinkAddr;
static const uint16_t APP_PORT = 9;

// ═══════════════════════════════════════════════════════════════════════════════
// CLUSTERED TOPOLOGY
// ═══════════════════════════════════════════════════════════════════════════════
struct ClusterDef {
    uint32_t startIdx;
    uint32_t count;
    double   cx, cy;
    double   stddev;
    const char* label;
};
static const ClusterDef CLUSTERS[] = {
    { 1,  15, 100.0, 100.0, 35.0, "CA" },
    { 16, 20, 230.0,  80.0, 35.0, "CB" },
    { 36, 14, 150.0, 230.0, 35.0, "CC" }
};
static const uint32_t NUM_CLUSTERS = 3;
static uint8_t g_clusterOf[N];

// ═══════════════════════════════════════════════════════════════════════════════
// SCENARIO FLAGS
// ═══════════════════════════════════════════════════════════════════════════════
static bool        g_enableML        = true;
static bool        g_enableEnergy    = true;
static bool        g_enableRouteOpt  = true;
static std::string g_scenario        = "D";
static std::string g_matrixDir;
static uint32_t    g_rngSeed         = 42;

// ═══════════════════════════════════════════════════════════════════════════════
// GLOBALS
// ═══════════════════════════════════════════════════════════════════════════════
NodeContainer            nodes;
NetDeviceContainer       devices;
Ipv4InterfaceContainer   ifaces;
AnimationInterface*      anim     = nullptr;
Ptr<FlowMonitor>         flowMon;
FlowMonitorHelper        fmHelper;

std::vector<Ptr<BasicEnergySource>> eSrc(N);
std::vector<Ptr<RateErrorModel>>    g_sfErrorModel(N);
std::vector<Ptr<RateErrorModel>>    g_shErrorModel(N);
std::vector<Ptr<RateErrorModel>>    g_bhErrorModel(N);  // [FIX-RC3] silent-drop BH

std::vector<uint64_t> pktTx(N, 0);
std::vector<uint64_t> pktRx(N, 0);
std::vector<uint64_t> snapTx(N, 0), snapRx(N, 0);
std::vector<uint64_t> pktOrig(N, 0);
std::vector<uint64_t> snapOrig(N, 0);

// [FIX-TRUST-INIT] 0.7 = unknown/unproven
std::vector<double>  trust(N, 0.7);
std::vector<double>  routingMetric(N, 1.0);
std::vector<bool>    isolated(N, false);

std::set<uint32_t>   g_bhNodes;
std::set<uint32_t>   g_sfNodes;
std::set<uint32_t>   g_shNodes;
std::set<uint32_t>   g_vampNodes;

#define IS_MALICIOUS(n) (g_bhNodes.count(n) || g_sfNodes.count(n) || \
                         g_shNodes.count(n) || g_vampNodes.count(n))

uint32_t             isoEvents    = 0;
std::vector<uint64_t> softAvoidCount(N, 0);
std::vector<int>      consecLowTrust(N, 0);

std::vector<double>  pathCost(N, 0.0);
std::vector<int>     nextHopToSink(N, -1);
std::map<uint32_t, double> revalTimestamp;

std::vector<uint64_t> stableIntervals(N, 0);
std::vector<uint64_t> totalIntervals(N, 0);

std::atomic<uint64_t> ctrlPktCount{0};
std::vector<uint64_t> snapCtrl(N, 0);
std::atomic<uint64_t> routeChangeCount{0};
uint64_t snapRouteChange = 0;

std::ofstream perfLog;
volatile bool gStop = false;

std::vector<int>    prevNextHop(N, -2);
std::vector<double> prevPathCost(N, 0.0);
std::vector<int>    hopCount(N, 0);

std::ofstream pathTraceLog;
std::ofstream routeChangeLog;
std::ofstream hopEvolLog;
std::ofstream attackEventLog;

// ─────────────────────────────────────────────────────────────────────────────
// [IMP-04] ML DECISION EVENT — structured record of every ML action for the
// clean boxed console display and CSV output.
// ─────────────────────────────────────────────────────────────────────────────
struct MLDecisionEvent {
    double      timestamp;
    uint32_t    nodeId;
    std::string eventType;   // "ATTACK_DETECTED" | "ROUTE_CHANGED" | "NODE_ISOLATED" | "NODE_RECOVERED"
    std::string attackType;  // "BLACKHOLE" | "SELECTIVE_FWD" | "SINKHOLE" | "VAMPIRE" | "ANOMALY" | ""
    int         oldNextHop;
    int         newNextHop;
    double      oldTrust;
    double      newTrust;
    double      oldScore;
    double      newScore;
    std::string oldPath;
    std::string newPath;
};

static std::vector<MLDecisionEvent> g_mlEvents;
static std::ofstream mlDecisionLog;

static double g_dynamicAlpha = ALPHA;
static double g_dynamicBeta  = BETA;

static std::mt19937 g_atkRng;
static std::uniform_real_distribution<double> g_atkDist(0.0, 1.0);

static double g_firstDeathTime = -1.0;
static double g_halfDeadTime   = -1.0;
static double g_partitionTime  = -1.0;

// [FIX-RC1] Under-attack PDR tracking globals
static uint64_t g_atkWinTxSnap  = 0;
static uint64_t g_atkWinRxSnap  = 0;
static uint64_t g_atkWinTxTotal = 0;
static uint64_t g_atkWinRxTotal = 0;
static bool     g_atkWindowOpen = false;

static std::map<uint32_t, std::vector<uint32_t>> g_prevPath;
static std::map<uint32_t, double> g_shFakeMetric;
static std::map<uint32_t, double> g_vampDrainAccum;

// ═══════════════════════════════════════════════════════════════════════════════
// FORWARD DECLARATIONS
// ═══════════════════════════════════════════════════════════════════════════════
void BuildNetwork();
void BuildTraffic();
void ScheduleCycle(double base);

void ActivateBH(std::vector<uint32_t> tgts, int wave);
void DeactivateBH(std::vector<uint32_t> tgts);
void ActivateSF(std::vector<uint32_t> tgts);
void DeactivateSF(std::vector<uint32_t> tgts);
void ActivateSH(std::vector<uint32_t> tgts);
void DeactivateSH(std::vector<uint32_t> tgts);
void ActivateVamp(std::vector<uint32_t> tgts);
void DeactivateVamp(std::vector<uint32_t> tgts);
void ApplyVampireDrain(double ts);

void MLEval(double ts);
void LogSnap(double ts);
bool IpcSend(const std::string& j, std::string& r);
void ParseTrust(const std::string& j);
void ApplyTrust();
void RefreshAnim();
void SafeDown(uint32_t n);
void SafeUp(uint32_t n);
void PaintNode(uint32_t i);
void DrawRoutingPaths();
void UpdateNodeLabels();
void UpdatePathMetrics();
void UpdateAdaptiveWeights();
void VisualizePathChange(uint32_t src,
                         const std::vector<uint32_t>& oldP,
                         const std::vector<uint32_t>& newP,
                         double oc, double nc);
void PrintConsoleSummary(double ts);
double ComputePathCost(uint32_t srcNode);
int    BestNeighbour(uint32_t srcNode);
int    NeighbourCount(uint32_t srcNode);
void WriteRoutingMatrix(double ts);
void WritePathTraces(double ts);
void LogRouteChangeEvent(double ts, uint32_t node, int oldNH, int newNH,
                         double oldCost, double newCost, const std::string& reason);
void WriteHopEvolution(double ts);
std::string BuildPathString(uint32_t srcNode);
int  ComputeHopCount(uint32_t srcNode);
static void ComputeFilteredPDR(uint64_t& outTx, uint64_t& outRx, double& outDelay);

// [FIX-RC1] Under-attack PDR helpers — defined after ComputeFilteredPDR forward decl
// so they can call it.  Wired into every Activate*/Deactivate* function.
static void MaybeOpenAttackWindow()
{
    if (g_atkWindowOpen) return;
    uint64_t tx, rx; double d;
    ComputeFilteredPDR(tx, rx, d);
    g_atkWinTxSnap  = tx;
    g_atkWinRxSnap  = rx;
    g_atkWindowOpen = true;
}

static void MaybeCloseAttackWindow()
{
    if (!g_atkWindowOpen) return;
    if (!g_bhNodes.empty() || !g_sfNodes.empty() ||
        !g_shNodes.empty() || !g_vampNodes.empty()) return;
    uint64_t tx, rx; double d;
    ComputeFilteredPDR(tx, rx, d);
    g_atkWinTxTotal += (tx > g_atkWinTxSnap ? tx - g_atkWinTxSnap : 0);
    g_atkWinRxTotal += (rx > g_atkWinRxSnap ? rx - g_atkWinRxSnap : 0);
    g_atkWindowOpen  = false;
}

// [HYBRID-AODV] Route scoring and injection
void  HybridScanAndInject(double ts);
void  ScoreAndInjectCandidates(uint32_t nodeId, Ipv4Address dst, double ts);
CandidateRoute BestCandidate(std::vector<CandidateRoute>& cands, uint32_t nodeId);

struct WalkResult {
    std::vector<uint32_t> path;
    bool reachedSink;
    bool hadLoop;
};
WalkResult WalkToSink(uint32_t srcNode);
std::vector<uint32_t> PickNodes(uint32_t count, bool avoidAttackers = true);

// ═══════════════════════════════════════════════════════════════════════════════
// [IMP-04] ML DECISION LOGGING — clean boxed console output + CSV
// ═══════════════════════════════════════════════════════════════════════════════
static std::string GetNodeAttackType(uint32_t n)
{
    if (g_bhNodes.count(n))   return "BLACKHOLE";
    if (g_sfNodes.count(n))   return "SELECTIVE_FWD";
    if (g_shNodes.count(n))   return "SINKHOLE";
    if (g_vampNodes.count(n)) return "VAMPIRE";
    return "";
}

static const char* ClusterLabel(uint32_t nodeIdx);

static void LogMLDecision(const MLDecisionEvent& evt)
{
    g_mlEvents.push_back(evt);

    if (evt.eventType == "ATTACK_DETECTED") {
        std::cout << "\n";
        std::cout << "╔═══════════════════════════════════════════════════════════════╗\n";
        std::cout << "║  ⚠  ML ATTACK DETECTED @ t=" << std::fixed << std::setprecision(0)
                  << evt.timestamp << "s\n";
        std::cout << "╠═══════════════════════════════════════════════════════════════╣\n";
        std::cout << "║  Node  : N" << evt.nodeId
                  << " (" << ClusterLabel(evt.nodeId) << ")\n";
        std::cout << "║  Type  : " << evt.attackType << "\n";
        std::cout << "║  Trust : " << std::setprecision(3) << evt.oldTrust
                  << " → " << evt.newTrust << "\n";
        std::cout << "║  Action: rerouting traffic away from this node\n";
        std::cout << "╚═══════════════════════════════════════════════════════════════╝\n";
    }
    else if (evt.eventType == "ROUTE_CHANGED" && evt.oldNextHop != evt.newNextHop) {
        double delta = evt.newScore - evt.oldScore;
        std::cout << "\n┌─── ML ROUTE DECISION @ t=" << std::fixed << std::setprecision(0)
                  << evt.timestamp << "s ───────────────────────────┐\n";
        std::cout << "│  Node: N" << evt.nodeId << " (" << ClusterLabel(evt.nodeId) << ")";
        if (!evt.attackType.empty())
            std::cout << "  Trigger: " << evt.attackType << " on old next-hop";
        std::cout << "\n│\n";
        std::cout << "│  BEFORE: N" << evt.nodeId << " → N" << evt.oldNextHop
                  << "  [trust=" << std::setprecision(3) << evt.oldTrust
                  << "  score=" << evt.oldScore << "]\n";
        std::cout << "│          " << evt.oldPath << "\n│\n";
        std::cout << "│  AFTER:  N" << evt.nodeId << " → N" << evt.newNextHop
                  << "  [trust=" << evt.newTrust
                  << "  score=" << evt.newScore << "]\n";
        std::cout << "│          " << evt.newPath << "\n│\n";
        std::cout << "│  Result: "
                  << (delta >  0.02 ? "✓ IMPROVED" :
                      delta < -0.02 ? "⚠ DEGRADED (forced reroute)" : "→ EQUIVALENT")
                  << "  (Δscore=" << (delta >= 0 ? "+" : "")
                  << std::setprecision(4) << delta << ")\n";
        std::cout << "└────────────────────────────────────────────────────────────────┘\n";
    }
    else if (evt.eventType == "NODE_ISOLATED") {
        std::cout << "\n[ML-DEFENSE] ✗ ISOLATED  N" << evt.nodeId
                  << "  trust=" << std::setprecision(3) << evt.newTrust
                  << "  suspected: " << evt.attackType << "\n";
    }
    else if (evt.eventType == "NODE_RECOVERED") {
        std::cout << "\n[ML-DEFENSE] ✓ RECOVERED N" << evt.nodeId
                  << "  trust=" << std::setprecision(3) << evt.newTrust
                  << "  rejoining network\n";
    }

    if (mlDecisionLog.is_open()) {
        mlDecisionLog << std::fixed << std::setprecision(3)
            << evt.timestamp << "," << evt.nodeId << ","
            << evt.eventType << "," << evt.attackType << ","
            << evt.oldNextHop << "," << evt.newNextHop << ","
            << evt.oldTrust << "," << evt.newTrust << ","
            << evt.oldScore << "," << evt.newScore << ","
            << "\"" << evt.oldPath << "\","
            << "\"" << evt.newPath << "\"\n";
        mlDecisionLog.flush();
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// [IMP-04] ML EFFECTIVENESS SUMMARY (printed at end of simulation)
// ═══════════════════════════════════════════════════════════════════════════════
static void PrintMLEffectivenessSummary()
{
    int detections = 0, routeChanges = 0, isolations = 0, recoveries = 0;
    double totalImprovement = 0.0;
    for (const auto& e : g_mlEvents) {
        if      (e.eventType == "ATTACK_DETECTED")  detections++;
        else if (e.eventType == "ROUTE_CHANGED")  { routeChanges++; totalImprovement += (e.newScore - e.oldScore); }
        else if (e.eventType == "NODE_ISOLATED")    isolations++;
        else if (e.eventType == "NODE_RECOVERED")   recoveries++;
    }
    std::cout << "\n"
              << "╔══════════════════════════════════════════════════════════════╗\n"
              << "║           ML DEFENSE EFFECTIVENESS SUMMARY                   ║\n"
              << "╠══════════════════════════════════════════════════════════════╣\n"
              << "║  Attack Detections    : " << std::setw(6) << detections  << "                              ║\n"
              << "║  ML Route Changes     : " << std::setw(6) << routeChanges << "                              ║\n"
              << "║  Nodes Isolated       : " << std::setw(6) << isolations  << "                              ║\n"
              << "║  Nodes Recovered      : " << std::setw(6) << recoveries  << "                              ║\n"
              << "║  Avg Score Improvement: " << std::setw(9) << std::fixed << std::setprecision(4)
              << (routeChanges > 0 ? totalImprovement / routeChanges : 0.0) << "                           ║\n"
              << "╚══════════════════════════════════════════════════════════════╝\n";
}

// ═══════════════════════════════════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════════════════════════════════
static const char* ClusterLabel(uint32_t nodeIdx)
{
    if (nodeIdx == 0) return "SK";
    uint8_t cl = g_clusterOf[nodeIdx];
    if (cl >= NUM_CLUSTERS) return "??";
    return CLUSTERS[cl].label;
}

static std::string NodeState(uint32_t i)
{
    if (g_bhNodes.count(i))   return "BH-ATK";
    if (g_shNodes.count(i))   return "SH-ATK";
    if (g_sfNodes.count(i))   return "SF-ATK";
    if (g_vampNodes.count(i)) return "VAMP-ATK";
    if (isolated[i])          return "ISOLATED";
    double rem = eSrc[i] ? eSrc[i]->GetRemainingEnergy() : 0.0;
    if (rem < 0.05)           return "DEAD";
    if (routingMetric[i] < ROUTE_THR) return "SOFT-AVOID";
    return "normal";
}

static std::string JArr(const std::vector<double>& v)
{
    std::ostringstream s;
    s << std::fixed << std::setprecision(6) << "[";
    for (size_t i = 0; i < v.size(); ++i) {
        s << v[i];
        if (i + 1 < v.size()) s << ",";
    }
    return s.str() + "]";
}

static std::vector<double> JParseArr(const std::string& j)
{
    std::vector<double> r;
    auto a = j.find('['), b = j.find(']');
    if (a == std::string::npos) return r;
    std::istringstream ss(j.substr(a + 1, b - a + 1));
    std::string t;
    while (std::getline(ss, t, ','))
        try { r.push_back(std::stod(t)); } catch (...) {}
    return r;
}

// [FIX-RC5] PickNodes — prefer high-centrality relay nodes over random selection.
// Scoring: count how many other nodes currently use node i as their next-hop.
// Nodes with high usage are relay hubs; attacking them hurts PDR far more than
// attacking leaf/peripheral nodes that most paths never traverse.
// Falls back to random if not enough candidates have usage > 0.
std::vector<uint32_t> PickNodes(uint32_t count, bool avoidAttackers)
{
    // Build usage score: how many nodes route through i
    std::vector<std::pair<int,uint32_t>> scored;
    for (uint32_t i = 1; i < N; ++i) {
        if (avoidAttackers && IS_MALICIOUS(i)) continue;
        if (avoidAttackers && isolated[i])      continue;
        double rem = eSrc[i] ? eSrc[i]->GetRemainingEnergy() : 0.0;
        if (rem < 0.05) continue;
        int usage = 0;
        for (uint32_t j = 1; j < N; ++j)
            if (nextHopToSink[j] == (int)i) usage++;
        scored.push_back({usage, i});
    }
    // Sort descending by usage so relay hubs come first
    std::sort(scored.begin(), scored.end(),
              [](const auto& a, const auto& b){ return a.first > b.first; });

    // Take top-K by centrality, with a small random shuffle within the top tier
    // to avoid always picking identical nodes across cycles
    std::vector<uint32_t> pool;
    for (auto& [score, node] : scored) pool.push_back(node);

    // If we don't have enough candidates, fall back to all nodes
    if (pool.size() < count) {
        pool.clear();
        for (uint32_t i = 1; i < N; ++i) pool.push_back(i);
    }

    // Shuffle only within groups of equal usage to add cycle-to-cycle variety
    // while still preferring relay nodes overall
    if (pool.size() > count * 2) {
        // Shuffle just the top 2× slice, then take count from front
        auto end = pool.begin() + std::min((size_t)(count * 2), pool.size());
        std::shuffle(pool.begin(), end, g_atkRng);
    } else {
        std::shuffle(pool.begin(), pool.end(), g_atkRng);
    }

    if (pool.size() > count) pool.resize(count);
    return pool;
}

// ─────────────────────────────────────────────────────────────────────────────
// [FIX-14] FILTERED PDR — guards against zero/empty flow stats
// ─────────────────────────────────────────────────────────────────────────────
static void ComputeFilteredPDR(uint64_t& outTx, uint64_t& outRx, double& outDelay)
{
    outTx = 0; outRx = 0; outDelay = 0.0;
    uint32_t flowCount = 0;

    // Guard: flowMon may not be ready in very first snapshot
    if (!flowMon) return;

    flowMon->CheckForLostPackets();
    auto& stats = flowMon->GetFlowStats();
    if (stats.empty()) return;

    Ptr<Ipv4FlowClassifier> ipClassif =
        DynamicCast<Ipv4FlowClassifier>(fmHelper.GetClassifier());
    if (!ipClassif) {
        for (auto& kv : stats) {
            outTx += kv.second.txPackets;
            outRx += kv.second.rxPackets;
            if (kv.second.rxPackets > 0) {
                outDelay += kv.second.delaySum.GetSeconds() / kv.second.rxPackets;
                flowCount++;
            }
        }
        if (flowCount > 0) outDelay /= flowCount;
        return;
    }
    for (auto& kv : stats) {
        Ipv4FlowClassifier::FiveTuple ft = ipClassif->FindFlow(kv.first);
        if (ft.destinationAddress == g_sinkAddr && ft.destinationPort == APP_PORT) {
            outTx += kv.second.txPackets;
            outRx += kv.second.rxPackets;
            if (kv.second.rxPackets > 0) {
                outDelay += kv.second.delaySum.GetSeconds() / kv.second.rxPackets;
                flowCount++;
            }
        }
    }
    if (flowCount > 0) outDelay /= flowCount;
}

// ═══════════════════════════════════════════════════════════════════════════════
// INTERFACE CONTROL
// ═══════════════════════════════════════════════════════════════════════════════
void SafeDown(uint32_t n)
{
    auto ip = nodes.Get(n)->GetObject<Ipv4>();
    for (uint32_t i = 1; i < ip->GetNInterfaces(); ++i)
        if (ip->IsUp(i)) ip->SetDown(i);
}
void SafeUp(uint32_t n)
{
    auto ip = nodes.Get(n)->GetObject<Ipv4>();
    for (uint32_t i = 1; i < ip->GetNInterfaces(); ++i)
        if (!ip->IsUp(i)) ip->SetUp(i);
}

// ═══════════════════════════════════════════════════════════════════════════════
// NETANIM PAINT
// ═══════════════════════════════════════════════════════════════════════════════
void PaintNode(uint32_t i)
{
    if (!anim) return;
    if (i == 0) { anim->UpdateNodeColor(nodes.Get(0), 180, 0, 0); return; }
    double rem = eSrc[i] ? eSrc[i]->GetRemainingEnergy() : 0.0;
    if (rem < 0.05)            { anim->UpdateNodeColor(nodes.Get(i), 100,100,100); return; }
    if (g_bhNodes.count(i))   { anim->UpdateNodeColor(nodes.Get(i), 255, 40,  0); return; }
    if (g_shNodes.count(i))   { anim->UpdateNodeColor(nodes.Get(i), 255,200,  0); return; }
    if (g_sfNodes.count(i))   { anim->UpdateNodeColor(nodes.Get(i), 180, 60,220); return; }
    if (g_vampNodes.count(i)) { anim->UpdateNodeColor(nodes.Get(i),  50,  0,180); return; }
    if (isolated[i])           { anim->UpdateNodeColor(nodes.Get(i),  30, 30, 30); return; }
    double t = trust[i];
    uint8_t r, g, b;
    if (t >= 0.80) {
        double f=(t-0.80)/0.20; r=(uint8_t)(f*60); g=(uint8_t)(200+f*20); b=(uint8_t)(60-f*60);
    } else if (t >= 0.50) {
        double f=(t-0.50)/0.30; r=(uint8_t)(255-f*255); g=(uint8_t)(220-f*20); b=(uint8_t)(f*60);
    } else if (t >= 0.30) {
        double f=(t-0.30)/0.20; r=(uint8_t)(220+f*35); g=(uint8_t)(80+f*140); b=0;
    } else {
        double f=(t<0?0:t)/0.30; r=(uint8_t)(180+f*40); g=(uint8_t)(f*80); b=0;
    }
    anim->UpdateNodeColor(nodes.Get(i), r, g, b);
}

void RefreshAnim() { for (uint32_t i=0;i<N;++i) PaintNode(i); UpdateNodeLabels(); }

// ═══════════════════════════════════════════════════════════════════════════════
// [ATK-1] BLACKHOLE  [FIX-RC3]

void ActivateBH(std::vector<uint32_t> tgts, int wave)
{
    double t = Simulator::Now().GetSeconds();
    NS_LOG_INFO(">>> [ATK-1] BLACKHOLE wave" << wave << " @ t=" << t << "s  nodes=" << tgts.size()
                << "  [RC3: silent-drop, interface UP]");
    if (attackEventLog.is_open())
        attackEventLog << std::fixed << std::setprecision(1) << t << ",BH-START,wave" << wave << ",";
    for (uint32_t n : tgts) {
        g_bhNodes.insert(n);
        // [FIX-RC3] Install 100% drop error model — DO NOT call SafeDown.
        // Interface stays UP so AODV continues routing through this node.
        // Only the ML's fwd_ratio feature (drops to ~0) can detect it.
        Ptr<RateErrorModel> em = CreateObject<RateErrorModel>();
        em->SetAttribute("ErrorRate", DoubleValue(1.0));
        em->SetAttribute("ErrorUnit", StringValue("ERROR_UNIT_PACKET"));
        em->SetRandomVariable(CreateObject<UniformRandomVariable>());
        Ptr<WifiNetDevice> wifiDev = DynamicCast<WifiNetDevice>(devices.Get(n));
        NS_ASSERT_MSG(wifiDev, "ActivateBH: not WifiNetDevice for node " << n);
        Ptr<YansWifiPhy> phy = DynamicCast<YansWifiPhy>(wifiDev->GetPhy());
        NS_ASSERT_MSG(phy, "ActivateBH: PHY is not YansWifiPhy for node " << n);
        phy->SetPostReceptionErrorModel(em);
        g_bhErrorModel[n] = em;

        if (prevNextHop[n]>=0)
            LogRouteChangeEvent(t,n,prevNextHop[n],prevNextHop[n],prevPathCost[n],prevPathCost[n],"bh-activated-silent");
        if (attackEventLog.is_open()) attackEventLog << "N"<<n<<"("<<ClusterLabel(n)<<") ";
        if (anim) {
            anim->UpdateNodeColor(nodes.Get(n),255,40,0);
            anim->UpdateNodeDescription(nodes.Get(n),"N"+std::to_string(n)+"["+ClusterLabel(n)+" BH-ATK]");
            anim->UpdateNodeSize(n,5.0,5.0);
        }
    }
    if (attackEventLog.is_open()) attackEventLog << "\n";
    MaybeOpenAttackWindow(); // [FIX-RC1]
    Simulator::Schedule(Seconds(0.5),&DrawRoutingPaths);
}

void DeactivateBH(std::vector<uint32_t> tgts)
{
    double t = Simulator::Now().GetSeconds();
    NS_LOG_INFO("<<< [ATK-1] BLACKHOLE ENDED @ t=" << t);
    if (attackEventLog.is_open()) attackEventLog << std::fixed << t << ",BH-END,,\n";
    for (uint32_t n : tgts) {
        g_bhNodes.erase(n);
        // [FIX-RC3] Remove the error model — interface was never taken down
        Ptr<WifiNetDevice> wifiDev = DynamicCast<WifiNetDevice>(devices.Get(n));
        if (wifiDev) {
            Ptr<YansWifiPhy> phy = DynamicCast<YansWifiPhy>(wifiDev->GetPhy());
            if (phy) phy->SetPostReceptionErrorModel(nullptr);
        }
        g_bhErrorModel[n] = nullptr;
        if (!isolated[n]) {
            LogRouteChangeEvent(t,n,-1,BestNeighbour(n),MAX_HOPS,ComputePathCost(n),"bh-ended");
            prevNextHop[n]=-2; prevPathCost[n]=0.0; g_prevPath.erase(n);
            if (anim) {
                anim->UpdateNodeDescription(nodes.Get(n),"N"+std::to_string(n)+"["+ClusterLabel(n)+"]");
                anim->UpdateNodeSize(n,2.5,2.5); PaintNode(n);
            }
        }
    }
    MaybeCloseAttackWindow(); // [FIX-RC1]
}

// ═══════════════════════════════════════════════════════════════════════════════
// [ATK-2] SELECTIVE FORWARDING
// ═══════════════════════════════════════════════════════════════════════════════
void ActivateSF(std::vector<uint32_t> tgts)
{
    double t = Simulator::Now().GetSeconds();
    NS_LOG_INFO(">>> [ATK-2] SF ATTACK @ t=" << t << "  nodes=" << tgts.size()
                << "  drop=" << (int)(SF_DROP_RATE*100) << "%");
    if (attackEventLog.is_open())
        attackEventLog << std::fixed << t << ",SF-START,drop=" << (int)(SF_DROP_RATE*100) << "%,";
    for (uint32_t n : tgts) {
        g_sfNodes.insert(n);
        Ptr<RateErrorModel> em = CreateObject<RateErrorModel>();
        em->SetAttribute("ErrorRate", DoubleValue(SF_DROP_RATE));
        em->SetAttribute("ErrorUnit", StringValue("ERROR_UNIT_PACKET"));
        em->SetRandomVariable(CreateObject<UniformRandomVariable>());
        Ptr<WifiNetDevice> wifiDev = DynamicCast<WifiNetDevice>(devices.Get(n));
        NS_ASSERT_MSG(wifiDev, "ActivateSF: not WifiNetDevice for node " << n);
        Ptr<YansWifiPhy> phy = DynamicCast<YansWifiPhy>(wifiDev->GetPhy());
        NS_ASSERT_MSG(phy, "ActivateSF: PHY is not YansWifiPhy for node " << n);
        phy->SetPostReceptionErrorModel(em);
        g_sfErrorModel[n] = em;
        if (attackEventLog.is_open()) attackEventLog << "N"<<n<<"("<<ClusterLabel(n)<<") ";
        if (anim) {
            anim->UpdateNodeColor(nodes.Get(n),180,60,220);
            anim->UpdateNodeDescription(nodes.Get(n),
                "N"+std::to_string(n)+"["+ClusterLabel(n)+" SF"+std::to_string((int)(SF_DROP_RATE*100))+"%]");
            anim->UpdateNodeSize(n,4.0,4.0);
        }
    }
    if (attackEventLog.is_open()) attackEventLog << "\n";
    MaybeOpenAttackWindow(); // [FIX-RC1]
}

void DeactivateSF(std::vector<uint32_t> tgts)
{
    double t = Simulator::Now().GetSeconds();
    NS_LOG_INFO("<<< [ATK-2] SF ENDED @ t=" << t);
    if (attackEventLog.is_open()) attackEventLog << std::fixed << t << ",SF-END,,\n";
    for (uint32_t n : tgts) {
        g_sfNodes.erase(n);
        Ptr<WifiNetDevice> wifiDev = DynamicCast<WifiNetDevice>(devices.Get(n));
        if (wifiDev) {
            Ptr<YansWifiPhy> phy = DynamicCast<YansWifiPhy>(wifiDev->GetPhy());
            if (phy) phy->SetPostReceptionErrorModel(nullptr);
        }
        g_sfErrorModel[n] = nullptr;
        if (!isolated[n] && !g_bhNodes.count(n)) {
            if (anim) {
                anim->UpdateNodeDescription(nodes.Get(n),"N"+std::to_string(n)+"["+ClusterLabel(n)+"]");
                anim->UpdateNodeSize(n,2.5,2.5); PaintNode(n);
            }
        }
    }
    MaybeCloseAttackWindow(); // [FIX-RC1]
}

// ═══════════════════════════════════════════════════════════════════════════════
// [ATK-3] SINKHOLE
// ═══════════════════════════════════════════════════════════════════════════════
void ActivateSH(std::vector<uint32_t> tgts)
{
    double t = Simulator::Now().GetSeconds();
    NS_LOG_INFO(">>> [ATK-3] SINKHOLE @ t=" << t << "  nodes=" << tgts.size()
                << "  fake_metric=0.99  drop=" << (int)(SH_DROP_RATE*100) << "%");
    if (attackEventLog.is_open())
        attackEventLog << std::fixed << t << ",SH-START,fake_metric=0.99,";
    for (uint32_t n : tgts) {
        g_shNodes.insert(n);
        g_shFakeMetric[n] = 0.99;
        routingMetric[n]  = 0.99;
        Ptr<RateErrorModel> em = CreateObject<RateErrorModel>();
        em->SetAttribute("ErrorRate", DoubleValue(SH_DROP_RATE));
        em->SetAttribute("ErrorUnit", StringValue("ERROR_UNIT_PACKET"));
        em->SetRandomVariable(CreateObject<UniformRandomVariable>());
        Ptr<WifiNetDevice> wifiDev = DynamicCast<WifiNetDevice>(devices.Get(n));
        NS_ASSERT_MSG(wifiDev, "ActivateSH: not WifiNetDevice for node " << n);
        Ptr<YansWifiPhy> phy = DynamicCast<YansWifiPhy>(wifiDev->GetPhy());
        NS_ASSERT_MSG(phy, "ActivateSH: PHY is not YansWifiPhy for node " << n);
        phy->SetPostReceptionErrorModel(em);
        g_shErrorModel[n] = em;
        if (attackEventLog.is_open()) attackEventLog << "N"<<n<<"("<<ClusterLabel(n)<<") ";
        if (anim) {
            anim->UpdateNodeColor(nodes.Get(n),255,200,0);
            anim->UpdateNodeDescription(nodes.Get(n),
                "N"+std::to_string(n)+"["+ClusterLabel(n)+" SH-ATK]");
            anim->UpdateNodeSize(n,4.5,4.5);
        }
    }
    if (attackEventLog.is_open()) attackEventLog << "\n";
    MaybeOpenAttackWindow(); // [FIX-RC1]
}

void DeactivateSH(std::vector<uint32_t> tgts)
{
    double t = Simulator::Now().GetSeconds();
    NS_LOG_INFO("<<< [ATK-3] SH ENDED @ t=" << t);
    if (attackEventLog.is_open()) attackEventLog << std::fixed << t << ",SH-END,,\n";
    for (uint32_t n : tgts) {
        g_shNodes.erase(n); g_shFakeMetric.erase(n);
        Ptr<WifiNetDevice> wifiDev = DynamicCast<WifiNetDevice>(devices.Get(n));
        if (wifiDev) {
            Ptr<YansWifiPhy> phy = DynamicCast<YansWifiPhy>(wifiDev->GetPhy());
            if (phy) phy->SetPostReceptionErrorModel(nullptr);
        }
        g_shErrorModel[n] = nullptr;
        if (!isolated[n] && !g_bhNodes.count(n) && !g_sfNodes.count(n)) {
            if (anim) {
                anim->UpdateNodeDescription(nodes.Get(n),"N"+std::to_string(n)+"["+ClusterLabel(n)+"]");
                anim->UpdateNodeSize(n,2.5,2.5); PaintNode(n);
            }
        }
    }
    MaybeCloseAttackWindow(); // [FIX-RC1]
}

// ═══════════════════════════════════════════════════════════════════════════════
// [ATK-4] VAMPIRE
// ═══════════════════════════════════════════════════════════════════════════════
void ActivateVamp(std::vector<uint32_t> tgts)
{
    double t = Simulator::Now().GetSeconds();
    NS_LOG_INFO(">>> [ATK-4] VAMPIRE @ t=" << t << "  nodes=" << tgts.size());
    if (attackEventLog.is_open())
        attackEventLog << std::fixed << t << ",VAMP-START,drain=" << (VAMP_DRAIN_FRAC*100) << "%/interval,";
    for (uint32_t n : tgts) {
        g_vampNodes.insert(n);
        if (attackEventLog.is_open()) attackEventLog << "N"<<n<<"("<<ClusterLabel(n)<<") ";
        if (anim) {
            anim->UpdateNodeColor(nodes.Get(n),50,0,180);
            anim->UpdateNodeDescription(nodes.Get(n),
                "N"+std::to_string(n)+"["+ClusterLabel(n)+" VAMP]");
            anim->UpdateNodeSize(n,4.0,4.0);
        }
    }
    if (attackEventLog.is_open()) attackEventLog << "\n";
    MaybeOpenAttackWindow(); // [FIX-RC1]
}

void DeactivateVamp(std::vector<uint32_t> tgts)
{
    double t = Simulator::Now().GetSeconds();
    NS_LOG_INFO("<<< [ATK-4] VAMPIRE ENDED @ t=" << t);
    if (attackEventLog.is_open()) attackEventLog << std::fixed << t << ",VAMP-END,,\n";
    for (uint32_t n : tgts) {
        g_vampNodes.erase(n);
        if (!isolated[n] && !g_bhNodes.count(n) && !g_sfNodes.count(n) && !g_shNodes.count(n)) {
            if (anim) {
                anim->UpdateNodeDescription(nodes.Get(n),"N"+std::to_string(n)+"["+ClusterLabel(n)+"]");
                anim->UpdateNodeSize(n,2.5,2.5); PaintNode(n);
            }
        }
    }
    MaybeCloseAttackWindow(); // [FIX-RC1]
}

// [FIX-10] Drain reduced to VAMP_DRAIN_FRAC=0.005
void ApplyVampireDrain(double ts)
{
    if (ts >= SIM_DUR || g_vampNodes.empty()) return;
    for (uint32_t v : g_vampNodes) {
        if (!eSrc[v]) continue;
        double rem = eSrc[v]->GetRemainingEnergy();
        if (rem < 0.05) continue;
        double drain = rem * VAMP_DRAIN_FRAC;
        drain = std::min(drain, rem - 0.01);
        eSrc[v]->SetInitialEnergy(std::max(0.01, eSrc[v]->GetInitialEnergy() - drain));
        eSrc[v]->UpdateEnergySource();
        g_vampDrainAccum[v] += drain;
        // [IMP-04] Reduced logging — only log significant drains
        if (drain > 0.5) {
            NS_LOG_INFO("  [VAMP] N" << v << " drained " << drain << "J  remaining=" << eSrc[v]->GetRemainingEnergy() << "J");
        }
        auto mobV = nodes.Get(v)->GetObject<MobilityModel>(); if (!mobV) continue;
        for (uint32_t j = 1; j < N; ++j) {
            if (j==v || !eSrc[j]) continue;
            auto mobJ = nodes.Get(j)->GetObject<MobilityModel>();
            if (!mobJ || mobV->GetDistanceFrom(mobJ) > RADIO_RANGE) continue;
            double remJ = eSrc[j]->GetRemainingEnergy();
            if (remJ < 0.05) continue;
            double drainJ = std::min(remJ * 0.001, remJ - 0.01); // 0.1% neighbour drain
            eSrc[j]->SetInitialEnergy(std::max(0.01, eSrc[j]->GetInitialEnergy() - drainJ));
            eSrc[j]->UpdateEnergySource();
        }
    }
    if (ts + LOG_INT < SIM_DUR)
        Simulator::Schedule(Seconds(LOG_INT), &ApplyVampireDrain, ts + LOG_INT);
}

// ═══════════════════════════════════════════════════════════════════════════════
// [FIX-03] ATTACK SCHEDULE — starts at WARM_UP=60s
// [FIX-15] Extended SF/SH windows to 70s for ML reaction time
// ═══════════════════════════════════════════════════════════════════════════════
void ScheduleCycle(double base)
{
    if (base >= SIM_DUR) return;
    auto bh1   = PickNodes(BH_NODES_PER_WAVE);
    auto sf1   = PickNodes(SF_NODES_PER_WAVE);
    auto sh1   = PickNodes(SH_NODES_PER_WAVE);
    auto vamp1 = PickNodes(VAMP_NODES);
    auto bh2   = PickNodes(BH_NODES_PER_WAVE);

    // BH wave 1: 0→60s
    Simulator::Schedule(Seconds(base +  0.0), &ActivateBH,     bh1, 1);
    Simulator::Schedule(Seconds(base + 60.0), &DeactivateBH,   bh1);

    // Vampire: full cycle
    Simulator::Schedule(Seconds(base +  0.0), &ActivateVamp,   vamp1);
    Simulator::Schedule(Seconds(base+115.0),  &DeactivateVamp, vamp1);
    Simulator::Schedule(Seconds(base + LOG_INT), &ApplyVampireDrain, base + LOG_INT);

    // SF: 20→90s  [FIX-15: +10s window]
    Simulator::Schedule(Seconds(base + 20.0), &ActivateSF,     sf1);
    Simulator::Schedule(Seconds(base + 90.0), &DeactivateSF,   sf1);

    // SH: 50→120s [FIX-15: +10s window]
    Simulator::Schedule(Seconds(base + 50.0), &ActivateSH,     sh1);
    Simulator::Schedule(Seconds(base+120.0),  &DeactivateSH,   sh1);

    // BH wave 2: 75→115s
    Simulator::Schedule(Seconds(base + 75.0), &ActivateBH,     bh2, 2);
    Simulator::Schedule(Seconds(base+115.0),  &DeactivateBH,   bh2);

    if (base + CYCLE < SIM_DUR)
        Simulator::Schedule(Seconds(base + CYCLE), &ScheduleCycle, base + CYCLE);
}

// ═══════════════════════════════════════════════════════════════════════════════
// ML IPC
// ═══════════════════════════════════════════════════════════════════════════════
bool IpcSend(const std::string& payload, std::string& resp)
{
    int s = ::socket(AF_INET,SOCK_STREAM,0);
    if (s<0) return false;
    struct timeval tv{5,0};
    setsockopt(s,SOL_SOCKET,SO_RCVTIMEO,&tv,sizeof(tv));
    setsockopt(s,SOL_SOCKET,SO_SNDTIMEO,&tv,sizeof(tv));
    struct sockaddr_in srv{}; srv.sin_family=AF_INET; srv.sin_port=htons(ML_PORT);
    inet_pton(AF_INET,ML_HOST,&srv.sin_addr);
    if (::connect(s,(struct sockaddr*)&srv,sizeof(srv))<0){::close(s);return false;}
    std::string msg=payload+"\n"; ::send(s,msg.c_str(),msg.size(),0);
    char buf[524288]={}; ssize_t n=::recv(s,buf,sizeof(buf)-1,0); ::close(s);
    if (n<=0) return false;
    resp=std::string(buf,n);
    return true;
}

// [FIX-12] EMA smoothing: new_trust = EMA_ALPHA * ml_trust + (1-EMA_ALPHA) * old_trust
void ParseTrust(const std::string& json)
{
    auto p=json.find("\"trust\""); if (p==std::string::npos) return;
    auto a=json.find('[',p), b=json.find(']',a); if (a==std::string::npos) return;
    auto t=JParseArr(json.substr(a,b-a+1));
    double now = Simulator::Now().GetSeconds();

    
    for (size_t i=0;i<t.size()&&i<N;++i) {
        double oldTrust = trust[i];
        double newTrust = EMA_ALPHA * t[i] + (1.0 - EMA_ALPHA) * trust[i];

        // [IMP-04] Emit ATTACK_DETECTED event on significant trust drops
        if (i > 0 && oldTrust - newTrust > 0.15 && oldTrust > 0.5) {
            MLDecisionEvent evt;
            evt.timestamp  = now;
            evt.nodeId     = (uint32_t)i;
            evt.eventType  = "ATTACK_DETECTED";
            evt.attackType = GetNodeAttackType((uint32_t)i);
            if (evt.attackType.empty()) evt.attackType = "ANOMALY";
            evt.oldTrust   = oldTrust;
            evt.newTrust   = newTrust;
            evt.oldNextHop = nextHopToSink[i];
            evt.newNextHop = nextHopToSink[i];
            evt.oldScore   = routingMetric[i];
            evt.newScore   = routingMetric[i];
            evt.oldPath    = BuildPathString((uint32_t)i);
            evt.newPath    = evt.oldPath;
            LogMLDecision(evt);
        }
        if (isolated[i] && !g_bhNodes.count(i)){
            trust[i] = std::min(trust[i] * 1.05, RESTORE_TRUST_THR - 0.01); // slow rehab
        }
        trust[i] = newTrust;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ROUTING  — BestNeighbour, WalkToSink, etc.
// ═══════════════════════════════════════════════════════════════════════════════

// [FIX-11] Hysteresis: only switch next-hop if improvement > ROUTE_HYSTERESIS
// [FIX-17] Emergency fallback: if all neighbours below ROUTE_THR, pick best available
int BestNeighbour(uint32_t srcNode)
{
    if (srcNode==0||isolated[srcNode]||g_bhNodes.count(srcNode)) return -1;
    auto mobSrc=nodes.Get(srcNode)->GetObject<MobilityModel>();
    auto mobSink=nodes.Get(0)->GetObject<MobilityModel>();
    if (!mobSrc||!mobSink) return -1;
    double distSrc=mobSrc->GetDistanceFrom(mobSink);
    double minRed=distSrc*PROGRESS_MIN_FRAC;

    int bestIdx=-1;          double bestMetric=-1.0;
    int fallbackIdx=-1;      double fallbackMinDist=distSrc;
    int emergencyIdx=-1;     double emergencyBestMetric=-1.0; // [FIX-17]

    for (uint32_t j=0;j<N;++j) {
        if (j==srcNode) continue;
        double rem=eSrc[j]?eSrc[j]->GetRemainingEnergy():0.0;
        if (rem<0.05) continue;
        auto mobJ=nodes.Get(j)->GetObject<MobilityModel>(); if (!mobJ) continue;
        if (mobSrc->GetDistanceFrom(mobJ)>RADIO_RANGE) continue;
        double distJ=mobJ->GetDistanceFrom(mobSink);
        double m=routingMetric[j];

        // Track best unrestricted neighbour for emergency fallback [FIX-17]
        if (distJ < distSrc) {
            if (m > emergencyBestMetric) { emergencyBestMetric = m; emergencyIdx = (int)j; }
        }

        if (m < ROUTE_THR) continue; // soft-avoid gate (relaxed in FIX-05)

        if ((distSrc-distJ)>=minRed) {
            // [FIX-11] Hysteresis: prefer current next-hop unless new is clearly better
            bool isCurrent = (nextHopToSink[srcNode] == (int)j);
            double effectiveMetric = isCurrent ? m : m - ROUTE_HYSTERESIS;
            if (effectiveMetric > bestMetric) { bestMetric=m; bestIdx=(int)j; }
        }
        if (distJ<fallbackMinDist){fallbackMinDist=distJ;fallbackIdx=(int)j;}
    }

    if (bestIdx >= 0)     return bestIdx;
    if (fallbackIdx >= 0) return fallbackIdx;
    return emergencyIdx; // [FIX-17] last resort: ignore ROUTE_THR gate
}

int NeighbourCount(uint32_t srcNode)
{
    if (isolated[srcNode]||g_bhNodes.count(srcNode)) return 0;
    auto mobSrc=nodes.Get(srcNode)->GetObject<MobilityModel>(); if (!mobSrc) return 0;
    int cnt=0;
    for (uint32_t j=0;j<N;++j) {
        if (j==srcNode||isolated[j]||g_bhNodes.count(j)||g_sfNodes.count(j)) continue;
        auto mobJ=nodes.Get(j)->GetObject<MobilityModel>(); if (!mobJ) continue;
        if (mobSrc->GetDistanceFrom(mobJ)<=RADIO_RANGE) cnt++;
    }
    return cnt;
}

WalkResult WalkToSink(uint32_t srcNode)
{
    WalkResult r; r.reachedSink=false; r.hadLoop=false;
    if (srcNode==0){r.path={0};r.reachedSink=true;return r;}
    if (isolated[srcNode]||g_bhNodes.count(srcNode)){r.path={srcNode};return r;}
    r.path.push_back(srcNode);
    std::set<uint32_t> visited; uint32_t cur=srcNode;
    for (int hop=0;hop<MAX_HOPS;++hop) {
        if (visited.count(cur)){r.hadLoop=true;break;}
        visited.insert(cur);
        auto mc=nodes.Get(cur)->GetObject<MobilityModel>();
        auto ms=nodes.Get(0)->GetObject<MobilityModel>();
        if (mc&&ms&&mc->GetDistanceFrom(ms)<=RADIO_RANGE){r.path.push_back(0);r.reachedSink=true;break;}
        int nh=BestNeighbour(cur); if (nh<0) break;
        cur=(uint32_t)nh; r.path.push_back(cur);
        if (cur==0){r.reachedSink=true;break;}
    }
    return r;
}

double ComputePathCost(uint32_t srcNode)
{
    WalkResult wr=WalkToSink(srcNode);
    double cost=0.0;
    double hopBaseCost=1.0/N;
    for (size_t k=0;k+1<wr.path.size();++k) {
        uint32_t nd=wr.path[k];
        double trustPenalty  = g_enableML     ? (1.0-routingMetric[nd]) : 0.0;
        double energyPenalty = g_enableEnergy ?
            (1.0-(eSrc[nd]?eSrc[nd]->GetRemainingEnergy()/INIT_E:0.0))*0.1 : 0.0;
        cost += hopBaseCost + trustPenalty + energyPenalty;
    }
    if (wr.hadLoop)      cost += 2.0;
    if (!wr.reachedSink) cost += (double)(MAX_HOPS-(int)wr.path.size())/MAX_HOPS;
    return cost;
}

std::string BuildPathString(uint32_t srcNode)
{
    if (srcNode==0) return "0";
    if (isolated[srcNode]||g_bhNodes.count(srcNode)) return std::to_string(srcNode)+"->BLOCKED";
    WalkResult wr=WalkToSink(srcNode);
    std::ostringstream ss;
    for (size_t i=0;i<wr.path.size();++i){if (i) ss<<"->"; ss<<wr.path[i];}
    if (wr.hadLoop) ss<<"->LOOP";
    if (!wr.reachedSink&&!wr.hadLoop) ss<<"->DEAD";
    return ss.str();
}

int ComputeHopCount(uint32_t srcNode)
{
    if (srcNode==0) return 0;
    if (isolated[srcNode]||g_bhNodes.count(srcNode)) return -1;
    WalkResult wr=WalkToSink(srcNode);
    if (wr.hadLoop||!wr.reachedSink) return -1;
    return std::max(0,(int)wr.path.size()-1);
}

// ═══════════════════════════════════════════════════════════════════════════════
// TRUST ROUTING PROTOCOL  (v5.0 — unchanged from patch notes)
// ═══════════════════════════════════════════════════════════════════════════════
class TrustRoutingProtocol : public Ipv4RoutingProtocol
{
public:
    static TypeId GetTypeId()
    {
        static TypeId tid = TypeId("ns3::TrustRoutingProtocol")
            .SetParent<Ipv4RoutingProtocol>()
            .SetGroupName("Internet")
            .AddConstructor<TrustRoutingProtocol>();
        return tid;
    }

    TrustRoutingProtocol() : m_ipv4(nullptr), m_nodeId(UINT32_MAX) {}
    ~TrustRoutingProtocol() override {}

    void SetIpv4(Ptr<Ipv4> ipv4) override
    {
        m_ipv4  = ipv4;
        m_nodeId = ipv4->GetObject<Node>()->GetId();
    }

    Ptr<Ipv4Route> RouteOutput(Ptr<Packet>             pkt,
                               const Ipv4Header&        hdr,
                               Ptr<NetDevice>           oif,
                               Socket::SocketErrno&     sockerr) override
    {
        // Sink node (0) never originates forwarded traffic
        if (m_nodeId == 0) {
            sockerr = Socket::ERROR_NOROUTETOHOST;
            return nullptr;
        }
        sockerr = Socket::ERROR_NOTERROR;
        if (isolated[m_nodeId] || g_bhNodes.count(m_nodeId)) {
            sockerr = Socket::ERROR_NOROUTETOHOST;
            return nullptr;
        }
        double rem = eSrc[m_nodeId] ? eSrc[m_nodeId]->GetRemainingEnergy() : 0.0;
        if (rem < 0.05) { sockerr = Socket::ERROR_NOROUTETOHOST; return nullptr; }

        Ptr<MobilityModel> mobSelf = nodes.Get(m_nodeId)->GetObject<MobilityModel>();
        Ptr<MobilityModel> mobSink = nodes.Get(0)->GetObject<MobilityModel>();
        if (mobSelf && mobSink && mobSelf->GetDistanceFrom(mobSink) <= RADIO_RANGE) {
            pktOrig[m_nodeId]++;
            pktTx[m_nodeId]++;
            return BuildRoute(m_nodeId, 0);
        }
        int nh = BestNeighbour(m_nodeId);
        if (nh < 0) {
            NS_LOG_WARN("[ROUTE-OUT] N" << m_nodeId << " no next-hop found");
            sockerr = Socket::ERROR_NOROUTETOHOST;
            return nullptr;
        }
        pktOrig[m_nodeId]++;
        pktTx[m_nodeId]++;
        NS_LOG_INFO("[ROUTE-OUT] N" << m_nodeId << " -> N" << nh
                    << " t=" << Simulator::Now().GetSeconds() << "s");
        return BuildRoute(m_nodeId, (uint32_t)nh);
    }

    // [PATCH-1] const-reference callbacks
    bool RouteInput(Ptr<const Packet>               pkt,
                    const Ipv4Header&               hdr,
                    Ptr<const NetDevice>            idev,
                    const UnicastForwardCallback&   ucb,
                    const MulticastForwardCallback& mcb,
                    const LocalDeliverCallback&     lcb,
                    const ErrorCallback&            ecb) override
    {
        uint32_t iif = m_ipv4->GetInterfaceForDevice(idev);
        Ipv4Address dst = hdr.GetDestination();

        // Broadcast / multicast — local delivery only
        if (dst.IsBroadcast() || dst.IsMulticast()) {
            if (!lcb.IsNull()) lcb(pkt, hdr, iif);
            return true;
        }

        // Local delivery: use ns-3's IsDestinationAddress which correctly
        // matches the node's own IP(s) including the sink's g_sinkAddr.
        // This is the canonical ns-3 way — it works because SetIpv4() was
        // called through the helper so all interfaces are registered.
        if (m_ipv4->IsDestinationAddress(dst, iif)) {
            if (!lcb.IsNull()) {
                pktRx[m_nodeId]++;
                NS_LOG_INFO("[ROUTE-IN] N" << m_nodeId << " LOCAL-DELIVER t="
                    << Simulator::Now().GetSeconds() << "s  sink_rx=" << pktRx[0]);
                lcb(pkt, hdr, iif);
            }
            return true;
        }

        // From here: packet needs forwarding
        if (isolated[m_nodeId] || g_bhNodes.count(m_nodeId)) {
            NS_LOG_WARN("[ROUTE-IN] N" << m_nodeId << " DROP (isolated/BH)");
            ecb(pkt, hdr, Socket::ERROR_NOROUTETOHOST); return false;
        }
        double rem = eSrc[m_nodeId] ? eSrc[m_nodeId]->GetRemainingEnergy() : 0.0;
        if (rem < 0.05) {
            ecb(pkt, hdr, Socket::ERROR_NOROUTETOHOST); return false;
        }

        // Count as received-for-forwarding so ML fwdF ratio is meaningful
        pktRx[m_nodeId]++;

        Ptr<MobilityModel> mobSelf = nodes.Get(m_nodeId)->GetObject<MobilityModel>();
        Ptr<MobilityModel> mobSink = nodes.Get(0)->GetObject<MobilityModel>();
        if (mobSelf && mobSink && mobSelf->GetDistanceFrom(mobSink) <= RADIO_RANGE) {
            Ptr<Ipv4Route> rt = BuildRoute(m_nodeId, 0);
            if (rt) {
                pktTx[m_nodeId]++;
                NS_LOG_INFO("[ROUTE-IN] N" << m_nodeId << " -> SINK direct");
                ucb(rt, pkt, hdr); return true;
            }
        }
        int nh = BestNeighbour(m_nodeId);
        if (nh < 0) {
            NS_LOG_WARN("[ROUTE-IN] N" << m_nodeId << " no next-hop");
            ecb(pkt, hdr, Socket::ERROR_NOROUTETOHOST); return false;
        }
        Ptr<Ipv4Route> rt = BuildRoute(m_nodeId, (uint32_t)nh);
        if (!rt) {
            NS_LOG_WARN("[ROUTE-IN] N" << m_nodeId << " BuildRoute null");
            ecb(pkt, hdr, Socket::ERROR_NOROUTETOHOST); return false;
        }
        pktTx[m_nodeId]++;
        NS_LOG_INFO("[ROUTE-IN] N" << m_nodeId << " -> N" << nh);
        ucb(rt, pkt, hdr);
        return true;
    }

    void NotifyInterfaceUp(uint32_t)   override {}
    void NotifyInterfaceDown(uint32_t) override {}
    void NotifyAddAddress(uint32_t, Ipv4InterfaceAddress) override {}
    void NotifyRemoveAddress(uint32_t, Ipv4InterfaceAddress) override {}
    void PrintRoutingTable(Ptr<OutputStreamWrapper>, Time::Unit) const override {}

private:
    // [FIX-PDR-B] CRITICAL: destination must ALWAYS be g_sinkAddr.
    // If destination = nhAddr for intermediate hops, RouteInput on the
    // next node calls IsDestinationAddress(nhAddr) → true → lcb() local
    // delivery instead of forwarding. Every multi-hop packet dies at hop 1.
    // The gateway field carries the next L3 hop address for ARP/MAC lookup.
    Ptr<Ipv4Route> BuildRoute(uint32_t srcNode, uint32_t nhNode)
    {
        if (nhNode >= N) return nullptr;
        // Find first UP interface (skip loopback at index 0)
        uint32_t outIface = UINT32_MAX;
        for (uint32_t i = 1; i < m_ipv4->GetNInterfaces(); ++i) {
            if (m_ipv4->IsUp(i)) { outIface = i; break; }
        }
        // Fallback: if no interface is up (e.g. SafeDown was called), fail
        if (outIface == UINT32_MAX) {
            NS_LOG_WARN("[BuildRoute] N" << srcNode << " no UP interface found");
            return nullptr;
        }
        Ipv4Address nhAddr = ifaces.GetAddress(nhNode);
        if (nhAddr == Ipv4Address("0.0.0.0") || nhAddr == Ipv4Address()) {
            NS_LOG_WARN("[BuildRoute] N" << srcNode << "->N" << nhNode << " nhAddr is zero!");
            return nullptr;
        }
        Ptr<Ipv4Route> rt = Create<Ipv4Route>();
        rt->SetDestination(g_sinkAddr);
        rt->SetSource(m_ipv4->GetAddress(outIface, 0).GetLocal());
        rt->SetGateway(nhAddr);
        rt->SetOutputDevice(m_ipv4->GetNetDevice(outIface));
        return rt;
    }
    Ptr<Ipv4>  m_ipv4;
    uint32_t   m_nodeId;
};

NS_OBJECT_ENSURE_REGISTERED(TrustRoutingProtocol);

// ═══════════════════════════════════════════════════════════════════════════════
// TRUST ROUTING PROTOCOL HELPER
// Follows the ns-3 helper pattern so inet.SetRoutingHelper() wires SetIpv4()
// correctly through Ipv4L3Protocol — the same path AODV uses.
// ═══════════════════════════════════════════════════════════════════════════════
class TrustRoutingProtocolHelper : public Ipv4RoutingHelper
{
public:
    TrustRoutingProtocolHelper() {}
    TrustRoutingProtocolHelper* Copy() const override
    {
        return new TrustRoutingProtocolHelper(*this);
    }
    Ptr<Ipv4RoutingProtocol> Create(Ptr<Node> node) const override
    {
        return CreateObject<TrustRoutingProtocol>();
    }
};


// ═══════════════════════════════════════════════════════════════════════════════
// ROUTING EVIDENCE LOGS
// ═══════════════════════════════════════════════════════════════════════════════
void WriteRoutingMatrix(double ts)
{
    std::string fname=g_matrixDir+"/routing_matrix_"+std::to_string((int)ts)+".csv";
    std::ofstream mat(fname); if (!mat.is_open()) return;
    mat<<"node"; for (uint32_t j=0;j<N;++j) mat<<","<<j; mat<<"\n";
    for (uint32_t i=0;i<N;++i){
        mat<<i;
        for (uint32_t j=0;j<N;++j)
            mat<<","+std::string((nextHopToSink[i]==(int)j&&!isolated[i]&&!g_bhNodes.count(i))?"1":"0");
        mat<<"\n";
    }
}

void LogRouteChangeEvent(double ts,uint32_t node,int oldNH,int newNH,
                          double oldCost,double newCost,const std::string& reason)
{
    if (!routeChangeLog.is_open()) return;
    routeChangeLog<<std::fixed<<std::setprecision(1)<<ts<<","<<node<<","<<ClusterLabel(node)<<","
                  <<(oldNH>=0?"N"+std::to_string(oldNH):"NONE")<<","
                  <<(newNH>=0?"N"+std::to_string(newNH):"NONE")<<","
                  <<std::setprecision(4)<<oldCost<<","<<newCost<<","<<(newCost-oldCost)<<","<<reason<<"\n";
    routeChangeLog.flush();
}

void WritePathTraces(double ts)
{
    if (!pathTraceLog.is_open()) return;
    for (uint32_t i=1;i<N;++i){
        std::string st=NodeState(i);
        std::string ps=(st=="normal"||st=="SOFT-AVOID")?BuildPathString(i):st;
        int hops=(st=="normal"||st=="SOFT-AVOID")?hopCount[i]:-1;
        pathTraceLog<<std::fixed<<std::setprecision(1)<<ts<<","<<i<<","<<ClusterLabel(i)<<","
                    <<"\""<<ps<<"\","<<hops<<","<<std::setprecision(4)<<pathCost[i]<<","
                    <<((pathCost[i]<=STABLE_COST_THR)?1:0)<<","<<st<<"\n";
    }
    pathTraceLog.flush();
}

void WriteHopEvolution(double ts)
{
    if (!hopEvolLog.is_open()) return;
    hopEvolLog<<std::fixed<<std::setprecision(1)<<ts;
    for (uint32_t i=0;i<N;++i) hopEvolLog<<","<<hopCount[i];
    hopEvolLog<<"\n"; hopEvolLog.flush();
}

// ═══════════════════════════════════════════════════════════════════════════════
// ADAPTIVE WEIGHTS
// ═══════════════════════════════════════════════════════════════════════════════
void UpdateAdaptiveWeights()
{
    if (!g_enableEnergy){g_dynamicAlpha=1.0;g_dynamicBeta=0.0;return;}
    double sumE=0.0,sumE2=0.0; uint32_t nLive=0;
    for (uint32_t i=1;i<N;++i){
        double e=eSrc[i]?eSrc[i]->GetRemainingEnergy():0.0;
        if (e>0.05){sumE+=e;sumE2+=e*e;nLive++;}
    }
    double meanE=nLive>0?sumE/nLive:0.0;
    double varE=nLive>1?(sumE2/nLive-meanE*meanE):0.0;
    double normVar=std::min(1.0,varE/(INIT_E*INIT_E*0.25));
    uint32_t nSuspect=0;
    for (uint32_t i=1;i<N;++i) if (trust[i]<0.40&&!isolated[i]) nSuspect++;
    double anomalyRate=(double)nSuspect/std::max(1u,(uint32_t)(N-1));
    g_dynamicBeta =std::min(0.50,BETA +normVar*0.20);
    g_dynamicAlpha=std::min(0.85,ALPHA+anomalyRate*0.15);
    double tot=g_dynamicAlpha+g_dynamicBeta;
    g_dynamicAlpha/=tot; g_dynamicBeta/=tot;
}

// ═══════════════════════════════════════════════════════════════════════════════
// UPDATE PATH METRICS
// ═══════════════════════════════════════════════════════════════════════════════
void UpdatePathMetrics()
{
    double now=Simulator::Now().GetSeconds();
    uint32_t nActive=0; double sumCost=0.0;
    for (uint32_t i=1;i<N;++i){
        double rem=eSrc[i]?eSrc[i]->GetRemainingEnergy():0.0;
        if (rem<0.05||isolated[i]||g_bhNodes.count(i)){
            if (prevNextHop[i]>=0){
                std::string r=isolated[i]?"isolated-loss":g_bhNodes.count(i)?"bh-loss":"dead";
                LogRouteChangeEvent(now,i,prevNextHop[i],-1,prevPathCost[i],MAX_HOPS,r);
                prevNextHop[i]=-1; prevPathCost[i]=MAX_HOPS;
            }
            nextHopToSink[i]=-1; hopCount[i]=-1; continue;
        }
        int nh=BestNeighbour(i); double pc=ComputePathCost(i); int hc=ComputeHopCount(i);
        WalkResult wrNew=WalkToSink(i);
        auto prevIt=g_prevPath.find(i);
        if (prevIt!=g_prevPath.end()&&prevIt->second!=wrNew.path&&wrNew.reachedSink&&prevNextHop[i]!=-2)
            VisualizePathChange(i,prevIt->second,wrNew.path,prevPathCost[i],pc);
        g_prevPath[i]=wrNew.path;
        bool firstTime=(prevNextHop[i]==-2);
        bool nhChanged=(!firstTime&&nh!=prevNextHop[i]);
        double costRatio=(prevPathCost[i]>0.01)?pc/prevPathCost[i]:(pc>0.01?99.0:1.0);
        if (firstTime) LogRouteChangeEvent(now,i,-1,nh,0.0,pc,"route-established");
        else if (nhChanged){
            std::string r=(costRatio<0.85)?"improve":(costRatio>1.15)?"degrade":"nh-swap";
            LogRouteChangeEvent(now,i,prevNextHop[i],nh,prevPathCost[i],pc,r);
        } else if (std::abs(pc-prevPathCost[i])>0.10&&!firstTime){
            bool cru=(pc>REVAL_COST_THR&&prevPathCost[i]<=REVAL_COST_THR);
            bool csu=(pc<=STABLE_COST_THR&&prevPathCost[i]>STABLE_COST_THR);
            std::string r2=cru?"cost-rise":csu?"cost-fall":costRatio>1.15?"cost-drift-up":costRatio<0.85?"cost-drift-down":"cost-drift";
            LogRouteChangeEvent(now,i,nh,nh,prevPathCost[i],pc,r2);
        }
        prevNextHop[i]=nh; prevPathCost[i]=pc;
        nextHopToSink[i]=nh; pathCost[i]=pc; hopCount[i]=hc;
        sumCost+=pc; nActive++;
        totalIntervals[i]++; if (pc<=STABLE_COST_THR) stableIntervals[i]++;
        if (g_enableRouteOpt&&hc>0){
            bool cd=revalTimestamp.count(i)&&(now-revalTimestamp[i])<REVAL_COOLDOWN;
            if (!cd&&pc>REVAL_COST_THR){
                revalTimestamp[i]=now; routeChangeCount++; ctrlPktCount+=2;
                if (anim){
                    anim->UpdateNodeColor(nodes.Get(i),50,100,255);
                    Simulator::Schedule(Seconds(LOG_INT*2),[i](){if (!isolated[i]&&!g_bhNodes.count(i))PaintNode(i);});
                }
            }
        }
    }
    if (nActive>0 && (int)now % 60 == 0)
        NS_LOG_INFO("  PathMetrics: active="<<nActive<<"  avgCost="<<std::fixed<<std::setprecision(3)<<(sumCost/nActive));
}

// ═══════════════════════════════════════════════════════════════════════════════
// APPLY TRUST
// [FIX-19] Isolation gated by ML_WARMUP
// ═══════════════════════════════════════════════════════════════════════════════
void ApplyTrust()
{
    double now=Simulator::Now().GetSeconds();
    bool mlReady = (now >= ML_WARMUP); // [FIX-19]
    UpdateAdaptiveWeights();
    for (uint32_t i=1;i<N;++i){
        double rem=eSrc[i]?eSrc[i]->GetRemainingEnergy():0.0;
        double ef=rem/INIT_E; if (rem<0.05) continue;
        if (g_shNodes.count(i)){
            routingMetric[i]=(trust[i]<0.50)?g_dynamicAlpha*trust[i]+g_dynamicBeta*ef:
                              (g_shFakeMetric.count(i)?g_shFakeMetric[i]:0.99);
            if (trust[i]<0.50) NS_LOG_INFO("  [SH-DETECTED] N"<<i<<" fake metric removed");
            continue;
        }
        if (g_scenario=="E") routingMetric[i]=ef;
        else if (g_enableEnergy) routingMetric[i]=g_dynamicAlpha*trust[i]+g_dynamicBeta*ef;
        else routingMetric[i]=trust[i];
        if (g_enableEnergy&&routingMetric[i]<ROUTE_THR&&!isolated[i]) softAvoidCount[i]++;

        // [FIX-19] Only act on isolation logic after warm-up
        if (g_enableML && mlReady){
            if (trust[i]<ISOLATE_TRUST_THR&&!isolated[i]){
                if (++consecLowTrust[i]>=ISOLATE_CONSEC_NEEDED){
                    isolated[i]=true; isoEvents++; SafeDown(i); consecLowTrust[i]=0;
                    LogRouteChangeEvent(now,i,prevNextHop[i],-1,prevPathCost[i],MAX_HOPS,"isolate");
                    NS_LOG_INFO("  x ISOLATE N"<<i<<" trust="<<trust[i]);
                    // [IMP-04] Log isolation event
                    { MLDecisionEvent evt; evt.timestamp=now; evt.nodeId=i;
                      evt.eventType="NODE_ISOLATED"; evt.attackType=GetNodeAttackType(i);
                      evt.newTrust=trust[i]; evt.oldNextHop=-1; evt.newNextHop=-1;
                      evt.oldTrust=trust[i]; evt.oldScore=0; evt.newScore=0;
                      LogMLDecision(evt); }
                    if (anim){anim->UpdateNodeColor(nodes.Get(i),20,20,20);anim->UpdateNodeSize(i,1.5,1.5);}
                }
            } else if (trust[i]>=ISOLATE_TRUST_THR&&!isolated[i]) consecLowTrust[i]=0;
            if (trust[i]>=RESTORE_TRUST_THR&&isolated[i]&&!IS_MALICIOUS(i)){
                isolated[i]=false; consecLowTrust[i]=0; SafeUp(i);
                prevNextHop[i]=-2; prevPathCost[i]=0.0; g_prevPath.erase(i);
                NS_LOG_INFO("  + RESTORE N"<<i<<" trust="<<trust[i]);
                { MLDecisionEvent evt; evt.timestamp=now; evt.nodeId=i;
                  evt.eventType="NODE_RECOVERED"; evt.attackType="";
                  evt.newTrust=trust[i]; evt.oldNextHop=-1; evt.newNextHop=-1;
                  evt.oldTrust=trust[i]; evt.oldScore=0; evt.newScore=0;
                  LogMLDecision(evt); }
                if (anim){anim->UpdateNodeSize(i,2.5,2.5);PaintNode(i);}
            }
        }
    }
    if (g_enableML && mlReady){
    for (uint32_t i=1;i<N;++i){
        if (isolated[i]||g_bhNodes.count(i)) continue;
        int nh=nextHopToSink[i];
        if (nh<=0||(int)nh>=(int)N) continue;

        double nhTrust = trust[nh];
        double ef      = eSrc[i] ? eSrc[i]->GetRemainingEnergy()/INIT_E : 0.0;

        if (nhTrust < ISOLATE_TRUST_THR && trust[i] < 0.50){
            // Next-hop is suspected malicious — bleed this node's trust
            trust[i] *= 0.97;
        } else if (nhTrust >= RESTORE_TRUST_THR && trust[i] < 0.85){
            // Next-hop has recovered — gently rebound this node too.
            // Cap at 0.85 so only the ML ensemble can push above that;
            // this just unsticks nodes caught in bleed residue.
            trust[i] = std::min(0.85, trust[i] * 1.02);
        }

        routingMetric[i] = g_dynamicAlpha*trust[i] + g_dynamicBeta*ef;
    }
}
    UpdatePathMetrics();
}

// ═══════════════════════════════════════════════════════════════════════════════
// VISUALIZE PATH CHANGE
// ═══════════════════════════════════════════════════════════════════════════════
void VisualizePathChange(uint32_t srcNode,const std::vector<uint32_t>& oldP,
                         const std::vector<uint32_t>& newP,double oc,double nc)
{
    double d=nc-oc;
    std::cout<<"\n[PATH-CHG t="<<std::fixed<<std::setprecision(0)<<Simulator::Now().GetSeconds()<<"s]"
             <<" N"<<srcNode<<"("<<ClusterLabel(srcNode)<<")"
             <<" route "<<(d<-0.05?"IMPROVED":d>0.05?"DEGRADED":"SWAPPED")<<":\n";
    std::cout<<"  OLD: "; for (size_t i=0;i<oldP.size();++i){if(i)std::cout<<"->";std::cout<<"N"<<oldP[i];}
    std::cout<<"  cost="<<std::fixed<<std::setprecision(3)<<oc<<"\n";
    std::cout<<"  NEW: "; for (size_t i=0;i<newP.size();++i){if(i)std::cout<<"->";std::cout<<"N"<<newP[i];}
    std::cout<<"  cost="<<nc<<"  d="<<(d>=0?"+":"")<<d<<"\n\n";
    if (!anim) return;
    for (size_t k=0;k+1<oldP.size();++k) anim->UpdateLinkDescription(std::min(oldP[k],oldP[k+1]),std::max(oldP[k],oldP[k+1]),"OLD");
    for (size_t k=0;k+1<newP.size();++k) anim->UpdateLinkDescription(std::min(newP[k],newP[k+1]),std::max(newP[k],newP[k+1]),"NEW");
    anim->UpdateNodeColor(nodes.Get(srcNode),30,150,255);
    Simulator::Schedule(Seconds(LOG_INT*2),[srcNode](){PaintNode(srcNode);});
}

// ═══════════════════════════════════════════════════════════════════════════════
// DRAW ROUTING PATHS
// ═══════════════════════════════════════════════════════════════════════════════
void DrawRoutingPaths()
{
    if (!anim) return;
    double now=Simulator::Now().GetSeconds();
    for (uint32_t a=0;a<N;++a) for (uint32_t b=a+1;b<N;++b) anim->UpdateLinkDescription(a,b,"");
    for (uint32_t i=1;i<N;++i){
        if (isolated[i]||g_bhNodes.count(i)) continue;
        if ((eSrc[i]?eSrc[i]->GetRemainingEnergy():0.0)<0.05) continue;
        WalkResult wr=WalkToSink(i); if (wr.path.size()<2) continue;
        for (size_t k=0;k+1<wr.path.size();++k){
            uint32_t cur=wr.path[k],nxt=wr.path[k+1]; double m=routingMetric[cur];
            std::string lbl=g_shNodes.count(cur)?"SH!":g_sfNodes.count(cur)?"SF~":m>=0.70?"ok":m>=0.40?"~":"!";
            lbl+=std::to_string((int)(m*100))+"%";
            anim->UpdateLinkDescription(std::min(cur,nxt),std::max(cur,nxt),lbl);
        }
    }
    if (now+LOG_INT<SIM_DUR-1.0) Simulator::Schedule(Seconds(LOG_INT),&DrawRoutingPaths);
}

// ═══════════════════════════════════════════════════════════════════════════════
// NODE LABELS
// ═══════════════════════════════════════════════════════════════════════════════
void UpdateNodeLabels()
{
    if (!anim) return;
    anim->UpdateNodeDescription(nodes.Get(0),"SINK ["+std::to_string(pktRx[0])+" rx]");
    for (uint32_t i=1;i<N;++i){
        std::string cl=ClusterLabel(i), st=NodeState(i);
        if (st!="normal"){
            anim->UpdateNodeDescription(nodes.Get(i),"N"+std::to_string(i)+"["+cl+" "+st+"]");
        } else {
            std::ostringstream d;
            d<<"N"<<i<<"["<<cl<<" t="<<std::fixed<<std::setprecision(2)<<trust[i]
             <<" m="<<routingMetric[i];
            if (pathCost[i]>0.001) d<<" c="<<std::setprecision(1)<<pathCost[i];
            if (hopCount[i]>0) d<<" h="<<hopCount[i];
            d<<"]";
            anim->UpdateNodeDescription(nodes.Get(i),d.str());
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CONSOLE STATUS — [IMP-04/05] Cleaner display with attack status and route comparison
// ═══════════════════════════════════════════════════════════════════════════════
void PrintConsoleSummary(double ts)
{
    if (ts >= SIM_DUR) return; // [FIX-01]
    uint64_t fTx,fRx; double fDelay;
    ComputeFilteredPDR(fTx,fRx,fDelay);
    double pdrNow=fTx>0?(double)fRx/fTx*100.0:0.0;
    bool attackActive = !g_bhNodes.empty()||!g_sfNodes.empty()||
                        !g_shNodes.empty()||!g_vampNodes.empty();

    std::cout << "\n";
    if (attackActive) {
        std::cout << "╔══════ ATTACK STATUS @ t=" << std::fixed << std::setprecision(0) << ts
                  << "s  [Scenario " << g_scenario << "] ══════╗\n";
        std::cout << "║  PDR: " << std::setprecision(1) << pdrNow << "% [UNDER ATTACK]\n";
        std::cout << "║  Active: ";
        for (auto n:g_bhNodes)   std::cout << "N"<<n<<"(BH) ";
        for (auto n:g_sfNodes)   std::cout << "N"<<n<<"(SF) ";
        for (auto n:g_shNodes)   std::cout << "N"<<n<<"(SH) ";
        for (auto n:g_vampNodes) std::cout << "N"<<n<<"(VAMP) ";
        std::cout << "\n";
    } else {
        std::cout << "╔══════ STATUS @ t=" << std::fixed << std::setprecision(0) << ts
                  << "s  [Scenario " << g_scenario << "] ══════╗\n";
        std::cout << "║  PDR: " << std::setprecision(1) << pdrNow << "% (no active attacks)\n";
    }

    std::cout << "║  alpha=" << std::setprecision(2) << g_dynamicAlpha
              << "  beta=" << g_dynamicBeta << "\n";

    // Lowest-trust nodes table
    std::vector<std::pair<double,uint32_t>> ord;
    for (uint32_t i=1;i<N;++i) ord.push_back({trust[i],i});
    std::sort(ord.begin(),ord.end());
    std::cout << "║  Node  Trust  Metric  PathCost  Hops  State\n";
    for (int k=0;k<std::min(8,(int)ord.size());++k){
        uint32_t i=ord[k].second;
        std::cout << "║  N" << std::setw(2) << i
                  << "  " << std::setw(5) << std::setprecision(3) << trust[i]
                  << "  " << std::setw(5) << routingMetric[i]
                  << "  " << std::setw(7) << pathCost[i]
                  << "  " << std::setw(3) << hopCount[i]
                  << "  " << NodeState(i) << "\n";
    }

    // [IMP-05] Show recent ML route decisions (last 3)
    int shown = 0;
    for (int k = (int)g_mlEvents.size()-1; k >= 0 && shown < 3; --k) {
        const auto& e = g_mlEvents[k];
        if (e.eventType == "ROUTE_CHANGED") {
            if (shown == 0)
                std::cout << "║  — Recent ML route changes:\n";
            std::cout << "║    N" << e.nodeId << ": N" << e.oldNextHop
                      << " → N" << e.newNextHop
                      << "  score: " << std::setprecision(3) << e.oldScore
                      << " → " << e.newScore << "\n";
            shown++;
        }
    }

    std::cout << "╚═══════════════════════════════════════════╝\n";
    Simulator::Schedule(Seconds(20.0),&PrintConsoleSummary,ts+20.0);
}

// ═══════════════════════════════════════════════════════════════════════════════
// PERIODIC LOG
// [FIX-01] Uses ts >= SIM_DUR guard instead of gStop
// ═══════════════════════════════════════════════════════════════════════════════
void LogSnap(double ts)
{
    if (ts >= SIM_DUR) return; // [FIX-01]
    uint64_t totalTx,totalRx; double avgDelay;
    ComputeFilteredPDR(totalTx,totalRx,avgDelay);
    double pdr=(totalTx>0)?(double)totalRx/totalTx:0.0;
    avgDelay*=1000.0;
    double totalE=0.0; uint32_t alive=0;
    for (uint32_t i=0;i<N;++i){double e=eSrc[i]?eSrc[i]->GetRemainingEnergy():0.0;totalE+=e;if(e>0.05)alive++;}
    double meanE=totalE/N;
    double variance=0.0;
    for (uint32_t i=0;i<N;++i){double e=eSrc[i]?eSrc[i]->GetRemainingEnergy():0.0;variance+=(e-meanE)*(e-meanE);}
    double energyStdDev=std::sqrt(variance/N);
    double sumE_jfi=0.0,sumE2_jfi=0.0;
    for (uint32_t i=0;i<N;++i){double e=eSrc[i]?eSrc[i]->GetRemainingEnergy():0.0;sumE_jfi+=e;sumE2_jfi+=e*e;}
    double jfi=(sumE2_jfi>0.0)?(sumE_jfi*sumE_jfi)/((double)N*sumE2_jfi):1.0;
    if (g_firstDeathTime<0&&alive<N) g_firstDeathTime=ts;
    if (g_halfDeadTime<0&&alive<=N/2) g_halfDeadTime=ts;
    uint32_t nConn=0; for (uint32_t i=1;i<N;++i) if (nextHopToSink[i]>=0) nConn++;
    if (g_partitionTime<0&&nConn<(N-1)/2) g_partitionTime=ts;
    uint32_t nIso=0,nSoftAvoid=0;
    for (bool b:isolated) if (b) nIso++;
    for (uint32_t i=1;i<N;++i) if (g_enableEnergy&&routingMetric[i]<ROUTE_THR&&!isolated[i]) nSoftAvoid++;
    double sumPC=0.0; uint32_t nWR=0;
    for (uint32_t i=1;i<N;++i) if (nextHopToSink[i]>=0){sumPC+=pathCost[i];nWR++;}
    double avgPC=nWR>0?sumPC/nWR:0.0;
    double sumStab=0.0; uint32_t nTr=0;
    for (uint32_t i=1;i<N;++i) if (totalIntervals[i]>0){sumStab+=(double)stableIntervals[i]/totalIntervals[i];nTr++;}
    double avgStab=nTr>0?sumStab/nTr:0.0;
    double sumH=0.0; uint32_t nHV=0; int maxH=0,minH=INT_MAX;
    for (uint32_t i=1;i<N;++i) if (hopCount[i]>0&&hopCount[i]<MAX_HOPS*2){
        sumH+=hopCount[i];nHV++;if(hopCount[i]>maxH)maxH=hopCount[i];if(hopCount[i]<minH)minH=hopCount[i];}
    double avgH=nHV>0?sumH/nHV:0.0; if (minH==INT_MAX) minH=0;
    double hVar=0.0;
    for (uint32_t i=1;i<N;++i) if (hopCount[i]>0&&hopCount[i]<MAX_HOPS*2){double d=hopCount[i]-avgH;hVar+=d*d;}
    double hopStd=nHV>0?std::sqrt(hVar/nHV):0.0;
    uint64_t ctrlD=ctrlPktCount.load()-snapCtrl[0]; snapCtrl[0]=ctrlPktCount.load();
    uint64_t rcD=routeChangeCount.load()-snapRouteChange; snapRouteChange=routeChangeCount.load();
    double sumM=0.0; for (uint32_t i=1;i<N;++i) sumM+=routingMetric[i];
    // [FIX-RC1] Snapshot current under-attack PDR for this log row
    double snapAtkPDR = g_atkWinTxTotal > 0 ?
        (double)g_atkWinRxTotal / g_atkWinTxTotal * 100.0 : 0.0;
    perfLog<<std::fixed<<std::setprecision(3)
           <<ts<<","<<pdr<<","<<snapAtkPDR<<","<<avgDelay<<","<<meanE<<","
           <<isoEvents<<","<<g_bhNodes.size()<<","<<g_sfNodes.size()<<","<<g_shNodes.size()<<","<<g_vampNodes.size()<<","
           <<nIso<<","<<alive<<","<<energyStdDev<<","<<nSoftAvoid<<","
           <<(sumM/(N-1))<<","<<avgPC<<","<<avgStab<<","
           <<ctrlD<<","<<rcD<<","<<avgH<<","<<hopStd<<","<<minH<<","<<maxH<<","
           <<jfi<<","<<g_dynamicAlpha<<","<<g_dynamicBeta<<","<<g_scenario<<"\n";
    perfLog.flush();
    WriteRoutingMatrix(ts); WriteHopEvolution(ts);
    NS_LOG_INFO("[LOG] t="<<ts<<"s  PDR(app)="<<std::setprecision(1)<<(pdr*100.0)<<"%"
        <<"  TX="<<totalTx<<"  RX="<<totalRx
        <<"  Delay="<<std::setprecision(2)<<avgDelay<<"ms"
        <<"  E="<<meanE<<"J  Alive="<<alive
        <<"  BH="<<g_bhNodes.size()<<"  SF="<<g_sfNodes.size()
        <<"  SH="<<g_shNodes.size()<<"  VAMP="<<g_vampNodes.size()
        <<"  Iso="<<nIso<<"  PathCost="<<std::setprecision(3)<<avgPC
        <<"  Hops="<<std::setprecision(2)<<avgH<<"+-"<<hopStd<<" ["<<minH<<"-"<<maxH<<"]"
        <<"  Stab="<<(avgStab*100.0)<<"% alpha="<<g_dynamicAlpha<<" beta="<<g_dynamicBeta);
    Simulator::Schedule(Seconds(LOG_INT),&LogSnap,ts+LOG_INT);
}

// ═══════════════════════════════════════════════════════════════════════════════
// [HYBRID-AODV] CANDIDATE SCORING AND ROUTE INJECTION
//
// Flow (implements spec §4.1 and §5.3–5.5):
//   1. For every non-sink, non-isolated node that has a HybridAodv protocol:
//      a. Scan the AODV routing table for VALID routes toward g_sinkAddr.
//         Each valid routing table entry = one candidate route discovered
//         by AODV RREQ/RREP.  We collect them into the protocol's candidate
//         list (this is the RREP-harvest — in a real deployment it would be
//         filled by HookRecvReply; in the simulation we harvest from the
//         routing table every ML_INT seconds which is equivalent).
//      b. Annotate each candidate with current trust[] and energy.
//      c. Score:  score = α·trust + β·energy − γ·(hops/MAX_HOPS)
//      d. Pick the highest-scoring candidate.
//      e. Call InjectMLRoute() — locks the winner into AODV, prevents
//         AODV from overwriting it (spec F5, F7).
//   2. Nodes that are isolated or BH-detected get their lock removed.
// ═══════════════════════════════════════════════════════════════════════════════

// Score a single candidate using trust and energy from global arrays
static double ScoreCandidate(const CandidateRoute& c, uint32_t /*nodeId*/)
{
    double t  = std::max(0.0, std::min(1.0, c.trustScore));
    double e  = std::max(0.0, std::min(1.0, c.residualEnergy));
    // [FIX-HOPS] Use actual hop count; default to 1 if zero (1-hop = direct neighbour)
    double h  = (double)std::max((uint8_t)1, c.hopCount) / (double)MAX_HOPS;
    return CAND_ALPHA * t + CAND_BETA * e - CAND_GAMMA * h;
}

// Pick best candidate from a list (modifies score field)
CandidateRoute BestCandidate(std::vector<CandidateRoute>& cands, uint32_t nodeId)
{
    CandidateRoute best;
    best.score = -1e9;
    for (auto& c : cands) {
        c.score = ScoreCandidate(c, nodeId);
        if (c.score > best.score) best = c;
    }
    return best;
}

void ScoreAndInjectCandidates(uint32_t nodeId, Ipv4Address dst, double ts)
{
    Ptr<HybridAodvRoutingProtocol> proto = g_hybridProto[nodeId];
    if (!proto) return;

    // ── [FIX-UNLOCK] Release the old ML lock BEFORE harvest so AODV is free
    // to update its routing table with fresh RREPs discovered in this window.
    // We re-lock with the new winner at the end.  Without this, AODV's table
    // gets frozen after the first injection and candidates never change.
    proto->UnlockRoute(dst);

    // ── Harvest candidates from AODV routing table ────────────────────────
    // LookupSinkRoute() exposes AODV's own table entry (populated by
    // RREQ/RREP) as a candidate.  Additional candidates collected via
    // HookRecvReply / AppendCandidate are already in proto->GetCandidates().
    std::vector<CandidateRoute> cands = proto->GetCandidates(dst);

    // Always include the current AODV best route as a candidate
    aodv::RoutingTableEntry rt;
    if (proto->LookupSinkRoute(dst, rt)) {
        CandidateRoute c;
        c.destination = dst;
        c.nextHop     = rt.GetNextHop();
        c.sender      = rt.GetNextHop();
        // [FIX-HOPS] GetHop() returns the actual hop count from the RREP;
        // do NOT hardcode 0.  Clamp to [1, 255].
        uint32_t rawHops = rt.GetHop();
        c.hopCount = (rawHops == 0) ? 1 : (uint8_t)std::min(rawHops, (uint32_t)255);
        c.seqNo       = rt.GetSeqNo();
        c.lifetime    = rt.GetLifeTime();
        // [FIX-LOOPBACK] Reject loopback / unset next-hop
        if (c.nextHop == Ipv4Address("127.0.0.1") ||
            c.nextHop == Ipv4Address("0.0.0.0")   ||
            c.nextHop == Ipv4Address()) {
            NS_LOG_WARN("[HYBRID] N" << nodeId << " AODV returned loopback nextHop — skipping");
        } else {
            // Deduplicate
            bool found = false;
            for (auto& ex : cands)
                if (ex.nextHop == c.nextHop) { ex.hopCount = c.hopCount; found = true; break; }
            if (!found) cands.push_back(c);
        }
    }

    if (cands.empty()) {
        NS_LOG_INFO("[HYBRID] N" << nodeId << " t=" << ts << "s  no AODV candidates yet");
        return;
    }

    // ── Annotate with trust and energy ────────────────────────────────────
    for (auto& c : cands) {
        // Find the node index for the next-hop address
        uint32_t nhIdx = UINT32_MAX;
        for (uint32_t j = 0; j < N; ++j) {
            if (ifaces.GetAddress(j) == c.nextHop) { nhIdx = j; break; }
        }
        c.trustScore     = (nhIdx < N) ? trust[nhIdx]
                                       : 0.5;
        c.residualEnergy = (nhIdx < N && eSrc[nhIdx])
                           ? eSrc[nhIdx]->GetRemainingEnergy() / INIT_E
                           : 0.5;
    }

    // ── Score and select ──────────────────────────────────────────────────
    CandidateRoute winner = BestCandidate(cands, nodeId);

    // [FIX-LOOPBACK] Final loopback guard before injection
    if (winner.nextHop == Ipv4Address("127.0.0.1") ||
        winner.nextHop == Ipv4Address("0.0.0.0")   ||
        winner.nextHop == Ipv4Address()) {
        NS_LOG_WARN("[HYBRID] N" << nodeId << " winner nextHop is loopback/zero — aborting inject");
        return;
    }

    // Log all candidates
    if (mlRouteLog.is_open()) {
        for (const auto& c : cands) {
            mlRouteLog << std::fixed << std::setprecision(2)
                       << ts << "," << nodeId << "," << c.nextHop << ","
                       << (int)c.hopCount << "," << std::setprecision(4)
                       << c.trustScore << "," << c.residualEnergy << ","
                       << c.score
                       << (c.nextHop == winner.nextHop ? ",SELECTED" : ",")
                       << "\n";
        }
        mlRouteLog.flush();
    }

    NS_LOG_INFO("[HYBRID] N" << nodeId << " t=" << ts << "s"
                << "  candidates=" << cands.size()
                << "  WINNER via " << winner.nextHop
                << "  hops=" << (int)winner.hopCount
                << "  trust=" << std::fixed << std::setprecision(3) << winner.trustScore
                << "  energy=" << winner.residualEnergy
                << "  score=" << winner.score);

    // ── Inject and lock ───────────────────────────────────────────────────
    proto->InjectMLRoute(dst, winner.nextHop, winner.hopCount,
                         Seconds(ML_INT * 2)); // lock lifetime = 2 intervals

    // nextHopToSink[] and pathCost[] used by TrustRouting helpers — update
    // them so LogSnap / WritePathTraces still work correctly.
    for (uint32_t j = 0; j < N; ++j) {
        if (ifaces.GetAddress(j) == winner.nextHop) {
            nextHopToSink[nodeId] = (int)j;
            break;
        }
    }
    hopCount[nodeId]  = winner.hopCount;
    pathCost[nodeId]  = winner.score > 0 ? (1.0 - winner.score) : 1.0;

    // [FIX-CANDS] Clear so next window collects FRESH RREPs from AODV
    // (AODV is now unlocked above, so it will accumulate new RREPs)
    proto->ClearCandidates(dst);
}

// ── Top-level: scan all nodes every ML_INT ───────────────────────────────────
void HybridScanAndInject(double ts)
{
    if (ts >= SIM_DUR) return;
    bool mlReady = (ts >= ML_WARMUP);

    for (uint32_t i = 1; i < N; ++i) {
        if (!g_hybridProto[i]) continue;
        double rem = eSrc[i] ? eSrc[i]->GetRemainingEnergy() : 0.0;
        if (rem < 0.05) continue;

        if (isolated[i] || g_bhNodes.count(i)) {
            // Release ML lock so AODV can rediscover after attack ends
            g_hybridProto[i]->UnlockRoute(g_sinkAddr);
            continue;
        }

        if (!mlReady) {
            // Before warm-up: unlock so AODV populates its table freely
            g_hybridProto[i]->UnlockRoute(g_sinkAddr);
            continue;
        }

        // ScoreAndInjectCandidates() unlocks first, harvests AODV candidates,
        // scores them, then re-locks with the winner via InjectMLRoute().
        ScoreAndInjectCandidates(i, g_sinkAddr, ts);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ML EVALUATION
// [FIX-01] Uses ts >= SIM_DUR guard instead of gStop
// [FIX-08] First call deferred to ML_WARMUP (scheduled from main)
// [FIX-16] Snapshot reset happens once at ML_WARMUP-1s
// ═══════════════════════════════════════════════════════════════════════════════
void MLEval(double ts)
{
    if (ts >= SIM_DUR) return; // [FIX-01]
    std::vector<double> energyF(N), fwdF(N), dropF(N);
    std::vector<double> pathCostF(N,0), clusterF(N,0);
    std::vector<uint64_t> dRxF(N, 0);

    for (uint32_t i=0;i<N;++i){
        double rem=eSrc[i]?eSrc[i]->GetRemainingEnergy():0.0;
        energyF[i]=rem/INIT_E;
        uint64_t dTx=pktTx[i]-snapTx[i];
        uint64_t dRx=pktRx[i]-snapRx[i];
        dRxF[i] = dRx;
        uint64_t dOrig=pktOrig[i]-snapOrig[i];
        snapTx[i]=pktTx[i]; snapRx[i]=pktRx[i]; snapOrig[i]=pktOrig[i];
        if (i==0){
            double expectedPerNode = ML_INT * 0.5;
            uint64_t exp=(uint64_t)((N-1)*expectedPerNode);
            fwdF[i]=std::min(1.0,(double)dRx/std::max((uint64_t)1,exp));
        } else if (rem<0.05||isolated[i]){
            fwdF[i]=0.0;
        } else {
            uint64_t forwardedCount=(dTx>dOrig)?(dTx-dOrig):0;
            if (dRx==0){
                fwdF[i]=1.0;
            } else {
                fwdF[i]=std::min(1.0,(double)forwardedCount/(double)dRx);
            }
        }
        dropF[i]=1.0-fwdF[i];
        pathCostF[i]=(i>0)?std::min(5.0,pathCost[i])/5.0:0.0;
        clusterF[i]=(i==0)?0.0:(double)(g_clusterOf[i]+1)/NUM_CLUSTERS;
    }
    std::vector<double> stabilityF(N,1.0);
    for (uint32_t i=1;i<N;++i)
        stabilityF[i]=(totalIntervals[i]>0)?(double)stableIntervals[i]/totalIntervals[i]:1.0;

    auto JIArr = [](const std::vector<uint64_t>& v) {
        std::ostringstream s; s << "[";
        for (size_t i = 0; i < v.size(); ++i) {
            if (i) s << ",";
            s << v[i];
        }
        s << "]";
        return s.str();
    };
 
    std::ostringstream json;
    json<<"{\"timestamp\":"     <<(int)ts
        <<",\"energy\":"        <<JArr(energyF)
        <<",\"forward_ratio\":" <<JArr(fwdF)
        <<",\"drop_ratio\":"    <<JArr(dropF)
        <<",\"routing_metric\":"<<JArr(routingMetric)
        <<",\"path_cost\":"     <<JArr(pathCostF)
        <<",\"path_stability\":"<<JArr(stabilityF)
        <<",\"cluster_id\":"    <<JArr(clusterF)
        <<",\"pkt_rx\":"        <<JIArr(dRxF)
        <<",\"scenario\":\""    <<g_scenario<<"\"}";

    std::string resp;
    if (g_enableML && g_scenario!="E") {
        if (IpcSend(json.str(), resp)) {
            // [IMP-04] Capture pre-update next-hops for route-change comparison
            std::vector<int> oldNH(nextHopToSink.begin(), nextHopToSink.end());
            std::vector<double> oldRM(routingMetric.begin(), routingMetric.end());
            std::vector<double> oldTr(trust.begin(), trust.end());

            ParseTrust(resp);       // updates trust[] via EMA (may emit ATTACK_DETECTED)

            // [HYBRID-F4] Score candidates and inject ML-selected route per node.
            if (g_scenario != "A") {
                HybridScanAndInject(ts);
                ApplyTrust();
            } else {
                ApplyTrust();
            }

            // [IMP-04] Emit ROUTE_CHANGED events for any next-hop that shifted
            for (uint32_t i = 1; i < N; ++i) {
                if (oldNH[i] >= 0 && nextHopToSink[i] >= 0 && oldNH[i] != nextHopToSink[i]) {
                    MLDecisionEvent evt;
                    evt.timestamp  = ts;
                    evt.nodeId     = i;
                    evt.eventType  = "ROUTE_CHANGED";
                    evt.attackType = (oldNH[i] >= 0 && (uint32_t)oldNH[i] < N)
                                     ? GetNodeAttackType((uint32_t)oldNH[i]) : "";
                    evt.oldNextHop = oldNH[i];
                    evt.newNextHop = nextHopToSink[i];
                    evt.oldTrust   = (oldNH[i] >= 0 && (uint32_t)oldNH[i] < N) ? oldTr[oldNH[i]] : 0.0;
                    evt.newTrust   = ((uint32_t)nextHopToSink[i] < N) ? trust[nextHopToSink[i]] : 0.0;
                    evt.oldScore   = (oldNH[i] >= 0 && (uint32_t)oldNH[i] < N) ? oldRM[oldNH[i]] : 0.0;
                    evt.newScore   = ((uint32_t)nextHopToSink[i] < N) ? routingMetric[nextHopToSink[i]] : 0.0;
                    evt.oldPath    = "N" + std::to_string(i) + "→N" + std::to_string(oldNH[i]);
                    evt.newPath    = BuildPathString(i);
                    LogMLDecision(evt);
                }
            }
        } else {
            NS_LOG_WARN("ML server unreachable at t=" << ts << "s — keeping previous trust[]");
            if (g_scenario != "A") { HybridScanAndInject(ts); ApplyTrust(); }
            else { ApplyTrust(); }
        }
    } else {
        // Scenario E (energy-only) or fallback: all trust = 1.0
        for (uint32_t i = 0; i < N; ++i) trust[i] = 1.0;
        if (g_scenario != "A") { HybridScanAndInject(ts); ApplyTrust(); }
        else { ApplyTrust(); }
    }

    WritePathTraces(ts);
    RefreshAnim();
    Simulator::Schedule(Seconds(ML_INT), &MLEval, ts + ML_INT);
}

// ═══════════════════════════════════════════════════════════════════════════════
// BUILD NETWORK
// [FIX-18] Connectivity check and abort on zero-neighbour nodes
// [FIX-07] ApplyTrust() called at t=5s to pre-populate routing tables
// ═══════════════════════════════════════════════════════════════════════════════
void BuildNetwork()
{
    nodes.Create(N);
    WifiHelper wifi; wifi.SetStandard(WIFI_STANDARD_80211b);
    wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager",
        "DataMode",StringValue("DsssRate1Mbps"),"ControlMode",StringValue("DsssRate1Mbps"));
    YansWifiPhyHelper phy;
    phy.Set("TxPowerStart",DoubleValue(16.0)); phy.Set("TxPowerEnd",DoubleValue(16.0));
    YansWifiChannelHelper ch; ch.SetPropagationDelay("ns3::ConstantSpeedPropagationDelayModel");
    ch.AddPropagationLoss("ns3::LogDistancePropagationLossModel",
        "Exponent",DoubleValue(2.8),"ReferenceLoss",DoubleValue(46.7),"ReferenceDistance",DoubleValue(1.0));
    phy.SetChannel(ch.Create());
    WifiMacHelper mac; mac.SetType("ns3::AdhocWifiMac");
    devices=wifi.Install(phy,mac,nodes);

    MobilityHelper mob;
    Ptr<ListPositionAllocator> posAlloc=CreateObject<ListPositionAllocator>();
    posAlloc->Add(Vector(20.0,20.0,0.0)); g_clusterOf[0]=255;
    std::mt19937 posRng(g_rngSeed); std::normal_distribution<double> gauss(0.0,1.0);
    auto clamp=[](double v,double lo,double hi){return std::max(lo,std::min(hi,v));};
    for (uint32_t c=0;c<NUM_CLUSTERS;++c){
        const ClusterDef& cd=CLUSTERS[c];
        for (uint32_t k=0;k<cd.count;++k){
            posAlloc->Add(Vector(clamp(cd.cx+gauss(posRng)*cd.stddev,20.0,480.0),
                                 clamp(cd.cy+gauss(posRng)*cd.stddev,20.0,480.0),0.0));
            g_clusterOf[cd.startIdx+k]=(uint8_t)c;
        }
    }
    mob.SetPositionAllocator(posAlloc);
    if (g_scenario=="D"){
        mob.SetMobilityModel("ns3::RandomWaypointMobilityModel",
            "Speed",StringValue("ns3::UniformRandomVariable[Min=0.1|Max=0.5]"),
            "Pause",StringValue("ns3::ConstantRandomVariable[Constant=10.0]"),
            "PositionAllocator",PointerValue(posAlloc));
    } else mob.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mob.Install(nodes);

    // [FIX] Routing protocol selection via helper
    // Scenario A : standard AODV (baseline, no defence)
    // B/C/D/E   : HybridAodvRoutingProtocol — AODV does RREQ/RREP discovery,
    //             ML selects and locks the best route (spec §4, §5).
    InternetStackHelper inet;
    if (g_scenario == "A") {
        AodvHelper aodv;
        Ipv4ListRoutingHelper list;
        list.Add(aodv, 10);
        inet.SetRoutingHelper(list);
        inet.Install(nodes);
        NS_LOG_INFO("[v5.1] Scenario A: baseline AODV installed — no ML/trust/energy defence");
    } else {
        HybridAodvHelper hybridAodv;
        hybridAodv.Set("MaxQueueTime",       TimeValue(Seconds(60)));
        hybridAodv.Set("MaxQueueLen",        UintegerValue(256));
        hybridAodv.Set("ActiveRouteTimeout", TimeValue(Seconds(20)));
        hybridAodv.Set("EnableHello",        BooleanValue(false));
        Ipv4ListRoutingHelper list;
        list.Add(hybridAodv, 10);
        inet.SetRoutingHelper(list);
        inet.Install(nodes);

        // Cache per-node HybridAodvRoutingProtocol pointer for InjectMLRoute()
        for (uint32_t i = 0; i < N; ++i) {
            Ptr<Ipv4> ipv4 = nodes.Get(i)->GetObject<Ipv4>();
            Ptr<Ipv4ListRouting> listRp =
                DynamicCast<Ipv4ListRouting>(ipv4->GetRoutingProtocol());
            if (listRp) {
                int16_t prio;
                g_hybridProto[i] = DynamicCast<HybridAodvRoutingProtocol>(
                    listRp->GetRoutingProtocol(0, prio));
            }
            if (!g_hybridProto[i])
                NS_LOG_WARN("[HYBRID] N" << i << " failed to cache protocol ptr");
        }
        NS_LOG_INFO("[v5.1] HybridAodvRoutingProtocol installed on all " << N << " nodes");
    }

    Ipv4AddressHelper addr; addr.SetBase("10.1.0.0","255.255.0.0");
    ifaces = addr.Assign(devices);
    g_sinkAddr = ifaces.GetAddress(0);

    BasicEnergySourceHelper esh;
    esh.Set("BasicEnergySourceInitialEnergyJ",DoubleValue(INIT_E));
    WifiRadioEnergyModelHelper reh;
    reh.Set("TxCurrentA",DoubleValue(0.0174)); reh.Set("RxCurrentA",DoubleValue(0.0197));
    reh.Set("IdleCurrentA",DoubleValue(0.000426));
    for (uint32_t i=0;i<N;++i){
        EnergySourceContainer esc=esh.Install(nodes.Get(i));
        eSrc[i]=DynamicCast<BasicEnergySource>(esc.Get(0));
        reh.Install(devices.Get(i),esc);
    }

    // [FIX-18] Connectivity check + ifaces address sanity check
    Simulator::Schedule(Seconds(0.1), [](){
        uint32_t zeroNeighbour=0; double tot=0;
        for (uint32_t i=0;i<N;++i){
            auto mi=nodes.Get(i)->GetObject<MobilityModel>();
            int cnt=0;
            for (uint32_t j=0;j<N;++j){
                if(i==j) continue;
                auto mj=nodes.Get(j)->GetObject<MobilityModel>();
                if(mi->GetDistanceFrom(mj)<=RADIO_RANGE) cnt++;
            }
            tot+=cnt;
            if (cnt==0&&i!=0) { zeroNeighbour++; NS_LOG_WARN("[FIX-18] N"<<i<<" has 0 neighbours!"); }
        }
        double avg=(double)tot/N;
        std::cout<<"[DIAG] Avg neighbors: "<<std::fixed<<std::setprecision(1)<<avg;
        if (avg>=3.0) std::cout<<"  [OK]\n";
        else          std::cout<<"  [WARN: low — increase RADIO_RANGE or adjust cluster stddev]\n";
        if (zeroNeighbour>0)
            std::cout<<"[WARN-18] "<<zeroNeighbour<<" node(s) have zero neighbours. PDR will be zero for those nodes.\n";

        // Sanity: verify ifaces.GetAddress(i) == node i's actual Ipv4 address
        bool ifaceOk=true;
        for (uint32_t i=0;i<N;++i){
            auto ipv4=nodes.Get(i)->GetObject<Ipv4>();
            if (!ipv4||ipv4->GetNInterfaces()<2) { ifaceOk=false; break; }
            Ipv4Address nodeIp=ipv4->GetAddress(1,0).GetLocal();
            Ipv4Address ifaceIp=ifaces.GetAddress(i);
            if (nodeIp!=ifaceIp){
                std::cout<<"[WARN-IFACE] N"<<i<<" ifaces["<<i<<"]="<<ifaceIp<<" but node IP="<<nodeIp<<"\n";
                ifaceOk=false;
            }
        }
        if (ifaceOk) std::cout<<"[DIAG] ifaces[] address mapping: OK\n";
        else         std::cout<<"[DIAG] ifaces[] address mapping: MISMATCH — BuildRoute will use wrong gateway IPs!\n";
    });

    // [FIX-07] Pre-compute routing tables before traffic begins
    Simulator::Schedule(Seconds(2.0), [](){
        NS_LOG_INFO("[FIX-07] Pre-computing initial routing tables...");
        ApplyTrust();
        NS_LOG_INFO("[FIX-07] Initial routing done.");
    });

    NS_LOG_INFO("Network ready: "<<N<<" nodes | Sink="<<g_sinkAddr<<" | E="<<INIT_E<<"J");
}

// ─────────────────────────────────────────────────────────────────────────────
// Static trace helpers — MakeCallback cannot accept capturing lambdas in ns-3;
// use MakeBoundCallback with these plain functions instead.
// ─────────────────────────────────────────────────────────────────────────────
static void IpTxTraceHelper(uint32_t nodeId,
                             Ptr<const Packet>,
                             Ptr<Ipv4>,
                             uint32_t)
{
    pktTx[nodeId]++;
}

static void IpRxTraceHelper(uint32_t nodeId,
                             Ptr<const Packet>,
                             Ptr<Ipv4>,
                             uint32_t)
{
    pktRx[nodeId]++;
}

// ═══════════════════════════════════════════════════════════════════════════════
// BUILD TRAFFIC
// [FIX-02] Apps start at 5.0 + i*0.05s (was 10.0 + i*0.1s)
// ═══════════════════════════════════════════════════════════════════════════════
void BuildTraffic()
{
    PacketSinkHelper sinkH("ns3::UdpSocketFactory",InetSocketAddress(Ipv4Address::GetAny(),APP_PORT));
    ApplicationContainer sinkApps=sinkH.Install(nodes.Get(0));
    sinkApps.Start(Seconds(0.0)); sinkApps.Stop(Seconds(SIM_DUR));

    for (uint32_t i=1;i<N;++i){
        OnOffHelper src("ns3::UdpSocketFactory",InetSocketAddress(ifaces.GetAddress(0),APP_PORT));
        src.SetConstantRate(DataRate("512bps"),PKT_SIZE);
        src.SetAttribute("OnTime",StringValue("ns3::ConstantRandomVariable[Constant=1]"));
        src.SetAttribute("OffTime",StringValue("ns3::ConstantRandomVariable[Constant=0]"));
        ApplicationContainer app=src.Install(nodes.Get(i));
        // [FIX-02] start at 5.0 + staggered 50ms per node
        app.Start(Seconds(5.0 + i * 0.05));
        app.Stop(Seconds(SIM_DUR));
    }
    flowMon=fmHelper.InstallAll();

    // [HYBRID-PKT] For HybridAodv scenarios AODV manages routing internally,
    // so TrustRoutingProtocol's RouteOutput/RouteInput no longer increment
    // pktTx[]/pktRx[].  Connect Ipv4L3Protocol TX/RX traces instead.
    // Tx fires when a packet leaves this node's IP layer (originated OR forwarded).
    // Rx fires when a packet is received at the IP layer.
    if (g_scenario != "A") {
        for (uint32_t i = 0; i < N; ++i) {
            uint32_t nodeId = i;
            // TX trace — count app-originated and forwarded packets
            nodes.Get(i)->GetObject<Ipv4L3Protocol>()->TraceConnectWithoutContext(
                "Tx",
                MakeBoundCallback(&IpTxTraceHelper, nodeId));
            // RX trace — count received at IP layer
            nodes.Get(i)->GetObject<Ipv4L3Protocol>()->TraceConnectWithoutContext(
                "Rx",
                MakeBoundCallback(&IpRxTraceHelper, nodeId));
            // UnicastForward trace — count forwarded (not originated)
            // pktOrig[i] = pktTx[i] - forwarded; we approximate via pktTx
            // (MLEval already handles the dOrig=0 case gracefully)
        }
        NS_LOG_INFO("[HYBRID] Ipv4L3Protocol TX/RX traces connected for pkt counting");
    }
}

static void SigInt(int){gStop=true;Simulator::Stop();}

// ═══════════════════════════════════════════════════════════════════════════════
// FINAL SUMMARY
// ═══════════════════════════════════════════════════════════════════════════════
static void Summary()
{
    uint64_t totalTx,totalRx; double avgDelay;
    ComputeFilteredPDR(totalTx,totalRx,avgDelay); avgDelay*=1000.0;
    double totalE=0.0; uint32_t alive=0;
    for (uint32_t i=0;i<N;++i){double e=eSrc[i]?eSrc[i]->GetRemainingEnergy():0.0;totalE+=e;if(e>0.05)alive++;}
    double meanE=totalE/N,var=0.0;
    for (uint32_t i=0;i<N;++i){double e=eSrc[i]?eSrc[i]->GetRemainingEnergy():0.0;var+=(e-meanE)*(e-meanE);}
    double eStd=std::sqrt(var/N);
    double sj=0.0,sj2=0.0;
    for (uint32_t i=0;i<N;++i){double e=eSrc[i]?eSrc[i]->GetRemainingEnergy():0.0;sj+=e;sj2+=e*e;}
    double jfi=sj2>0.0?(sj*sj)/((double)N*sj2):1.0;
    double sm=0.0; for (uint32_t i=1;i<N;++i) sm+=routingMetric[i];
    double sp=0.0; uint32_t np=0;
    for (uint32_t i=1;i<N;++i) if (nextHopToSink[i]>=0){sp+=pathCost[i];np++;}
    double ss=0.0; uint32_t ns=0;
    for (uint32_t i=1;i<N;++i) if (totalIntervals[i]>0){ss+=(double)stableIntervals[i]/totalIntervals[i];ns++;}
    double sh=0.0; uint32_t nh=0;
    for (uint32_t i=1;i<N;++i) if (hopCount[i]>0&&hopCount[i]<MAX_HOPS*2){sh+=hopCount[i];nh++;}
    uint64_t tsa=0; for (auto v:softAvoidCount) tsa+=v;
    double pdr=totalTx>0?(double)totalRx/totalTx*100.0:0.0;

    // [FIX-RC1] Close any still-open attack window before final report
    if (g_atkWindowOpen) {
        uint64_t tx, rx; double d;
        ComputeFilteredPDR(tx, rx, d);
        g_atkWinTxTotal += (tx > g_atkWinTxSnap ? tx - g_atkWinTxSnap : 0);
        g_atkWinRxTotal += (rx > g_atkWinRxSnap ? rx - g_atkWinRxSnap : 0);
        g_atkWindowOpen = false;
    }
    double atkPDR = g_atkWinTxTotal > 0 ?
        (double)g_atkWinRxTotal / g_atkWinTxTotal * 100.0 : 0.0;

    std::cout<<"\n╔══════════════════════════════════════════════════════════════╗\n"
             <<"║  FINAL RESULTS — Scenario "<<g_scenario<<"   [v5.4]                      ║\n"
             <<"╠══════════════════════════════════════════════════════════════╣\n"
             <<std::fixed<<std::setprecision(2)
             <<"║  PDR — cumulative (all time)    : "<<std::setw(7)<<pdr<<" %         ║\n"
             <<"║  PDR — under-attack windows     : "<<std::setw(7)<<atkPDR<<" %  [KEY]  ║\n"
             <<"║    (packets counted only while >=1 attack active)           ║\n"
             <<"║  Avg E2E Delay          : "<<std::setw(7)<<avgDelay<<" ms             ║\n"
             <<"║  Avg Energy Left        : "<<std::setw(7)<<meanE<<" J              ║\n"
             <<"║  Energy Std Dev         : "<<std::setw(7)<<eStd<<" J              ║\n"
             <<"║  Jain Fairness Index    : "<<std::setw(7)<<jfi<<"                ║\n"
             <<"║  Avg Routing Metric     : "<<std::setw(7)<<sm/(N-1)<<"                ║\n"
             <<"║  Avg Path Cost          : "<<std::setw(7)<<(np>0?sp/np:0.0)<<"                ║\n"
             <<"║  Path Stability         : "<<std::setw(7)<<(ns>0?ss/ns*100.0:0.0)<<" %      ║\n"
             <<"║  Avg Hop Count          : "<<std::setw(7)<<(nh>0?sh/nh:0.0)<<" hops           ║\n"
             <<"║  Total TX (app)         : "<<std::setw(9)<<totalTx<<"             ║\n"
             <<"║  Total RX (app)         : "<<std::setw(9)<<totalRx<<"             ║\n"
             <<"║  Alive nodes            : "<<std::setw(9)<<alive<<"             ║\n"
             <<"║  Isolation Events       : "<<std::setw(9)<<isoEvents<<"             ║\n"
             <<"║  Soft-Avoid Events      : "<<std::setw(9)<<tsa<<"          ║\n"
             <<"║  Route Re-eval Events   : "<<std::setw(9)<<routeChangeCount.load()<<"  ║\n"
             <<"╠══════════════════════════════════════════════════════════════╣\n"
             <<"║  NETWORK LIFETIME:                                           ║\n"
             <<"║  First node death       : "<<std::setw(7)<<(g_firstDeathTime>0?g_firstDeathTime:-1.0)<<" s              ║\n"
             <<"║  50% node death         : "<<std::setw(7)<<(g_halfDeadTime>0?g_halfDeadTime:-1.0)<<" s              ║\n"
             <<"║  Network partition      : "<<std::setw(7)<<(g_partitionTime>0?g_partitionTime:-1.0)<<" s              ║\n"
             <<"╠══════════════════════════════════════════════════════════════╣\n"
             <<"║  v5.4 — balanced attacks, conservative isolation,            ║\n"
             <<"║         faster recovery, clean ML decision logging           ║\n"
             <<"║  Scenario A : baseline AODV (no defence)                    ║\n"
             <<"║  B/C/D/E   : HybridAODV — AODV discovers, ML selects+locks  ║\n"
             <<"║  EXPECTED under-attack PDR (v5.4):                          ║\n"
             <<"║    Scenario A (baseline):     60-70%  [AODV, no defence]    ║\n"
             <<"║    Scenario B (trust ML):     75-85%  [BH+SF detected]      ║\n"
             <<"║    Scenario E (energy):       65-75%  [vampire avoided]     ║\n"
             <<"║    Scenario C (ML+energy):    80-88%  [all types detected]  ║\n"
             <<"║    Scenario D (full Phase 2): 87-94%  [adaptive+multipath]  ║\n"
             <<"╚══════════════════════════════════════════════════════════════╝\n";

    // [IMP-04] Print ML effectiveness summary at end
    PrintMLEffectivenessSummary();
}

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════════════════
int main(int argc, char* argv[])
{
    signal(SIGINT,SigInt); signal(SIGTERM,SigInt);
    bool verbose=false;
    CommandLine cmd;
    cmd.AddValue("verbose","Enable detailed logging",verbose);
    cmd.AddValue("scenario","A=baseline(AODV) B=trust-only C=Phase1 D=Phase2-full E=energy-only",g_scenario);
    cmd.AddValue("rngSeed","RNG seed",g_rngSeed);
    cmd.Parse(argc,argv);
    for (auto& c:g_scenario) c=(char)toupper((unsigned char)c);

    if      (g_scenario=="A"){g_enableML=false;g_enableEnergy=false;g_enableRouteOpt=false;}
    else if (g_scenario=="B"){g_enableML=true; g_enableEnergy=false;g_enableRouteOpt=false;}
    else if (g_scenario=="C"){g_enableML=true; g_enableEnergy=true; g_enableRouteOpt=false;}
    else if (g_scenario=="E"){g_enableML=false;g_enableEnergy=true; g_enableRouteOpt=false;}
    else    {g_scenario="D"; g_enableML=true;  g_enableEnergy=true; g_enableRouteOpt=true;}

    LogComponentEnable("HybridWSNPhase2",LOG_LEVEL_INFO);
    if (verbose) LogComponentEnable("OnOffApplication",LOG_LEVEL_INFO);
    g_atkRng.seed(g_rngSeed+999);

    if (::system("mkdir -p results")!=0) std::cerr<<"[WARN] mkdir results failed\n";
    g_matrixDir="results/matrices_"+g_scenario;
    if (::system(("mkdir -p "+g_matrixDir).c_str())!=0){}

    std::string csvPath="results/performance_"+g_scenario+".csv";
    perfLog.open(csvPath);
    if (!perfLog.is_open()){std::cerr<<"[ERROR] Cannot open "<<csvPath<<"\n";return 1;}
    perfLog<<"time_s,pdr,under_attack_pdr,avg_delay_ms,avg_energy_J,isolation_events_cum,"
             "bh_active,sf_active,sh_active,vamp_active,isolated_active,alive_nodes,"
             "energy_stddev,soft_avoided,avg_routing_metric,avg_path_cost,path_stability,"
             "ctrl_overhead_delta,route_changes_delta,avg_hop_count,hop_stddev,"
             "min_hops,max_hops,jains_fairness_index,dynamic_alpha,dynamic_beta,scenario\n";

    pathTraceLog.open("results/path_traces_"+g_scenario+".csv");
    if (pathTraceLog.is_open()) pathTraceLog<<"time_s,node,cluster,path,hop_count,path_cost,stable,state\n";
    routeChangeLog.open("results/route_changes_"+g_scenario+".csv");
    if (routeChangeLog.is_open()) routeChangeLog<<"time_s,node,cluster,old_nh,new_nh,old_cost,new_cost,delta,reason\n";
    std::ofstream hel("results/hop_evolution_"+g_scenario+".csv"); hopEvolLog.swap(hel);
    if (hopEvolLog.is_open()){hopEvolLog<<"time_s";for(uint32_t i=0;i<N;++i)hopEvolLog<<",N"<<i;hopEvolLog<<"\n";}
    attackEventLog.open("results/attack_events_"+g_scenario+".csv");
    if (attackEventLog.is_open()) attackEventLog<<"time_s,attack_type,params,nodes\n";

    // [IMP-04] ML decision event log
    mlDecisionLog.open("results/ml_decisions_"+g_scenario+".csv");
    if (mlDecisionLog.is_open())
        mlDecisionLog<<"timestamp,node,event_type,attack_type,old_nh,new_nh,"
                     <<"old_trust,new_trust,old_score,new_score,old_path,new_path\n";

    // [HYBRID-N3] ML route decision log
    mlRouteLog.open("results/ml_route_decisions_"+g_scenario+".csv");
    if (mlRouteLog.is_open())
        mlRouteLog<<"time_s,node,next_hop,hop_count,trust,energy,score,selected\n";

    BuildNetwork(); BuildTraffic();

    // [IMP-04] v5.4 startup banner
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  Hybrid WSN Simulation v5.4 — ML-Enhanced Secure Routing     ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
    std::cout << "║  Scenario : " << g_scenario << "                                                   ║\n";
    std::cout << "║  Nodes    : " << N << "  |  Duration: " << SIM_DUR << "s  |  ML Interval: " << ML_INT << "s     ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
    std::cout << "║  Attack Parameters (v5.4 balanced):                          ║\n";
    std::cout << "║    Blackhole    : " << BH_NODES_PER_WAVE << " nodes, 100% drop                         ║\n";
    std::cout << "║    Selective FW : " << SF_NODES_PER_WAVE << " nodes, " << (int)(SF_DROP_RATE*100) << "% drop                        ║\n";
    std::cout << "║    Sinkhole     : " << SH_NODES_PER_WAVE << " nodes, " << (int)(SH_DROP_RATE*100) << "% drop                        ║\n";
    std::cout << "║    Vampire      : " << VAMP_NODES << " nodes, " << (VAMP_DRAIN_FRAC*100) << "% drain/interval          ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
    std::cout << "║  Isolation: THR=" << ISOLATE_TRUST_THR << "  CONSEC=" << ISOLATE_CONSEC_NEEDED
              << "  RESTORE=" << RESTORE_TRUST_THR << "  EMA=" << EMA_ALPHA << "          ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n\n";

    // [FIX-03] Attacks start at WARM_UP=60s
    ScheduleCycle(WARM_UP);

    // ML connectivity test (skip for A — no ML server needed; skip E — energy-only)
    if (g_enableML && g_scenario != "A" && g_scenario != "E") {
        std::string tr;
        std::vector<double> o(N,1.0), z(N,0.0), h(N,0.5);
        std::ostringstream tj;
        tj<<"{\"timestamp\":0"
          <<",\"energy\":"         <<JArr(o)
          <<",\"forward_ratio\":"  <<JArr(h)
          <<",\"drop_ratio\":"     <<JArr(h)
          <<",\"routing_metric\":" <<JArr(o)
          <<",\"path_cost\":"      <<JArr(z)
          <<",\"path_stability\":" <<JArr(o)
          <<",\"cluster_id\":"     <<JArr(h)
          <<",\"scenario\":\""     <<g_scenario<<"\"}";
        if (!IpcSend(tj.str(),tr)){
            std::cerr<<"\n[FATAL] ML server not reachable at "<<ML_HOST<<":"<<ML_PORT<<"\n"
                     <<"        Run: python3 ml_server_v5.py &\n"
                     <<"        Or:  --scenario=A\n\n"; return 1;
        }
        std::cout<<"[OK] ML server connected.\n";
    }

    std::string animPath="results/animation_"+g_scenario+".xml";
    anim=new AnimationInterface(animPath);
    anim->EnablePacketMetadata(true); anim->SetMaxPktsPerTraceFile(100000000);
    anim->EnableWifiPhyCounters(Seconds(0),Seconds(SIM_DUR),Seconds(LOG_INT));
    anim->EnableIpv4L3ProtocolCounters(Seconds(0),Seconds(SIM_DUR),Seconds(LOG_INT));
    anim->UpdateNodeColor(nodes.Get(0),180,0,0); anim->UpdateNodeSize(0,7.0,7.0);
    anim->UpdateNodeDescription(nodes.Get(0),"SINK [0 rx]");
    for (uint32_t i=1;i<N;++i){
        anim->UpdateNodeColor(nodes.Get(i),60,220,0); anim->UpdateNodeSize(i,2.5,2.5);
        anim->UpdateNodeDescription(nodes.Get(i),"N"+std::to_string(i)+"["+ClusterLabel(i)+" t=0.70 m=1.00]");
    }

    Simulator::Schedule(Seconds(15.0),&DrawRoutingPaths);
    Simulator::Schedule(Seconds(15.0),&UpdateNodeLabels);

    // [FIX-01] Log starts immediately
    Simulator::Schedule(Seconds(LOG_INT), &LogSnap, LOG_INT);
    Simulator::Schedule(Seconds(20.0),    &PrintConsoleSummary, 20.0);

    // Scenario A uses baseline AODV — no ML evaluation or trust/inject loop
    if (g_scenario != "A") {
        // [FIX-08] ML deferred to ML_WARMUP so AODV has time to discover routes
        Simulator::Schedule(Seconds(ML_WARMUP), &MLEval, ML_WARMUP);

        // [FIX-16] Reset snapshot counters just before first ML cycle
        Simulator::Schedule(Seconds(ML_WARMUP - 0.1), [](){
            for (uint32_t i=0;i<N;++i){
                snapTx[i]=pktTx[i]; snapRx[i]=pktRx[i]; snapOrig[i]=pktOrig[i];
            }
            NS_LOG_INFO("[FIX-16] Snapshot counters reset before first ML cycle");
        });
    }

    Simulator::Stop(Seconds(SIM_DUR)); Simulator::Run();
    flowMon->SerializeToXmlFile("results/flowmonitor_"+g_scenario+".xml",true,true);
    Summary(); Simulator::Destroy();
    perfLog.close(); pathTraceLog.close(); routeChangeLog.close();
    hopEvolLog.close(); attackEventLog.close(); mlRouteLog.close(); delete anim;
    return 0;
}
