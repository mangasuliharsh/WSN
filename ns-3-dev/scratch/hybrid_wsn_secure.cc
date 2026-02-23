/* =============================================================================
 * Hybrid Real-Time Closed-Loop Secure WSN — Phase 2  [BUG-FIXED]
 * Based on: Priyadarshi (2024), Wireless Networks 30:2647-2673
 * =============================================================================
 * BUGS FIXED (C++ side):
 *
 *  [BF-3] ApplyTrust() isolation threshold 0.30 with zero confirmation:
 *         Lowered hard-isolate threshold to 0.20 and added a per-node
 *         consecutive-low-trust counter.  A node must stay below 0.20 for
 *         ISOLATE_CONSEC_NEEDED=2 consecutive ML rounds before being isolated.
 *         A single bad ML round cannot isolate a healthy node anymore.
 *
 *  [BF-4] BestNeighbour() had no progress-to-sink constraint.  The greedy
 *         walk was picking the "healthiest" metric neighbour regardless of
 *         direction, causing routing loops (N8↔N29 style) and hops=-1.
 *         Fixed: candidates must reduce Euclidean distance to sink by at
 *         least PROGRESS_MIN_FRAC (5%) compared to the current node.
 *         When no progress-making neighbour exists, the closest-to-sink
 *         neighbour is accepted as a fallback (prevents total route failure).
 *
 *  [BF-5] UpdatePathMetrics() was triggering re-evaluation when hops==-1
 *         (i.e., no valid path exists).  552 wasteful re-evals were fired and
 *         1104 control packets generated for broken paths that couldn't be
 *         optimised.  Fixed: re-eval only fires when hopCount[i] > 0.
 *
 *  [BF-10] ComputePathCost() / BuildPathString() / ComputeHopCount() all
 *          used independent BestNeighbour() calls without a shared visited
 *          set across the three functions, so the loop-detection in each
 *          function only prevented intra-function loops.  A 3-node cycle
 *          (A→B→C→A) could appear as a valid path of length 3+.
 *          Fixed: all three now use the same greedy walk helper WalkToSink().
 *
 *  [BF-11] ActivateWin() called SafeDown(n) before malicious.insert(n), so
 *          the AODV stack for that node was brought down while the node was
 *          still marked "clean" — causing one LOG_INT window where AODV
 *          continued to compute routes through a link-down attacker.
 *          Fixed: malicious.insert(n) now precedes SafeDown(n).
 *
 *  [BF-12] DeactivateWin() called BestNeighbour(n) and ComputePathCost(n)
 *          before malicious.erase(n), so the route-change event logged
 *          "new_nexthop=NONE, new_cost=MAX_HOPS" (still treated as malicious).
 *          Fixed: erase happens first.
 *
 *  [BF-13] LogSnap() hop-count stats used INT_MAX sentinel in minHops but
 *          wrote it directly to CSV when no valid hop count existed.
 *          Fixed: minHops is sanitised to 0 before any output path.
 *
 * =============================================================================
 * ROUTING METRIC DEFINITION (unchanged):
 *   nodeScore(i) = α·trust(i) + β·energy_fraction(i)
 *   pathCost(P)  = Σ (1 - nodeScore(i))  for i in P
 *   Lower pathCost → better path
 * =============================================================================
 */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/wifi-module.h"
#include "ns3/mobility-module.h"
#include "ns3/aodv-module.h"
#include "ns3/applications-module.h"
#include "ns3/energy-module.h"
#include "ns3/netanim-module.h"
#include "ns3/flow-monitor-module.h"

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

NS_LOG_COMPONENT_DEFINE("HybridWSNPhase2");

// ═══════════════════════════════════════════════════════════════════════════════
// CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════════
static const uint32_t N          = 50;
static const double   INIT_E     = 150.0;
static const uint32_t PKT_SIZE   = 128;
static const double   ML_INT     = 20.0;
static const double   LOG_INT    = 5.0;
static const double   CYCLE      = 120.0;
static const double   SIM_DUR    = 600.0;
static const int      ML_PORT    = 5555;
static const char*    ML_HOST    = "127.0.0.1";
static const uint32_t GRID_W     = 10;
static const double   DX         = 50.0, DY = 50.0;
static const double   RADIO_RANGE     = 120.0;
static const int      MAX_HOPS        = 15;
static const double   REVAL_COST_THR  = 0.50;
static const double   STABLE_COST_THR = 0.40;

// Phase 1 metric weights
static const double   ALPHA      = 0.7;
static const double   BETA       = 0.3;
static const double   ROUTE_THR  = 0.4;

// Phase 2 routing thresholds
static const double   REVAL_COOLDOWN = 30.0;

// ── [BF-3] Isolation confirmation threshold and consecutive rounds needed ─────
static const double   ISOLATE_TRUST_THR    = 0.20;   // was 0.30 (too aggressive)
static const int      ISOLATE_CONSEC_NEEDED = 2;      // rounds needed before isolation
static const double   RESTORE_TRUST_THR    = 0.30;   // unchanged

// ── [BF-4] Progress-to-sink fraction for BestNeighbour ───────────────────────
// A candidate next-hop must reduce distance to sink by at least this fraction
// of the current node's distance.  5% gives directional bias without being
// too strict to route around obstacles.
static const double   PROGRESS_MIN_FRAC    = 0.05;

// ═══════════════════════════════════════════════════════════════════════════════
// SCENARIO FLAGS
// ═══════════════════════════════════════════════════════════════════════════════
static bool        g_enableML        = true;
static bool        g_enableEnergy    = true;
static bool        g_enableRouteOpt  = true;
static std::string g_scenario        = "D";

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

// Packet counters
std::vector<uint64_t> pktTx(N, 0);
std::vector<uint64_t> pktRx(N, 0);
std::vector<uint64_t> snapTx(N, 0), snapRx(N, 0);

// Trust / routing metric / isolation
std::vector<double>  trust(N, 1.0);
std::vector<double>  routingMetric(N, 1.0);
std::vector<bool>    isolated(N, false);
std::set<uint32_t>   malicious;
uint32_t             isoEvents    = 0;

// Phase 1 soft avoidance
std::vector<uint64_t> softAvoidCount(N, 0);

// ── [BF-3] Per-node consecutive low-trust counter ─────────────────────────────
std::vector<int>     consecLowTrust(N, 0);

// ── Phase 2: Route optimisation state ────────────────────────────────────────
std::vector<double>  pathCost(N, 0.0);
std::vector<int>     nextHopToSink(N, -1);

std::map<uint32_t, double> revalTimestamp;

// Stability tracking
std::vector<uint64_t> stableIntervals(N, 0);
std::vector<uint64_t> totalIntervals(N, 0);

// Control overhead
std::atomic<uint64_t> ctrlPktCount{0};
std::vector<uint64_t> snapCtrl(N, 0);

// Route change events
std::atomic<uint64_t> routeChangeCount{0};
uint64_t snapRouteChange = 0;

std::ofstream perfLog;
volatile bool gStop = false;

// ── Enhanced routing evidence state ──────────────────────────────────────────
std::vector<int>    prevNextHop(N, -2);
std::vector<double> prevPathCost(N, 0.0);
std::vector<int>    hopCount(N, 0);

std::ofstream pathTraceLog;
std::ofstream routeChangeLog;
std::ofstream hopEvolLog;

// ═══════════════════════════════════════════════════════════════════════════════
// FORWARD DECLARATIONS
// ═══════════════════════════════════════════════════════════════════════════════
void BuildNetwork();
void BuildTraffic();
void ScheduleCycle(double base);
void ActivateWin(int id, std::vector<uint32_t> tgts);
void DeactivateWin(std::vector<uint32_t> tgts);
void MLEval(double ts);
void LogSnap(double ts);
bool IpcSend(const std::string& j, std::string& r);
void ParseTrust(const std::string& j);
void ApplyTrust();
void RefreshAnim();
void SafeDown(uint32_t n);
void SafeUp(uint32_t n);
void PaintNode(uint32_t i);
std::vector<uint32_t> RandPair();
void DrawRoutingPaths();
void UpdateNodeLabels();

void   UpdatePathMetrics();
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

// ── [BF-10] Unified greedy walk result ───────────────────────────────────────
struct WalkResult {
    std::vector<uint32_t> path;   // ordered node IDs sink-inclusive
    bool   reachedSink;
    bool   hadLoop;
};
WalkResult WalkToSink(uint32_t srcNode);

// ═══════════════════════════════════════════════════════════════════════════════
// JSON HELPERS
// ═══════════════════════════════════════════════════════════════════════════════
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
// NETANIM COLOUR SCHEME
// ═══════════════════════════════════════════════════════════════════════════════
void PaintNode(uint32_t i)
{
    if (!anim) return;
    if (i == 0) { anim->UpdateNodeColor(nodes.Get(0), 200, 0, 0); return; }

    double rem = eSrc[i] ? eSrc[i]->GetRemainingEnergy() : 0.0;
    if (rem < 0.05) { anim->UpdateNodeColor(nodes.Get(i), 100, 100, 100); return; }
    if (malicious.count(i))  { anim->UpdateNodeColor(nodes.Get(i), 255, 120,   0); return; }
    if (isolated[i])          { anim->UpdateNodeColor(nodes.Get(i),  20,  20,  20); return; }

    double frac = rem / INIT_E;
    if (frac < 0.20) { anim->UpdateNodeColor(nodes.Get(i), 255, 230,   0); return; }

    double now = Simulator::Now().GetSeconds();
    auto it = revalTimestamp.find(i);
    if (g_enableRouteOpt && it != revalTimestamp.end() &&
        (now - it->second) < LOG_INT * 2)
    { anim->UpdateNodeColor(nodes.Get(i),  50, 100, 255); return; }

    if (g_enableEnergy && routingMetric[i] < ROUTE_THR && routingMetric[i] >= 0.2)
    { anim->UpdateNodeColor(nodes.Get(i),   0, 200, 220); return; }

    if (trust[i] < 0.5) { anim->UpdateNodeColor(nodes.Get(i), 140,   0, 140); return; }
    anim->UpdateNodeColor(nodes.Get(i), 0, 190, 60);
}

void RefreshAnim()
{
    for (uint32_t i = 0; i < N; ++i) PaintNode(i);
    UpdateNodeLabels();
}

// ═══════════════════════════════════════════════════════════════════════════════
// PACKET FLOW TAG
// ═══════════════════════════════════════════════════════════════════════════════
class PktFlowTag : public Tag
{
public:
    static TypeId GetTypeId() {
        static TypeId tid = TypeId("PktFlowTag")
            .SetParent<Tag>().AddConstructor<PktFlowTag>();
        return tid;
    }
    TypeId GetInstanceTypeId() const override { return GetTypeId(); }
    uint32_t GetSerializedSize()       const override { return 2; }
    void Serialize(TagBuffer b)        const override { b.WriteU8(m_type); b.WriteU8(m_src); }
    void Deserialize(TagBuffer b)            override { m_type=b.ReadU8(); m_src=b.ReadU8(); }
    void Print(std::ostream& os)       const override {
        const char* names[] = {"NORMAL","SOFT-AVOID","REVAL","","ATTACK"};
        os << "Flow=" << names[m_type==99?4:m_type] << " Src=N" << (int)m_src;
    }
    uint8_t m_type = 0;
    uint8_t m_src  = 0;
};

static void OnOffTxCb(uint32_t nodeId, Ptr<const Packet>)
{
    if (nodeId >= N) return;
    pktTx[nodeId]++;
}

static void SinkRxCb(Ptr<const Packet>, const Address&)
{
    pktRx[0]++;
}

// ═══════════════════════════════════════════════════════════════════════════════
// LIVE NODE LABELS
// ═══════════════════════════════════════════════════════════════════════════════
void UpdateNodeLabels()
{
    if (!anim) return;
    anim->UpdateNodeDescription(nodes.Get(0),
        "SINK [" + std::to_string(pktRx[0]) + " rx]");

    for (uint32_t i = 1; i < N; ++i) {
        double rem = eSrc[i] ? eSrc[i]->GetRemainingEnergy() : 0.0;

        if (malicious.count(i)) {
            anim->UpdateNodeDescription(nodes.Get(i),
                "N" + std::to_string(i) + " [ATTACK]");
        } else if (isolated[i]) {
            anim->UpdateNodeDescription(nodes.Get(i),
                "N" + std::to_string(i) + " [ISOL t=" +
                [&]{ std::ostringstream s; s<<std::fixed<<std::setprecision(2)<<trust[i]; return s.str(); }() + "]");
        } else if (rem < 0.05) {
            anim->UpdateNodeDescription(nodes.Get(i),
                "N" + std::to_string(i) + " [DEAD]");
        } else {
            std::ostringstream d;
            d << "N" << i << " [m=" << std::fixed << std::setprecision(2) << routingMetric[i];
            if (pathCost[i] > 0.001)
                d << " c=" << std::setprecision(1) << pathCost[i];
            if (hopCount[i] > 0)
                d << " h=" << hopCount[i];
            d << "]";
            anim->UpdateNodeDescription(nodes.Get(i), d.str());
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ROUTING PATH EDGE OVERLAY
// ═══════════════════════════════════════════════════════════════════════════════
void DrawRoutingPaths()
{
    if (!anim) return;
    double now = Simulator::Now().GetSeconds();

    for (uint32_t a = 0; a < N; ++a)
        for (uint32_t b = a + 1; b < N; ++b)
            anim->UpdateLinkDescription(a, b, "");

    for (uint32_t i = 1; i < N; ++i) {
        if (isolated[i] || malicious.count(i)) continue;
        double rem = eSrc[i] ? eSrc[i]->GetRemainingEnergy() : 0.0;
        if (rem < 0.05) continue;

        // [BF-10] Use unified walk
        WalkResult wr = WalkToSink(i);
        if (wr.path.size() < 2) continue;

        for (size_t k = 0; k + 1 < wr.path.size(); ++k) {
            uint32_t cur = wr.path[k];
            uint32_t nxt = wr.path[k + 1];
            double m = routingMetric[cur];
            std::ostringstream lbl;
            if      (m >= 0.70) lbl << "";
            else if (m >= 0.40) lbl << "~";
            else                lbl << "!";
            lbl << std::fixed << std::setprecision(2) << m;
            anim->UpdateLinkDescription(
                std::min(cur, nxt), std::max(cur, nxt), lbl.str());
        }
    }

    if (now + LOG_INT < SIM_DUR - 1.0)
        Simulator::Schedule(Seconds(LOG_INT), &DrawRoutingPaths);
}

// ═══════════════════════════════════════════════════════════════════════════════
// [BF-4] BEST NEIGHBOUR — with progress-to-sink constraint
// ═══════════════════════════════════════════════════════════════════════════════
int BestNeighbour(uint32_t srcNode)
{
    if (srcNode == 0) return -1;
    if (isolated[srcNode] || malicious.count(srcNode)) return -1;

    auto mobSrc  = nodes.Get(srcNode)->GetObject<MobilityModel>();
    auto mobSink = nodes.Get(0)->GetObject<MobilityModel>();
    if (!mobSrc || !mobSink) return -1;

    double distSrc = mobSrc->GetDistanceFrom(mobSink);

    // Minimum distance reduction required (5% of current distance to sink)
    double minReduction = distSrc * PROGRESS_MIN_FRAC;

    int    bestIdx      = -1;
    double bestMetric   = -1.0;

    // Fallback: closest-to-sink neighbour regardless of metric, used only when
    // no progress-making neighbour is found
    int    fallbackIdx     = -1;
    double fallbackMinDist = distSrc;   // must at least be closer than src

    for (uint32_t j = 0; j < N; ++j) {
        if (j == srcNode) continue;
        if (isolated[j] || malicious.count(j)) continue;
        double rem = eSrc[j] ? eSrc[j]->GetRemainingEnergy() : 0.0;
        if (rem < 0.05) continue;

        auto mobJ = nodes.Get(j)->GetObject<MobilityModel>();
        if (!mobJ) continue;
        if (mobSrc->GetDistanceFrom(mobJ) > RADIO_RANGE) continue;

        double distJ = mobJ->GetDistanceFrom(mobSink);

        // Progress constraint: candidate must get meaningfully closer to sink
        if ((distSrc - distJ) >= minReduction) {
            double m = routingMetric[j];
            if (m > bestMetric) {
                bestMetric = m;
                bestIdx    = (int)j;
            }
        }

        // Track closest neighbour for fallback (no metric bias)
        if (distJ < fallbackMinDist) {
            fallbackMinDist = distJ;
            fallbackIdx     = (int)j;
        }
    }

    if (bestIdx >= 0) return bestIdx;

    // Fallback: if no neighbour makes forward progress, accept the one that
    // gets us closest to the sink (prevents total route black-hole)
    if (fallbackIdx >= 0) {
        NS_LOG_DEBUG("BestNeighbour N" << srcNode
            << ": no progress neighbour, fallback to N" << fallbackIdx);
        return fallbackIdx;
    }

    return -1;
}

int NeighbourCount(uint32_t srcNode)
{
    if (isolated[srcNode] || malicious.count(srcNode)) return 0;
    auto mobSrc = nodes.Get(srcNode)->GetObject<MobilityModel>();
    if (!mobSrc) return 0;
    int cnt = 0;
    for (uint32_t j = 0; j < N; ++j) {
        if (j == srcNode) continue;
        if (isolated[j] || malicious.count(j)) continue;
        auto mobJ = nodes.Get(j)->GetObject<MobilityModel>();
        if (!mobJ) continue;
        if (mobSrc->GetDistanceFrom(mobJ) <= RADIO_RANGE) cnt++;
    }
    return cnt;
}

// ═══════════════════════════════════════════════════════════════════════════════
// [BF-10] UNIFIED GREEDY WALK — single source of truth for all path queries
// ═══════════════════════════════════════════════════════════════════════════════
WalkResult WalkToSink(uint32_t srcNode)
{
    WalkResult result;
    result.reachedSink = false;
    result.hadLoop     = false;

    if (srcNode == 0) {
        result.path = {0};
        result.reachedSink = true;
        return result;
    }

    if (isolated[srcNode] || malicious.count(srcNode)) {
        result.path = {srcNode};
        return result;
    }

    result.path.push_back(srcNode);
    std::set<uint32_t> visited;
    uint32_t cur = srcNode;

    for (int hop = 0; hop < MAX_HOPS; ++hop) {
        if (visited.count(cur)) {
            result.hadLoop = true;
            break;
        }
        visited.insert(cur);

        // Check if directly within range of sink
        auto mobCur  = nodes.Get(cur)->GetObject<MobilityModel>();
        auto mobSink = nodes.Get(0)->GetObject<MobilityModel>();
        if (mobCur && mobSink &&
            mobCur->GetDistanceFrom(mobSink) <= RADIO_RANGE) {
            result.path.push_back(0);
            result.reachedSink = true;
            break;
        }

        int nh = BestNeighbour(cur);
        if (nh < 0) break;   // dead end — path terminates here
        cur = (uint32_t)nh;
        result.path.push_back(cur);

        if (cur == 0) {
            result.reachedSink = true;
            break;
        }
    }

    return result;
}

// ═══════════════════════════════════════════════════════════════════════════════
// PATH METRIC UTILITIES — all delegate to WalkToSink
// ═══════════════════════════════════════════════════════════════════════════════
double ComputePathCost(uint32_t srcNode)
{
    if (srcNode == 0) return 0.0;
    if (isolated[srcNode] || malicious.count(srcNode)) return (double)MAX_HOPS;

    WalkResult wr = WalkToSink(srcNode);
    double cost = 0.0;
    for (uint32_t node : wr.path) {
        cost += (1.0 - routingMetric[node]);
    }
    if (wr.hadLoop) cost += 2.0;
    if (!wr.reachedSink) cost += (double)(MAX_HOPS - (int)wr.path.size());
    return cost;
}

std::string BuildPathString(uint32_t srcNode)
{
    if (srcNode == 0) return "0";
    if (isolated[srcNode] || malicious.count(srcNode))
        return std::to_string(srcNode) + "->BLOCKED";

    WalkResult wr = WalkToSink(srcNode);
    std::ostringstream ss;
    for (size_t i = 0; i < wr.path.size(); ++i) {
        if (i > 0) ss << "->";
        ss << wr.path[i];
    }
    if (wr.hadLoop)     ss << "->LOOP";
    if (!wr.reachedSink && !wr.hadLoop) ss << "->DEAD";
    return ss.str();
}

int ComputeHopCount(uint32_t srcNode)
{
    if (srcNode == 0) return 0;
    if (isolated[srcNode] || malicious.count(srcNode)) return -1;

    WalkResult wr = WalkToSink(srcNode);
    if (wr.hadLoop)     return -1;
    if (!wr.reachedSink) return -1;
    // path includes src; hop count = path length - 1
    return std::max(0, (int)wr.path.size() - 1);
}

// ═══════════════════════════════════════════════════════════════════════════════
// ENHANCED ROUTING EVIDENCE
// ═══════════════════════════════════════════════════════════════════════════════
void WriteRoutingMatrix(double ts)
{
    std::string fname = "results/routing_matrix_" +
                        std::to_string((int)ts) + ".csv";
    std::ofstream mat(fname);
    if (!mat.is_open()) {
        NS_LOG_WARN("Cannot write routing matrix: " << fname);
        return;
    }
    mat << "node";
    for (uint32_t j = 0; j < N; ++j) mat << "," << j;
    mat << "\n";

    for (uint32_t i = 0; i < N; ++i) {
        mat << i;
        for (uint32_t j = 0; j < N; ++j) {
            if (i == 0 || isolated[i] || malicious.count(i)) {
                mat << ",0";
            } else {
                mat << "," << ((nextHopToSink[i] == (int)j) ? "1" : "0");
            }
        }
        mat << "\n";
    }
    mat.close();
    NS_LOG_INFO("  [MATRIX] Written: " << fname);
}

void WritePathTraces(double ts)
{
    if (!pathTraceLog.is_open()) return;

    for (uint32_t i = 1; i < N; ++i) {
        double rem = eSrc[i] ? eSrc[i]->GetRemainingEnergy() : 0.0;

        std::string state = "normal";
        if (isolated[i])           state = "isolated";
        else if (malicious.count(i)) state = "malicious";
        else if (rem < 0.05)       state = "dead";
        else if (routingMetric[i] < ROUTE_THR) state = "soft-avoid";

        std::string pathStr  = (state == "normal" || state == "soft-avoid")
                                ? BuildPathString(i) : state;
        int    hops          = (state == "normal" || state == "soft-avoid")
                                ? hopCount[i] : -1;
        double pc            = pathCost[i];
        int    stableFlag    = (pc <= STABLE_COST_THR) ? 1 : 0;

        pathTraceLog << std::fixed << std::setprecision(1) << ts << ","
                     << i << ","
                     << "\"" << pathStr << "\","
                     << hops << ","
                     << std::setprecision(4) << pc << ","
                     << stableFlag << ","
                     << state << "\n";
    }
    pathTraceLog.flush();
    NS_LOG_INFO("  [TRACES] Path traces written for t=" << ts);
}

void LogRouteChangeEvent(double ts, uint32_t node, int oldNH, int newNH,
                          double oldCost, double newCost, const std::string& reason)
{
    if (!routeChangeLog.is_open()) return;

    std::string oldLabel = (oldNH >= 0) ? "N" + std::to_string(oldNH) : "NONE";
    std::string newLabel = (newNH >= 0) ? "N" + std::to_string(newNH) : "NONE";

    routeChangeLog << std::fixed << std::setprecision(1) << ts << ","
                   << node << ","
                   << oldLabel << ","
                   << newLabel << ","
                   << std::setprecision(4) << oldCost << ","
                   << newCost << ","
                   << std::setprecision(4) << (newCost - oldCost) << ","
                   << reason << "\n";
    routeChangeLog.flush();

    NS_LOG_INFO("  [ROUTE-CHG] t=" << ts << " N" << node
        << " " << oldLabel << "->" << newLabel
        << " cost: " << std::setprecision(3) << oldCost
        << "->" << newCost
        << " (" << reason << ")");
}

void WriteHopEvolution(double ts)
{
    if (!hopEvolLog.is_open()) return;

    hopEvolLog << std::fixed << std::setprecision(1) << ts;
    for (uint32_t i = 0; i < N; ++i) {
        hopEvolLog << "," << hopCount[i];
    }
    hopEvolLog << "\n";
    hopEvolLog.flush();
}

// ═══════════════════════════════════════════════════════════════════════════════
// PHASE 2: UPDATE PATH METRICS
// [BF-5] Re-eval only fires when hopCount[i] > 0 (valid path exists)
// ═══════════════════════════════════════════════════════════════════════════════
void UpdatePathMetrics()
{
    uint32_t nReeval  = 0;
    uint32_t nStable  = 0;
    uint32_t nActive  = 0;
    double   sumCost  = 0.0;
    double   now      = Simulator::Now().GetSeconds();

    for (uint32_t i = 1; i < N; ++i) {
        double rem = eSrc[i] ? eSrc[i]->GetRemainingEnergy() : 0.0;
        if (rem < 0.05 || isolated[i] || malicious.count(i)) {
            if (prevNextHop[i] >= 0) {
                std::string reason = isolated[i]          ? "isolated-loss" :
                                     malicious.count(i)   ? "malicious-loss" : "dead";
                if (!isolated[i]) {
                    LogRouteChangeEvent(now, i, prevNextHop[i], -1,
                                        prevPathCost[i], (double)MAX_HOPS, reason);
                }
                prevNextHop[i]  = -1;
                prevPathCost[i] = (double)MAX_HOPS;
            }
            nextHopToSink[i] = -1;
            hopCount[i]      = -1;
            continue;
        }

        int    nh = BestNeighbour(i);
        double pc = ComputePathCost(i);
        int    hc = ComputeHopCount(i);

        // Route change detection (same logic as before)
        bool firstTime   = (prevNextHop[i] == -2);
        bool nhChanged   = (!firstTime && nh != prevNextHop[i]);
        double costDelta = pc - prevPathCost[i];
        double costRatio = (prevPathCost[i] > 0.01)
                           ? pc / prevPathCost[i] : (pc > 0.01 ? 99.0 : 1.0);
        bool costChanged = (std::abs(costDelta) > 0.10);

        if (firstTime) {
            LogRouteChangeEvent(now, i, -1, nh, 0.0, pc, "route-established");
        } else if (nhChanged) {
            std::string reason;
            if      (costRatio < 0.85) reason = "improve";
            else if (costRatio > 1.15) reason = "degrade";
            else                       reason = "nh-swap";
            LogRouteChangeEvent(now, i, prevNextHop[i], nh,
                                prevPathCost[i], pc, reason);
        } else if (costChanged && !firstTime) {
            std::string reason;
            bool crossedRevalUp    = (pc > REVAL_COST_THR && prevPathCost[i] <= REVAL_COST_THR);
            bool crossedStableUp   = (pc <= STABLE_COST_THR && prevPathCost[i] > STABLE_COST_THR);
            if      (crossedRevalUp)   reason = "cost-rise";
            else if (crossedStableUp)  reason = "cost-fall";
            else if (costRatio > 1.15) reason = "cost-drift-up";
            else if (costRatio < 0.85) reason = "cost-drift-down";
            else                       reason = "cost-drift";
            LogRouteChangeEvent(now, i, nh, nh,
                                prevPathCost[i], pc, reason);
        }

        prevNextHop[i]  = nh;
        prevPathCost[i] = pc;

        nextHopToSink[i] = nh;
        pathCost[i]      = pc;
        hopCount[i]      = hc;
        sumCost         += pc;
        nActive++;

        totalIntervals[i]++;
        if (pc <= STABLE_COST_THR) {
            stableIntervals[i]++;
            nStable++;
        }

        // ── [BF-5] FIXED: Re-eval only when a valid path exists (hops > 0) ───
        // Previously fired even when hc == -1 (loop or no path), wasting 552
        // re-evaluations and 1104 control packets on paths that couldn't improve.
        if (g_enableRouteOpt && hc > 0) {
            bool cooldown = (revalTimestamp.count(i) &&
                             (now - revalTimestamp[i]) < REVAL_COOLDOWN);

            if (!cooldown && pc > REVAL_COST_THR) {
                revalTimestamp[i] = now;
                routeChangeCount++;
                nReeval++;
                ctrlPktCount += 2;

                NS_LOG_INFO("  ↻ REVAL N" << i
                    << "  bestNH=" << (nh >= 0 ? "N"+std::to_string(nh) : "none")
                    << "  pathCost=" << std::fixed << std::setprecision(3) << pc
                    << "  hops=" << hc
                    << "  neighbours=" << NeighbourCount(i));

                if (anim) {
                    anim->UpdateNodeColor(nodes.Get(i), 50, 100, 255);
                    anim->UpdateNodeDescription(nodes.Get(i),
                        "N" + std::to_string(i) + " ↻REVAL");
                    Simulator::Schedule(Seconds(LOG_INT * 2), [i]() {
                        if (!isolated[i] && !malicious.count(i)) PaintNode(i);
                        if (anim && !isolated[i] && !malicious.count(i))
                            anim->UpdateNodeDescription(nodes.Get(i), "N"+std::to_string(i));
                    });
                }
            }
        }
    }

    if (nActive > 0 || nReeval > 0)
        NS_LOG_INFO("  PathMetrics: active=" << nActive
            << "  avgCost=" << std::fixed << std::setprecision(3)
            << (nActive > 0 ? sumCost/nActive : 0.0)
            << "  stable=" << nStable << "/" << nActive
            << "  revals=" << nReeval);
}

// ═══════════════════════════════════════════════════════════════════════════════
// BUILD NETWORK
// ═══════════════════════════════════════════════════════════════════════════════
void BuildNetwork()
{
    NS_LOG_INFO("=== BuildNetwork (Phase 2 Enhanced + Bug-Fixed) ===");
    nodes.Create(N);

    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211b);
    wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager",
        "DataMode",    StringValue("DsssRate1Mbps"),
        "ControlMode", StringValue("DsssRate1Mbps"));

    YansWifiPhyHelper phy;
    phy.Set("TxPowerStart", DoubleValue(16.0));
    phy.Set("TxPowerEnd",   DoubleValue(16.0));

    YansWifiChannelHelper ch;
    ch.SetPropagationDelay("ns3::ConstantSpeedPropagationDelayModel");
    ch.AddPropagationLoss("ns3::LogDistancePropagationLossModel",
        "Exponent",          DoubleValue(2.8),
        "ReferenceLoss",     DoubleValue(46.7),
        "ReferenceDistance", DoubleValue(1.0));
    phy.SetChannel(ch.Create());

    WifiMacHelper mac;
    mac.SetType("ns3::AdhocWifiMac");
    devices = wifi.Install(phy, mac, nodes);

    MobilityHelper mob;
    mob.SetPositionAllocator("ns3::GridPositionAllocator",
        "MinX",      DoubleValue(20.0),
        "MinY",      DoubleValue(20.0),
        "DeltaX",    DoubleValue(DX),
        "DeltaY",    DoubleValue(DY),
        "GridWidth", UintegerValue(GRID_W),
        "LayoutType",StringValue("RowFirst"));
    mob.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mob.Install(nodes);

    AodvHelper aodv;
    aodv.Set("RreqRetries", UintegerValue(5));
    aodv.Set("ActiveRouteTimeout", TimeValue(Seconds(15.0)));

    InternetStackHelper inet;
    inet.SetRoutingHelper(aodv);
    inet.Install(nodes);

    Ipv4AddressHelper addr;
    addr.SetBase("10.1.0.0", "255.255.0.0");
    ifaces = addr.Assign(devices);

    BasicEnergySourceHelper esh;
    esh.Set("BasicEnergySourceInitialEnergyJ", DoubleValue(INIT_E));

    WifiRadioEnergyModelHelper reh;
    reh.Set("TxCurrentA",   DoubleValue(0.0174));
    reh.Set("RxCurrentA",   DoubleValue(0.0197));
    reh.Set("IdleCurrentA", DoubleValue(0.000426));

    for (uint32_t i = 0; i < N; ++i) {
        EnergySourceContainer esc = esh.Install(nodes.Get(i));
        eSrc[i] = DynamicCast<BasicEnergySource>(esc.Get(0));
        reh.Install(devices.Get(i), esc);
    }

    NS_LOG_INFO("Network ready: " << N << " nodes | AODV (Phase 2 BF) | 802.11b | E=" << INIT_E << "J");
}

// ═══════════════════════════════════════════════════════════════════════════════
// BUILD TRAFFIC
// ═══════════════════════════════════════════════════════════════════════════════
void BuildTraffic()
{
    NS_LOG_INFO("=== BuildTraffic ===");
    uint16_t port = 9;

    PacketSinkHelper sinkH("ns3::UdpSocketFactory",
        InetSocketAddress(Ipv4Address::GetAny(), port));
    ApplicationContainer sinkApps = sinkH.Install(nodes.Get(0));
    sinkApps.Start(Seconds(0.0));
    sinkApps.Stop(Seconds(SIM_DUR));

    for (uint32_t i = 1; i < N; ++i) {
        OnOffHelper src("ns3::UdpSocketFactory",
            InetSocketAddress(ifaces.GetAddress(0), port));
        src.SetConstantRate(DataRate("512bps"), PKT_SIZE);
        src.SetAttribute("OnTime",
            StringValue("ns3::ConstantRandomVariable[Constant=1]"));
        src.SetAttribute("OffTime",
            StringValue("ns3::ConstantRandomVariable[Constant=0]"));
        ApplicationContainer app = src.Install(nodes.Get(i));
        app.Start(Seconds(10.0 + i * 0.1));
        app.Stop(Seconds(SIM_DUR));
        app.Get(0)->TraceConnectWithoutContext("Tx",
            MakeBoundCallback(&OnOffTxCb, i));
    }

    Simulator::Schedule(Seconds(0.1), [](){
        Config::ConnectWithoutContext(
            "/NodeList/0/ApplicationList/0/$ns3::PacketSink/Rx",
            MakeCallback(&SinkRxCb));
        NS_LOG_INFO("SinkRxCb hooked via Config path");
    });

    flowMon = fmHelper.InstallAll();
    NS_LOG_INFO("Traffic ready: 49 CBR UDP → sink");
}

// ═══════════════════════════════════════════════════════════════════════════════
// ATTACK CYCLE
// [BF-11] malicious.insert() now precedes SafeDown()
// [BF-12] malicious.erase() now precedes BestNeighbour()/ComputePathCost()
// ═══════════════════════════════════════════════════════════════════════════════
std::vector<uint32_t> RandPair()
{
    static std::mt19937 rng(std::random_device{}());
    std::vector<uint32_t> pool;
    for (uint32_t i = 1; i < N; ++i)
        if (!isolated[i] && !malicious.count(i)) pool.push_back(i);
    if (pool.size() < 2) {
        pool.clear();
        for (uint32_t i = 1; i < N; ++i) pool.push_back(i);
    }
    std::shuffle(pool.begin(), pool.end(), rng);
    return {pool[0], pool[1]};
}

void ActivateWin(int id, std::vector<uint32_t> tgts)
{
    double t = Simulator::Now().GetSeconds();
    NS_LOG_INFO(">>> ATTACK WINDOW " << id << " at t=" << t
                << "s  [" << tgts[0] << ", " << tgts[1] << "]");
    for (uint32_t n : tgts) {
        // [BF-11] FIXED: mark malicious BEFORE SafeDown so AODV immediately
        // stops computing routes through this node
        malicious.insert(n);
        SafeDown(n);

        if (prevNextHop[n] >= 0) {
            LogRouteChangeEvent(t, n, prevNextHop[n], -1,
                                prevPathCost[n], (double)MAX_HOPS, "attack-activated");
        }
        prevNextHop[n]  = -1;
        prevPathCost[n] = (double)MAX_HOPS;

        if (anim) {
            anim->UpdateNodeColor(nodes.Get(n), 255, 120, 0);
            anim->UpdateNodeDescription(nodes.Get(n),
                "N" + std::to_string(n) + " [ATTACK]");
            anim->UpdateNodeSize(n, 4.0, 4.0);
        }
    }
    Simulator::Schedule(Seconds(0.5), &DrawRoutingPaths);
}

void DeactivateWin(std::vector<uint32_t> tgts)
{
    double t = Simulator::Now().GetSeconds();
    NS_LOG_INFO("<<< ATTACK WINDOW ENDED at t=" << t << "s");
    for (uint32_t n : tgts) {
        // [BF-12] FIXED: erase malicious status BEFORE calling BestNeighbour()
        // and ComputePathCost() so the new-route event uses valid (non-malicious)
        // routing state rather than logging NONE/MAX_HOPS for the restored node.
        malicious.erase(n);

        if (!isolated[n]) {
            SafeUp(n);

            int    newNH   = BestNeighbour(n);
            double newCost = ComputePathCost(n);
            LogRouteChangeEvent(t, n, -1, newNH,
                                (double)MAX_HOPS, newCost, "attack-ended");

            // Reset sentinel so UpdatePathMetrics logs a fresh route-established
            prevNextHop[n]  = -2;
            prevPathCost[n] = 0.0;

            if (anim) {
                anim->UpdateNodeDescription(nodes.Get(n),
                    "N" + std::to_string(n));
                anim->UpdateNodeSize(n, 2.0, 2.0);
                PaintNode(n);
            }
        }
    }
}

void ScheduleCycle(double base)
{
    if (base >= SIM_DUR) return;
    auto w1 = RandPair(), w2 = RandPair(), w3 = RandPair();
    NS_LOG_INFO("Cycle @ base=" << base
        << "  W1=[" << w1[0] << "," << w1[1] << "]"
        << "  W2=[" << w2[0] << "," << w2[1] << "]"
        << "  W3=[" << w3[0] << "," << w3[1] << "]");

    Simulator::Schedule(Seconds(base + 20.0),  &ActivateWin,   1, w1);
    Simulator::Schedule(Seconds(base + 40.0),  &DeactivateWin,    w1);
    Simulator::Schedule(Seconds(base + 60.0),  &ActivateWin,   2, w2);
    Simulator::Schedule(Seconds(base + 80.0),  &DeactivateWin,    w2);
    Simulator::Schedule(Seconds(base + 90.0),  &ActivateWin,   3, w3);
    Simulator::Schedule(Seconds(base + 110.0), &DeactivateWin,    w3);

    if (base + CYCLE < SIM_DUR)
        Simulator::Schedule(Seconds(base + CYCLE), &ScheduleCycle, base + CYCLE);
}

// ═══════════════════════════════════════════════════════════════════════════════
// ML IPC
// ═══════════════════════════════════════════════════════════════════════════════
bool IpcSend(const std::string& payload, std::string& resp)
{
    int s = ::socket(AF_INET, SOCK_STREAM, 0);
    if (s < 0) return false;
    struct timeval tv{5, 0};
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    struct sockaddr_in srv{};
    srv.sin_family = AF_INET;
    srv.sin_port   = htons(ML_PORT);
    inet_pton(AF_INET, ML_HOST, &srv.sin_addr);
    if (::connect(s, (struct sockaddr*)&srv, sizeof(srv)) < 0) {
        ::close(s);
        NS_LOG_WARN("ML connect failed: " << strerror(errno));
        return false;
    }
    std::string msg = payload + "\n";
    ::send(s, msg.c_str(), msg.size(), 0);
    char buf[524288] = {};
    ssize_t n = ::recv(s, buf, sizeof(buf)-1, 0);
    ::close(s);
    if (n <= 0) return false;
    resp = std::string(buf, n);
    return true;
}

void ParseTrust(const std::string& json)
{
    auto p = json.find("\"trust\"");
    if (p == std::string::npos) return;
    auto a = json.find('[', p), b = json.find(']', a);
    if (a == std::string::npos) return;
    auto t = JParseArr(json.substr(a, b - a + 1));
    for (size_t i = 0; i < t.size() && i < N; ++i)
        trust[i] = t[i];
}

// ═══════════════════════════════════════════════════════════════════════════════
// APPLY TRUST
// [BF-3] Threshold lowered to 0.20; 2 consecutive rounds required before
//        isolating.  Reset counter immediately when trust recovers.
// ═══════════════════════════════════════════════════════════════════════════════
void ApplyTrust()
{
    uint32_t nSoftAvoided = 0;
    double now = Simulator::Now().GetSeconds();

    for (uint32_t i = 1; i < N; ++i) {
        double rem        = eSrc[i] ? eSrc[i]->GetRemainingEnergy() : 0.0;
        double energyFrac = rem / INIT_E;
        if (rem < 0.05) continue;

        if (g_enableEnergy) {
            routingMetric[i] = ALPHA * trust[i] + BETA * energyFrac;
        } else {
            routingMetric[i] = trust[i];
        }

        if (g_enableEnergy && routingMetric[i] < ROUTE_THR && !isolated[i]) {
            softAvoidCount[i]++;
            nSoftAvoided++;
            NS_LOG_INFO("  ⚠ SOFT-AVOID N" << i
                << "  metric=" << std::fixed << std::setprecision(3)
                << routingMetric[i]);
        }

        if (g_enableML) {
            // ── [BF-3] FIXED: require 2 consecutive rounds below threshold ──
            if (trust[i] < ISOLATE_TRUST_THR && !isolated[i]) {
                consecLowTrust[i]++;
                NS_LOG_INFO("  ⚠ LOW-TRUST N" << i
                    << "  trust=" << std::fixed << std::setprecision(3) << trust[i]
                    << "  consec=" << consecLowTrust[i] << "/" << ISOLATE_CONSEC_NEEDED);

                if (consecLowTrust[i] >= ISOLATE_CONSEC_NEEDED) {
                    isolated[i] = true;
                    isoEvents++;
                    SafeDown(i);
                    consecLowTrust[i] = 0;  // reset after acting

                    LogRouteChangeEvent(now, i, prevNextHop[i], -1,
                                        prevPathCost[i], (double)MAX_HOPS, "isolate");

                    NS_LOG_INFO("  ✗ ISOLATE N" << i
                        << "  trust=" << std::fixed << std::setprecision(3) << trust[i]);
                    if (anim) {
                        anim->UpdateNodeColor(nodes.Get(i), 20, 20, 20);
                        anim->UpdateNodeDescription(nodes.Get(i),
                            "N" + std::to_string(i) + " ✗ISO");
                        anim->UpdateNodeSize(i, 1.5, 1.5);
                    }
                }
            } else if (trust[i] >= ISOLATE_TRUST_THR && !isolated[i]) {
                // Clear counter as soon as trust is above threshold
                consecLowTrust[i] = 0;
            }

            if (trust[i] >= RESTORE_TRUST_THR && isolated[i] && !malicious.count(i)) {
                isolated[i] = false;
                consecLowTrust[i] = 0;
                SafeUp(i);

                prevNextHop[i]  = -2;
                prevPathCost[i] = 0.0;

                NS_LOG_INFO("  ✓ RESTORE N" << i
                    << "  trust=" << std::fixed << std::setprecision(3) << trust[i]);
                if (anim) {
                    anim->UpdateNodeDescription(nodes.Get(i), "N" + std::to_string(i));
                    anim->UpdateNodeSize(i, 2.0, 2.0);
                    PaintNode(i);
                }
            }
        }
    }

    if (nSoftAvoided > 0)
        NS_LOG_INFO("  → " << nSoftAvoided << " nodes soft-avoided (metric < " << ROUTE_THR << ")");

    UpdatePathMetrics();
}

// ═══════════════════════════════════════════════════════════════════════════════
// PERIODIC LOG
// [BF-13] minHops INT_MAX sentinel sanitised before any output
// ═══════════════════════════════════════════════════════════════════════════════
void LogSnap(double ts)
{
    if (gStop) return;

    uint64_t totalTx = 0;
    for (uint32_t i = 1; i < N; ++i) totalTx += pktTx[i];
    uint64_t totalRx = pktRx[0];
    double pdr = (totalTx > 0) ? (double)totalRx / totalTx : 0.0;

    flowMon->CheckForLostPackets();
    auto stats = flowMon->GetFlowStats();
    double sumDelay = 0.0; uint32_t fc = 0;
    for (auto& kv : stats) {
        if (kv.second.rxPackets > 0) {
            sumDelay += kv.second.delaySum.GetSeconds() / kv.second.rxPackets;
            fc++;
        }
    }
    double avgDelay = fc > 0 ? sumDelay / fc * 1000.0 : 0.0;

    double totalE = 0.0; uint32_t alive = 0;
    for (uint32_t i = 0; i < N; ++i) {
        double e = eSrc[i] ? eSrc[i]->GetRemainingEnergy() : 0.0;
        totalE += e;
        if (e > 0.05) alive++;
    }
    double meanE = totalE / N;

    double variance = 0.0;
    for (uint32_t i = 0; i < N; ++i) {
        double e    = eSrc[i] ? eSrc[i]->GetRemainingEnergy() : 0.0;
        double diff = e - meanE;
        variance += diff * diff;
    }
    double energyStdDev = std::sqrt(variance / N);

    uint32_t nMal = (uint32_t)malicious.size();
    uint32_t nIso = 0, nSoftAvoid = 0;
    for (bool b : isolated) if (b) nIso++;
    for (uint32_t i = 1; i < N; ++i)
        if (g_enableEnergy && routingMetric[i] < ROUTE_THR && !isolated[i])
            nSoftAvoid++;

    double sumMetric = 0.0;
    for (uint32_t i = 1; i < N; ++i) sumMetric += routingMetric[i];
    double avgMetric = sumMetric / (N - 1);

    double sumPathCost = 0.0; uint32_t nWithRoute = 0;
    for (uint32_t i = 1; i < N; ++i) {
        if (nextHopToSink[i] >= 0) {
            sumPathCost += pathCost[i];
            nWithRoute++;
        }
    }
    double avgPathCost = nWithRoute > 0 ? sumPathCost / nWithRoute : 0.0;

    double sumStability = 0.0; uint32_t nTracked = 0;
    for (uint32_t i = 1; i < N; ++i) {
        if (totalIntervals[i] > 0) {
            sumStability += (double)stableIntervals[i] / totalIntervals[i];
            nTracked++;
        }
    }
    double avgStability = nTracked > 0 ? sumStability / nTracked : 0.0;

    uint64_t ctrlTotal   = ctrlPktCount.load();
    uint64_t ctrlDelta   = ctrlTotal - snapCtrl[0];
    snapCtrl[0]          = ctrlTotal;

    uint64_t rcTotal     = routeChangeCount.load();
    uint64_t rcDelta     = rcTotal - snapRouteChange;
    snapRouteChange      = rcTotal;

    // ── [BF-13] Hop count stats with sanitised minHops ───────────────────────
    double sumHops = 0.0; uint32_t nHopValid = 0;
    int    maxHops = 0;
    int    minHops = INT_MAX;   // sentinel initialised here

    for (uint32_t i = 1; i < N; ++i) {
        if (hopCount[i] > 0 && hopCount[i] < MAX_HOPS * 2) {
            sumHops += hopCount[i];
            nHopValid++;
            if (hopCount[i] > maxHops) maxHops = hopCount[i];
            if (hopCount[i] < minHops) minHops = hopCount[i];
        }
    }
    double avgHops   = nHopValid > 0 ? sumHops / nHopValid : 0.0;
    double hopVariance = 0.0;
    for (uint32_t i = 1; i < N; ++i) {
        if (hopCount[i] > 0 && hopCount[i] < MAX_HOPS * 2) {
            double d = hopCount[i] - avgHops;
            hopVariance += d * d;
        }
    }
    double hopStdDev = nHopValid > 0 ? std::sqrt(hopVariance / nHopValid) : 0.0;

    // [BF-13] FIXED: sanitise sentinel before writing to CSV or logging
    if (minHops == INT_MAX) minHops = 0;

    perfLog << std::fixed << std::setprecision(3)
            << ts              << ","
            << pdr             << ","
            << avgDelay        << ","
            << meanE           << ","
            << isoEvents       << ","
            << nMal            << ","
            << nIso            << ","
            << alive           << ","
            << energyStdDev    << ","
            << nSoftAvoid      << ","
            << avgMetric       << ","
            << avgPathCost     << ","
            << avgStability    << ","
            << ctrlDelta       << ","
            << rcDelta         << ","
            << avgHops         << ","
            << hopStdDev       << ","
            << minHops         << ","
            << maxHops         << ","
            << g_scenario      << "\n";
    perfLog.flush();

    WriteRoutingMatrix(ts);
    WriteHopEvolution(ts);

    NS_LOG_INFO("[LOG] t=" << ts
        << "s  PDR=" << std::setprecision(1) << (pdr*100.0) << "%"
        << "  TX=" << totalTx << "  RX=" << totalRx
        << "  Delay=" << std::setprecision(2) << avgDelay << "ms"
        << "  E=" << meanE << "J"
        << "  Alive=" << alive
        << "  Mal=" << nMal << "  Iso=" << nIso
        << "  PathCost=" << std::setprecision(3) << avgPathCost
        << "  Hops=" << std::setprecision(2) << avgHops << "±" << hopStdDev
        << "  [" << minHops << "-" << maxHops << "]"
        << "  Stability=" << std::setprecision(2) << (avgStability*100.0) << "%");

    Simulator::Schedule(Seconds(LOG_INT), &LogSnap, ts + LOG_INT);
}

// ═══════════════════════════════════════════════════════════════════════════════
// ML EVALUATION
// ═══════════════════════════════════════════════════════════════════════════════
void MLEval(double ts)
{
    if (gStop) return;
    NS_LOG_INFO("=== ML Evaluation @ t=" << ts << "s ===");

    std::vector<double> energyF(N), fwdF(N), dropF(N), isoF(N), malF(N);
    std::vector<double> pathCostF(N, 0.0);

    for (uint32_t i = 0; i < N; ++i) {
        double rem = eSrc[i] ? eSrc[i]->GetRemainingEnergy() : 0.0;
        energyF[i]   = rem / INIT_E;

        uint64_t dTx = pktTx[i] - snapTx[i];
        uint64_t dRx = pktRx[i] - snapRx[i];
        snapTx[i] = pktTx[i]; snapRx[i] = pktRx[i];

        if (i == 0) {
            uint64_t expected = (uint64_t)((N - 1) * (ML_INT / 0.5));
            fwdF[i] = std::min(1.0, (double)dRx / std::max((uint64_t)1, expected));
        } else if (dTx + dRx > 0) {
            fwdF[i] = (dTx > 0) ? 1.0 : 0.0;
        } else {
            fwdF[i] = 0.5;
        }

        dropF[i] = 1.0 - fwdF[i];
        isoF[i]  = isolated[i]        ? 1.0 : 0.0;
        malF[i]  = malicious.count(i) ? 1.0 : 0.0;
        pathCostF[i] = (i > 0) ? std::min(5.0, pathCost[i]) / 5.0 : 0.0;
    }

    std::vector<double> stabilityF(N, 1.0);
    for (uint32_t i = 1; i < N; ++i) {
        stabilityF[i] = (totalIntervals[i] > 0)
            ? (double)stableIntervals[i] / totalIntervals[i]
            : 1.0;
    }

    std::ostringstream json;
    json << "{\"timestamp\":"        << (int)ts
         << ",\"energy\":"           << JArr(energyF)
         << ",\"forward_ratio\":"    << JArr(fwdF)
         << ",\"drop_ratio\":"       << JArr(dropF)
         << ",\"isolated\":"         << JArr(isoF)
         << ",\"known_malicious\":"  << JArr(malF)
         << ",\"routing_metric\":"   << JArr(routingMetric)
         << ",\"path_cost\":"        << JArr(pathCostF)
         << ",\"path_stability\":"   << JArr(stabilityF)
         << ",\"scenario\":\""       << g_scenario << "\""
         << "}";

    std::string resp;
    if (g_enableML) {
        if (IpcSend(json.str(), resp)) {
            NS_LOG_INFO("  ML ← " << resp.substr(0, 120) << "...");
            ParseTrust(resp);
            ApplyTrust();
        } else {
            NS_LOG_WARN("  ML server unreachable — trust scores unchanged");
        }
    } else {
        for (uint32_t i = 0; i < N; ++i) trust[i] = 1.0;
        ApplyTrust();
    }

    WritePathTraces(ts);
    RefreshAnim();
    Simulator::Schedule(Seconds(ML_INT), &MLEval, ts + ML_INT);
}

// ═══════════════════════════════════════════════════════════════════════════════
// SIGNAL HANDLER
// ═══════════════════════════════════════════════════════════════════════════════
static void SigInt(int)
{
    gStop = true;
    std::cout << "\n[SIM] Ctrl+C caught — stopping cleanly...\n";
    Simulator::Stop();
}

// ═══════════════════════════════════════════════════════════════════════════════
// FINAL SUMMARY
// ═══════════════════════════════════════════════════════════════════════════════
static void Summary()
{
    uint64_t totalTx = 0;
    for (uint32_t i = 1; i < N; ++i) totalTx += pktTx[i];
    uint64_t totalRx = pktRx[0];

    flowMon->CheckForLostPackets();
    auto stats = flowMon->GetFlowStats();
    double sumD = 0.0; uint32_t fc = 0;
    for (auto& kv : stats)
        if (kv.second.rxPackets > 0) {
            sumD += kv.second.delaySum.GetSeconds() / kv.second.rxPackets;
            fc++;
        }

    double totalE = 0.0; uint32_t alive = 0;
    for (uint32_t i = 0; i < N; ++i) {
        double e = eSrc[i] ? eSrc[i]->GetRemainingEnergy() : 0.0;
        totalE += e; if (e > 0.05) alive++;
    }
    double meanE = totalE / N;

    double variance = 0.0;
    for (uint32_t i = 0; i < N; ++i) {
        double e = eSrc[i] ? eSrc[i]->GetRemainingEnergy() : 0.0;
        variance += (e - meanE) * (e - meanE);
    }
    double energyStdDev = std::sqrt(variance / N);

    double sumMetric = 0.0;
    for (uint32_t i = 1; i < N; ++i) sumMetric += routingMetric[i];
    double avgMetric = sumMetric / (N - 1);

    double sumPathCost = 0.0; uint32_t nWithRoute = 0;
    for (uint32_t i = 1; i < N; ++i) {
        if (nextHopToSink[i] >= 0) { sumPathCost += pathCost[i]; nWithRoute++; }
    }
    double avgPathCost = nWithRoute > 0 ? sumPathCost / nWithRoute : 0.0;

    double sumStab = 0.0; uint32_t nTracked = 0;
    for (uint32_t i = 1; i < N; ++i) {
        if (totalIntervals[i] > 0) {
            sumStab += (double)stableIntervals[i] / totalIntervals[i];
            nTracked++;
        }
    }
    double avgStability = nTracked > 0 ? sumStab / nTracked : 0.0;

    double sumHops = 0.0; uint32_t nHopValid = 0;
    for (uint32_t i = 1; i < N; ++i) {
        if (hopCount[i] > 0 && hopCount[i] < MAX_HOPS * 2) {
            sumHops += hopCount[i]; nHopValid++;
        }
    }
    double avgHops = nHopValid > 0 ? sumHops / nHopValid : 0.0;

    uint64_t totalSoftAvoid = 0;
    for (auto v : softAvoidCount) totalSoftAvoid += v;

    double pdr   = totalTx > 0 ? (double)totalRx / totalTx * 100.0 : 0.0;
    double delay = fc > 0 ? sumD / fc * 1000.0 : 0.0;
    uint64_t totalCtrl = ctrlPktCount.load();
    uint64_t totalRc   = routeChangeCount.load();

    std::cout << "\n╔══════════════════════════════════════════════════════════╗\n"
              << "║  FINAL RESULTS — Scenario " << g_scenario << " (Phase 2 BUG-FIXED)   ║\n"
              << "╠══════════════════════════════════════════════════════════╣\n"
              << std::fixed << std::setprecision(2)
              << "║  PDR                   : " << std::setw(7)  << pdr        << " %                   ║\n"
              << "║  Avg E2E Delay         : " << std::setw(7)  << delay      << " ms                  ║\n"
              << "║  Avg Energy Left       : " << std::setw(7)  << meanE      << " J                   ║\n"
              << "║  Energy Std Dev        : " << std::setw(7)  << energyStdDev << " J              [P1] ║\n"
              << "║  Avg Routing Metric    : " << std::setw(7)  << avgMetric  << "              [P1] ║\n"
              << "║  Avg Path Cost         : " << std::setw(7)  << avgPathCost << "         [NEW P2] ║\n"
              << "║  Path Stability        : " << std::setw(7)  << (avgStability*100.0) << " %        [NEW P2] ║\n"
              << "║  Avg Hop Count         : " << std::setw(7)  << avgHops    << "  hops     [NEW P2] ║\n"
              << "║  Total TX packets      : " << std::setw(9)  << totalTx    << "           ║\n"
              << "║  Total RX packets      : " << std::setw(9)  << totalRx    << "           ║\n"
              << "║  Alive nodes           : " << std::setw(9)  << alive      << "           ║\n"
              << "║  Isolation Events      : " << std::setw(9)  << isoEvents  << "           ║\n"
              << "║  Soft-Avoid Events     : " << std::setw(9)  << totalSoftAvoid << "       [P1] ║\n"
              << "║  Route Re-eval Events  : " << std::setw(9)  << totalRc    << "   [NEW P2] ║\n"
              << "║  Ctrl Overhead Pkts    : " << std::setw(9)  << totalCtrl  << "   [NEW P2] ║\n"
              << "╠══════════════════════════════════════════════════════════╣\n"
              << "║  OUTPUT FILES:                                            ║\n"
              << "║  results/performance_" << g_scenario << ".csv                          ║\n"
              << "║  results/routing_matrix_T.csv                             ║\n"
              << "║  results/path_traces.csv                                  ║\n"
              << "║  results/route_changes.csv                                ║\n"
              << "║  results/hop_evolution.csv                                ║\n"
              << "╚══════════════════════════════════════════════════════════╝\n";
}

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════════════════
int main(int argc, char* argv[])
{
    signal(SIGINT,  SigInt);
    signal(SIGTERM, SigInt);

    bool verbose = false;
    CommandLine cmd;
    cmd.AddValue("verbose",  "Enable detailed logging",                         verbose);
    cmd.AddValue("scenario", "Scenario: A=baseline, B=trust, C=P1, D=P2full",  g_scenario);
    cmd.Parse(argc, argv);

    for (auto& c : g_scenario) c = (char)toupper((unsigned char)c);

    if (g_scenario == "A") {
        g_enableML = false; g_enableEnergy = false; g_enableRouteOpt = false;
    } else if (g_scenario == "B") {
        g_enableML = true;  g_enableEnergy = false; g_enableRouteOpt = false;
    } else if (g_scenario == "C") {
        g_enableML = true;  g_enableEnergy = true;  g_enableRouteOpt = false;
    } else {
        g_scenario = "D";
        g_enableML = true;  g_enableEnergy = true;  g_enableRouteOpt = true;
    }

    LogComponentEnable("HybridWSNPhase2", LOG_LEVEL_INFO);
    if (verbose) LogComponentEnable("OnOffApplication", LOG_LEVEL_INFO);

    NS_LOG_INFO("=== Scenario " << g_scenario
        << " | ML=" << g_enableML
        << " | EnergyWeight=" << g_enableEnergy
        << " | RouteOpt=" << g_enableRouteOpt << " ===");

    if (::system("mkdir -p results") != 0)
        std::cerr << "[WARN] mkdir results failed\n";

    std::string csvPath = "results/performance_" + g_scenario + ".csv";
    perfLog.open(csvPath);
    if (!perfLog.is_open()) {
        std::cerr << "[ERROR] Cannot open " << csvPath << "\n";
        return 1;
    }
    perfLog << "time_s,pdr,avg_delay_ms,avg_energy_J,isolation_events_cum,"
               "malicious_active,isolated_active,alive_nodes,"
               "energy_stddev,soft_avoided,avg_routing_metric,"
               "avg_path_cost,path_stability,ctrl_overhead_delta,"
               "route_changes_delta,"
               "avg_hop_count,hop_stddev,min_hops,max_hops,"
               "scenario\n";

    pathTraceLog.open("results/path_traces.csv");
    if (!pathTraceLog.is_open()) {
        std::cerr << "[WARN] Cannot open results/path_traces.csv\n";
    } else {
        pathTraceLog << "time_s,node,path,hop_count,path_cost,stable,state\n";
    }

    routeChangeLog.open("results/route_changes.csv");
    if (!routeChangeLog.is_open()) {
        std::cerr << "[WARN] Cannot open results/route_changes.csv\n";
    } else {
        routeChangeLog << "time_s,node,old_nexthop,new_nexthop,"
                          "old_cost,new_cost,cost_delta,reason\n";
    }

    hopEvolLog.open("results/hop_evolution.csv");
    if (!hopEvolLog.is_open()) {
        std::cerr << "[WARN] Cannot open results/hop_evolution.csv\n";
    } else {
        hopEvolLog << "time_s";
        for (uint32_t i = 0; i < N; ++i) hopEvolLog << ",N" << i;
        hopEvolLog << "\n";
    }

    BuildNetwork();
    BuildTraffic();
    ScheduleCycle(0.0);

    Simulator::Schedule(Seconds(0.5), [](){
        uint32_t totalNeighbors = 0;
        for (uint32_t i = 0; i < N; ++i) {
            auto mob_i = nodes.Get(i)->GetObject<MobilityModel>();
            uint32_t cnt = 0;
            for (uint32_t j = 0; j < N; ++j) {
                if (i == j) continue;
                auto mob_j = nodes.Get(j)->GetObject<MobilityModel>();
                if (mob_i->GetDistanceFrom(mob_j) <= 120.0) cnt++;
            }
            totalNeighbors += cnt;
        }
        double avg = (double)totalNeighbors / N;
        std::cout << "[DIAG] Avg neighbors per node (within 120m): "
                  << std::fixed << std::setprecision(1) << avg << "\n";
        std::cout << (avg >= 2.0 ? "[OK]  Connectivity adequate.\n"
                                  : "[WARN] Low connectivity.\n");
    });

    std::string animPath = "results/animation_" + g_scenario + ".xml";
    anim = new AnimationInterface(animPath);
    anim->EnablePacketMetadata(true);
    anim->SetMaxPktsPerTraceFile(100000000);
    anim->EnableWifiPhyCounters(Seconds(0), Seconds(SIM_DUR), Seconds(LOG_INT));
    anim->EnableIpv4L3ProtocolCounters(Seconds(0), Seconds(SIM_DUR), Seconds(LOG_INT));

    anim->UpdateNodeColor(nodes.Get(0), 200, 0, 0);
    anim->UpdateNodeDescription(nodes.Get(0), "SINK [0 rx]");
    anim->UpdateNodeSize(0, 6.0, 6.0);

    for (uint32_t i = 1; i < N; ++i) {
        anim->UpdateNodeColor(nodes.Get(i), 0, 190, 60);
        anim->UpdateNodeDescription(nodes.Get(i),
            "N" + std::to_string(i) + " [m=1.00]");
        anim->UpdateNodeSize(i, 2.5, 2.5);
    }

    Simulator::Schedule(Seconds(15.0), &DrawRoutingPaths);
    Simulator::Schedule(Seconds(15.0), &UpdateNodeLabels);
    Simulator::Schedule(Seconds(LOG_INT), &LogSnap, LOG_INT);
    Simulator::Schedule(Seconds(ML_INT),  &MLEval,  ML_INT);

    std::cout
        << "\n╔═══════════════════════════════════════════════════════════════════╗\n"
        << "║  Hybrid Real-Time Secure WSN — Phase 2 ENHANCED + BUG-FIXED       ║\n"
        << "║  50 nodes | AODV | 802.11b | Energy=150J | Dur=600s               ║\n"
        << "╠═══════════════════════════════════════════════════════════════════╣\n"
        << "║  FIXES ACTIVE:                                                    ║\n"
        << "║  [BF-3]  Isolation threshold 0.30→0.20 + 2-round confirmation     ║\n"
        << "║  [BF-4]  BestNeighbour: progress-to-sink constraint added         ║\n"
        << "║  [BF-5]  Re-eval guard: skipped when hops==-1                     ║\n"
        << "║  [BF-10] Unified WalkToSink() for all path queries                ║\n"
        << "║  [BF-11] Attack: malicious.insert() precedes SafeDown()           ║\n"
        << "║  [BF-12] Deactivate: malicious.erase() precedes route query       ║\n"
        << "║  [BF-13] minHops INT_MAX sentinel sanitised in CSV output         ║\n"
        << "╚═══════════════════════════════════════════════════════════════════╝\n\n";

    Simulator::Stop(Seconds(SIM_DUR));
    Simulator::Run();

    std::string fmPath = "results/flowmonitor_" + g_scenario + ".xml";
    flowMon->SerializeToXmlFile(fmPath, true, true);
    Summary();
    Simulator::Destroy();
    perfLog.close();
    pathTraceLog.close();
    routeChangeLog.close();
    hopEvolLog.close();
    delete anim;
    return 0;
}
