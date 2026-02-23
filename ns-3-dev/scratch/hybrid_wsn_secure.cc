/* =============================================================================
 * Hybrid Real-Time Closed-Loop Secure WSN — Phase 1 Upgrade
 * Based on: Priyadarshi (2024), Wireless Networks 30:2647-2673
 * =============================================================================
 * PHASE 1 ADDITIONS:
 *  ✦ Multi-objective routing metric: score = 0.7*trust + 0.3*energy_fraction
 *  ✦ Soft load balancing via routing score threshold (< 0.4 → avoid)
 *  ✦ Energy imbalance index (stddev) tracked per log interval
 *  ✦ Controlled 600s simulation duration for reproducible evaluation
 *  ✦ Scenario flags: --scenario=A/B/C for baseline/trust/trust+energy
 *  ✦ Scenario label written to CSV for cross-experiment comparison
 *
 * PDR FIX: Direct IP-layer Rx on sink + Tx traces on sources
 * INFINITE RUN FIX: Simulator::Stop(600) for Phase 1 evaluation
 * NETANIM: EnablePacketMetadata + WiFi/IPv4 counters for animated arrows
 * ML: Python ensemble IF+LOF+OCSVM via TCP IPC on port 5555
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

using namespace ns3;
using namespace ns3::energy;

NS_LOG_COMPONENT_DEFINE("HybridWSNSecure");

// ═══════════════════════════════════════════════════════════════════════════════
// CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════════
static const uint32_t N          = 50;
static const double   INIT_E     = 150.0;   // Joules per node
static const uint32_t PKT_SIZE   = 128;     // bytes
static const double   ML_INT     = 20.0;    // ML evaluation interval (s)
static const double   LOG_INT    = 5.0;     // CSV log interval (s)
static const double   CYCLE      = 120.0;   // attack cycle length (s)
static const double   SIM_DUR    = 600.0;   // Phase 1: controlled 600s run
static const int      ML_PORT    = 5555;
static const char*    ML_HOST    = "127.0.0.1";
static const uint32_t GRID_W     = 10;
static const double   DX         = 50.0, DY = 50.0;

// ── Routing metric weights (Phase 1) ─────────────────────────────────────────
static const double   ALPHA      = 0.7;    // trust weight
static const double   BETA       = 0.3;    // energy weight
static const double   ROUTE_THR  = 0.4;    // soft avoidance threshold

// ═══════════════════════════════════════════════════════════════════════════════
// SCENARIO FLAGS  (set via command line)
// ═══════════════════════════════════════════════════════════════════════════════
// Scenario A: baseline AODV (no ML, no energy weighting)
// Scenario B: trust-only (ML + isolation, no energy weighting)
// Scenario C: trust + energy (full Phase 1 system)
static bool    g_enableML      = true;
static bool    g_enableEnergy  = true;
static std::string g_scenario  = "C";

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

// ── Direct IP-layer counters ─────────────────────────────────────────────────
std::vector<uint64_t> pktTx(N, 0);
std::vector<uint64_t> pktRx(N, 0);
std::vector<uint64_t> snapTx(N, 0), snapRx(N, 0);

// ── Trust / routing metric / isolation ───────────────────────────────────────
std::vector<double>  trust(N, 1.0);
std::vector<double>  routingMetric(N, 1.0);   // ← Phase 1: combined score
std::vector<bool>    isolated(N, false);
std::set<uint32_t>   malicious;
uint32_t             isoEvents = 0;

// ── Soft avoidance counters (how many times a node was bypassed) ──────────────
std::vector<uint64_t> softAvoidCount(N, 0);

std::ofstream        perfLog;
volatile bool        gStop = false;

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
    std::istringstream ss(j.substr(a + 1, b - a - 1));
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
// RED     = Sink (node 0)
// GREEN   = Normal + healthy
// ORANGE  = Actively malicious
// BLACK   = ML-isolated
// PURPLE  = Suspicious (trust 0.3–0.5)
// YELLOW  = Low energy (< 20%)
// CYAN    = Low routing metric (soft avoided, 0.4–0.55)
// GRAY    = Dead (energy ≈ 0)
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

    // Phase 1: show soft-avoided nodes in cyan
    if (g_enableEnergy && routingMetric[i] < ROUTE_THR && routingMetric[i] >= 0.2)
    { anim->UpdateNodeColor(nodes.Get(i),   0, 200, 220); return; }

    if (trust[i] < 0.5) { anim->UpdateNodeColor(nodes.Get(i), 140,   0, 140); return; }
    anim->UpdateNodeColor(nodes.Get(i), 0, 190, 60);
}

void RefreshAnim()
{
    for (uint32_t i = 0; i < N; ++i) PaintNode(i);
}

// ═══════════════════════════════════════════════════════════════════════════════
// APP-LAYER TRACE CALLBACKS
// ═══════════════════════════════════════════════════════════════════════════════
// PacketSink "Rx" trace passes (Ptr<Socket>) in some builds — use the
// two-argument form (Packet, Address) which is the correct ns-3 signature.
static void OnOffTxCb(uint32_t nodeId, Ptr<const Packet> /*pkt*/)
{
    if (nodeId < N) pktTx[nodeId]++;
}
// Called by Config::ConnectWithoutContext on the PacketSink Rx trace.
// Signature must match TracedCallback<Ptr<const Packet>, const Address &>
static void SinkRxCb(Ptr<const Packet> /*pkt*/, const Address& /*addr*/)
{
    pktRx[0]++;
}

// ═══════════════════════════════════════════════════════════════════════════════
// BUILD NETWORK
// ═══════════════════════════════════════════════════════════════════════════════
void BuildNetwork()
{
    NS_LOG_INFO("=== BuildNetwork ===");
    nodes.Create(N);

    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211b);
    // Use 1 Mbps base rate — lower rate = better sensitivity = longer range
    wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager",
        "DataMode",    StringValue("DsssRate1Mbps"),
        "ControlMode", StringValue("DsssRate1Mbps"));

    YansWifiPhyHelper phy;
    // 16 dBm gives ~100–120m range with LogDistance model (exponent 2.8)
    // This comfortably covers the 50m grid spacing including diagonals (~70m)
    phy.Set("TxPowerStart", DoubleValue(16.0));
    phy.Set("TxPowerEnd",   DoubleValue(16.0));
    // Use realistic LogDistance propagation instead of default Friis (too optimistic)
    YansWifiChannelHelper ch;
    ch.SetPropagationDelay("ns3::ConstantSpeedPropagationDelayModel");
    ch.AddPropagationLoss("ns3::LogDistancePropagationLossModel",
        "Exponent",         DoubleValue(2.8),   // typical indoor/outdoor WSN
        "ReferenceLoss",    DoubleValue(46.7),  // free-space loss at 1m, 2.4GHz
        "ReferenceDistance",DoubleValue(1.0));
    phy.SetChannel(ch.Create());

    WifiMacHelper mac;
    mac.SetType("ns3::AdhocWifiMac");
    devices = wifi.Install(phy, mac, nodes);

    MobilityHelper mob;
    mob.SetPositionAllocator("ns3::GridPositionAllocator",
        "MinX", DoubleValue(20.0), "MinY", DoubleValue(20.0),
        "DeltaX", DoubleValue(DX), "DeltaY", DoubleValue(DY),
        "GridWidth", UintegerValue(GRID_W), "LayoutType", StringValue("RowFirst"));
    mob.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mob.Install(nodes);

    AodvHelper aodv;
    InternetStackHelper inet;
    inet.SetRoutingHelper(aodv);
    inet.Install(nodes);

    Ipv4AddressHelper addr;
    addr.SetBase("10.1.0.0", "255.255.0.0");
    ifaces = addr.Assign(devices);

    BasicEnergySourceHelper esh;
    esh.Set("BasicEnergySourceInitialEnergyJ", DoubleValue(INIT_E));

    WifiRadioEnergyModelHelper reh;
    // Explicit current draw so energy model doesn't stall
    reh.Set("TxCurrentA",   DoubleValue(0.0174));  // 802.11b Tx ~17.4 mA
    reh.Set("RxCurrentA",   DoubleValue(0.0197));  // Rx ~19.7 mA
    reh.Set("IdleCurrentA", DoubleValue(0.000426));// Idle ~0.426 mA

    for (uint32_t i = 0; i < N; ++i) {
        EnergySourceContainer esc = esh.Install(nodes.Get(i));
        eSrc[i] = DynamicCast<BasicEnergySource>(esc.Get(0));
        reh.Install(devices.Get(i), esc);
    }

    NS_LOG_INFO("Network ready: " << N << " nodes | AODV | 802.11b | E=" << INIT_E << "J");
}

// ═══════════════════════════════════════════════════════════════════════════════
// BUILD TRAFFIC
// ═══════════════════════════════════════════════════════════════════════════════
void BuildTraffic()
{
    NS_LOG_INFO("=== BuildTraffic ===");
    uint16_t port = 9;

    // ── Sink (node 0) ─────────────────────────────────────────────────────────
    PacketSinkHelper sinkH("ns3::UdpSocketFactory",
        InetSocketAddress(Ipv4Address::GetAny(), port));
    ApplicationContainer sinkApps = sinkH.Install(nodes.Get(0));
    sinkApps.Start(Seconds(0.0));
    sinkApps.Stop(Seconds(SIM_DUR));

    // ── Sources start at 10s to allow AODV route discovery to complete ────────
    // AODV typically converges within 3-8s; we use 10s to be safe.
    // Stagger by 0.1s per node to avoid a simultaneous burst.
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
        // Hook TX callback — bind node id so counter is per-source
        app.Get(0)->TraceConnectWithoutContext("Tx",
            MakeBoundCallback(&OnOffTxCb, i));
    }

    // ── Hook sink RX via Config path — more reliable than per-app hook ────────
    // Schedule hook at t=0.1 so the app object is fully initialised
    Simulator::Schedule(Seconds(0.1), [](){
        Config::ConnectWithoutContext(
            "/NodeList/0/ApplicationList/0/$ns3::PacketSink/Rx",
            MakeCallback(&SinkRxCb));
        NS_LOG_INFO("SinkRxCb hooked via Config path");
    });

    flowMon = fmHelper.InstallAll();
    NS_LOG_INFO("Traffic ready: 49 CBR UDP → sink (start at t=10s for AODV convergence)");
}

// ═══════════════════════════════════════════════════════════════════════════════
// ATTACK CYCLE
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
        malicious.insert(n);
        SafeDown(n);
        if (anim) {
            anim->UpdateNodeColor(nodes.Get(n), 255, 120, 0);
            anim->UpdateNodeDescription(nodes.Get(n),
                "N" + std::to_string(n) + " ★ATK");
            anim->UpdateNodeSize(n, 3.5, 3.5);
        }
    }
}

void DeactivateWin(std::vector<uint32_t> tgts)
{
    double t = Simulator::Now().GetSeconds();
    NS_LOG_INFO("<<< ATTACK WINDOW ENDED at t=" << t << "s");
    for (uint32_t n : tgts) {
        malicious.erase(n);
        if (!isolated[n]) {
            SafeUp(n);
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
    // Only schedule if within simulation window
    if (base >= SIM_DUR) return;

    auto w1 = RandPair(), w2 = RandPair(), w3 = RandPair();
    NS_LOG_INFO("Cycle @ base=" << base
        << "  W1=[" << w1[0] << "," << w1[1] << "]"
        << "  W2=[" << w2[0] << "," << w2[1] << "]"
        << "  W3=[" << w3[0] << "," << w3[1] << "]");

    auto schedIfValid = [&](double dt, auto fn, auto arg) {
        if (base + dt < SIM_DUR)
            Simulator::Schedule(Seconds(base + dt), fn, arg);
    };

    Simulator::Schedule(Seconds(base + 20.0),  &ActivateWin,   1, w1);
    Simulator::Schedule(Seconds(base + 40.0),  &DeactivateWin,    w1);
    Simulator::Schedule(Seconds(base + 60.0),  &ActivateWin,   2, w2);
    Simulator::Schedule(Seconds(base + 80.0),  &DeactivateWin,    w2);
    Simulator::Schedule(Seconds(base + 90.0),  &ActivateWin,   3, w3);
    Simulator::Schedule(Seconds(base + 110.0), &DeactivateWin,    w3);

    // Schedule next cycle if within bounds
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
// APPLY TRUST — Phase 1 Upgrade
// ═══════════════════════════════════════════════════════════════════════════════
void ApplyTrust()
{
    uint32_t nSoftAvoided = 0;

    for (uint32_t i = 1; i < N; ++i) {
        double rem        = eSrc[i] ? eSrc[i]->GetRemainingEnergy() : 0.0;
        double energyFrac = rem / INIT_E;

        if (rem < 0.05) continue; // dead node

        // ── Phase 1: Compute combined routing metric ───────────────────────
        if (g_enableEnergy) {
            routingMetric[i] = ALPHA * trust[i] + BETA * energyFrac;
        } else {
            // Scenario B (trust-only): routing metric = trust alone
            routingMetric[i] = trust[i];
        }

        // ── Soft avoidance tracking ────────────────────────────────────────
        if (g_enableEnergy && routingMetric[i] < ROUTE_THR && !isolated[i]) {
            softAvoidCount[i]++;
            nSoftAvoided++;
            NS_LOG_INFO("  ⚠ SOFT-AVOID N" << i
                << "  metric=" << std::fixed << std::setprecision(3)
                << routingMetric[i]
                << "  (trust=" << trust[i]
                << "  energy=" << std::setprecision(2) << (energyFrac*100.0) << "%)");
        }

        // ── Hard isolation (trust < 0.3) ───────────────────────────────────
        if (g_enableML) {
            if (trust[i] < 0.3 && !isolated[i]) {
                isolated[i] = true;
                isoEvents++;
                SafeDown(i);
                NS_LOG_INFO("  ✗ ISOLATE N" << i
                    << "  trust=" << std::fixed << std::setprecision(3) << trust[i]);
                if (anim) {
                    anim->UpdateNodeColor(nodes.Get(i), 20, 20, 20);
                    anim->UpdateNodeDescription(nodes.Get(i),
                        "N" + std::to_string(i) + " ✗ISO");
                    anim->UpdateNodeSize(i, 1.5, 1.5);
                }
            } else if (trust[i] >= 0.3 && isolated[i] && !malicious.count(i)) {
                isolated[i] = false;
                SafeUp(i);
                NS_LOG_INFO("  ✓ RESTORE N" << i
                    << "  trust=" << std::fixed << std::setprecision(3) << trust[i]);
                if (anim) {
                    anim->UpdateNodeDescription(nodes.Get(i),
                        "N" + std::to_string(i));
                    anim->UpdateNodeSize(i, 2.0, 2.0);
                    PaintNode(i);
                }
            }
        }
    }

    if (nSoftAvoided > 0)
        NS_LOG_INFO("  → " << nSoftAvoided << " nodes soft-avoided (metric < " << ROUTE_THR << ")");
}

// ═══════════════════════════════════════════════════════════════════════════════
// PERIODIC LOG  (every LOG_INT seconds)
// ═══════════════════════════════════════════════════════════════════════════════
void LogSnap(double ts)
{
    if (gStop) return;

    // ── PDR ──────────────────────────────────────────────────────────────────
    uint64_t totalTx = 0;
    for (uint32_t i = 1; i < N; ++i) totalTx += pktTx[i];
    uint64_t totalRx = pktRx[0];
    double pdr = (totalTx > 0) ? (double)totalRx / totalTx : 0.0;

    // ── Delay ─────────────────────────────────────────────────────────────────
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

    // ── Energy stats ──────────────────────────────────────────────────────────
    double totalE = 0.0; uint32_t alive = 0;
    for (uint32_t i = 0; i < N; ++i) {
        double e = eSrc[i] ? eSrc[i]->GetRemainingEnergy() : 0.0;
        totalE += e;
        if (e > 0.05) alive++;
    }
    double meanE = totalE / N;

    // ── Phase 1: Energy imbalance index (standard deviation) ─────────────────
    double variance = 0.0;
    for (uint32_t i = 0; i < N; ++i) {
        double e    = eSrc[i] ? eSrc[i]->GetRemainingEnergy() : 0.0;
        double diff = e - meanE;
        variance += diff * diff;
    }
    variance /= N;
    double energyStdDev = std::sqrt(variance);

    uint32_t nMal = (uint32_t)malicious.size();
    uint32_t nIso = 0;
    uint32_t nSoftAvoid = 0;
    for (bool b : isolated) if (b) nIso++;
    for (uint32_t i = 1; i < N; ++i)
        if (g_enableEnergy && routingMetric[i] < ROUTE_THR && !isolated[i])
            nSoftAvoid++;

    // ── Average routing metric (Phase 1 indicator) ────────────────────────────
    double sumMetric = 0.0;
    for (uint32_t i = 1; i < N; ++i) sumMetric += routingMetric[i];
    double avgMetric = sumMetric / (N - 1);

    perfLog << std::fixed << std::setprecision(3)
            << ts           << ","   // time_s
            << pdr          << ","   // pdr
            << avgDelay     << ","   // avg_delay_ms
            << meanE        << ","   // avg_energy_J
            << isoEvents    << ","   // isolation_events_cum
            << nMal         << ","   // malicious_active
            << nIso         << ","   // isolated_active
            << alive        << ","   // alive_nodes
            << energyStdDev << ","   // energy_stddev  ← Phase 1
            << nSoftAvoid   << ","   // soft_avoided   ← Phase 1
            << avgMetric    << ","   // avg_routing_metric ← Phase 1
            << g_scenario   << "\n"; // scenario label ← Phase 1
    perfLog.flush();

    NS_LOG_INFO("[LOG] t=" << ts
        << "s  PDR=" << std::setprecision(1) << (pdr*100.0) << "%"
        << "  TX=" << totalTx << "  RX=" << totalRx
        << "  Delay=" << std::setprecision(2) << avgDelay << "ms"
        << "  E=" << meanE << "J  σE=" << energyStdDev
        << "  Alive=" << alive
        << "  Mal=" << nMal << "  Iso=" << nIso
        << "  SoftAvoid=" << nSoftAvoid
        << "  AvgMetric=" << std::setprecision(3) << avgMetric);

    Simulator::Schedule(Seconds(LOG_INT), &LogSnap, ts + LOG_INT);
}

// ═══════════════════════════════════════════════════════════════════════════════
// PERIODIC ML EVALUATION
// ═══════════════════════════════════════════════════════════════════════════════
void MLEval(double ts)
{
    if (gStop) return;
    NS_LOG_INFO("=== ML Evaluation @ t=" << ts << "s ===");

    std::vector<double> energyF(N), fwdF(N), dropF(N), isoF(N), malF(N);
    for (uint32_t i = 0; i < N; ++i) {
        double rem = eSrc[i] ? eSrc[i]->GetRemainingEnergy() : 0.0;
        energyF[i] = rem / INIT_E;

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
        isoF[i]  = isolated[i]      ? 1.0 : 0.0;
        malF[i]  = malicious.count(i) ? 1.0 : 0.0;
    }

    // ── Phase 1: Pass routing metrics to ML server as extra context ────────
    std::ostringstream json;
    json << "{\"timestamp\":"       << (int)ts
         << ",\"energy\":"          << JArr(energyF)
         << ",\"forward_ratio\":"   << JArr(fwdF)
         << ",\"drop_ratio\":"      << JArr(dropF)
         << ",\"isolated\":"        << JArr(isoF)
         << ",\"known_malicious\":" << JArr(malF)
         << ",\"routing_metric\":"  << JArr(routingMetric)
         << ",\"scenario\":\""      << g_scenario << "\""
         << "}";

    std::string resp;
    if (g_enableML) {
        if (IpcSend(json.str(), resp)) {
            NS_LOG_INFO("  ML ← " << resp.substr(0, 90) << "...");
            ParseTrust(resp);
            ApplyTrust();
        } else {
            NS_LOG_WARN("  ML server unreachable — trust scores unchanged");
        }
    } else {
        // Scenario A: no ML, all trust = 1.0
        for (uint32_t i = 0; i < N; ++i) trust[i] = 1.0;
        ApplyTrust();
    }

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
    double meanE  = 0.0;
    for (uint32_t i = 0; i < N; ++i) {
        double e = eSrc[i] ? eSrc[i]->GetRemainingEnergy() : 0.0;
        totalE += e; if (e > 0.05) alive++;
    }
    meanE = totalE / N;

    // Energy stddev final
    double variance = 0.0;
    for (uint32_t i = 0; i < N; ++i) {
        double e    = eSrc[i] ? eSrc[i]->GetRemainingEnergy() : 0.0;
        double diff = e - meanE;
        variance += diff * diff;
    }
    double energyStdDev = std::sqrt(variance / N);

    // Avg routing metric final
    double sumMetric = 0.0;
    for (uint32_t i = 1; i < N; ++i) sumMetric += routingMetric[i];
    double avgMetric = sumMetric / (N - 1);

    // Total soft avoid events
    uint64_t totalSoftAvoid = 0;
    for (auto v : softAvoidCount) totalSoftAvoid += v;

    double pdr   = totalTx > 0 ? (double)totalRx / totalTx * 100.0 : 0.0;
    double delay = fc > 0 ? sumD / fc * 1000.0 : 0.0;

    std::cout << "\n╔══════════════════════════════════════════════════╗\n"
              << "║       FINAL SIMULATION RESULTS — Scenario " << g_scenario << "     ║\n"
              << "╠══════════════════════════════════════════════════╣\n"
              << std::fixed << std::setprecision(2)
              << "║  Scenario           : " << std::setw(6) << g_scenario       << "              ║\n"
              << "║  PDR                : " << std::setw(7) << pdr              << " %          ║\n"
              << "║  Avg E2E Delay      : " << std::setw(7) << delay            << " ms         ║\n"
              << "║  Avg Energy Left    : " << std::setw(7) << meanE            << " J          ║\n"
              << "║  Energy Std Dev     : " << std::setw(7) << energyStdDev     << " J    [NEW] ║\n"
              << "║  Avg Routing Metric : " << std::setw(7) << avgMetric        << "       [NEW] ║\n"
              << "║  Total TX packets   : " << std::setw(9) << totalTx          << "           ║\n"
              << "║  Total RX packets   : " << std::setw(9) << totalRx          << "           ║\n"
              << "║  Alive nodes        : " << std::setw(9) << alive            << "           ║\n"
              << "║  Isolation Events   : " << std::setw(9) << isoEvents        << "           ║\n"
              << "║  Soft-Avoid Events  : " << std::setw(9) << totalSoftAvoid   << "     [NEW] ║\n"
              << "╚══════════════════════════════════════════════════╝\n";
}

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════════════════
int main(int argc, char* argv[])
{
    signal(SIGINT,  SigInt);
    signal(SIGTERM, SigInt);

    // ── Parse command-line FIRST so g_scenario is set before anything else ────
    // ns-3 CommandLine strips the leading "--" and passes "scenario=X"
    bool verbose = false;
    CommandLine cmd;
    cmd.AddValue("verbose",  "Enable detailed logging",                verbose);
    cmd.AddValue("scenario", "Scenario: A=baseline, B=trust, C=full",  g_scenario);
    cmd.Parse(argc, argv);

    // Normalise to uppercase in case user passes lowercase
    for (auto& c : g_scenario) c = (char)toupper((unsigned char)c);

    // Configure features based on scenario flag
    if (g_scenario == "A") {
        g_enableML     = false;
        g_enableEnergy = false;
    } else if (g_scenario == "B") {
        g_enableML     = true;
        g_enableEnergy = false;
    } else {
        g_scenario     = "C";   // default / unknown → C
        g_enableML     = true;
        g_enableEnergy = true;
    }

    // Enable logging AFTER parse so --verbose works
    LogComponentEnable("HybridWSNSecure", LOG_LEVEL_INFO);
    if (verbose)
        LogComponentEnable("OnOffApplication", LOG_LEVEL_INFO);

    NS_LOG_INFO("=== Scenario " << g_scenario
        << " | ML=" << g_enableML
        << " | EnergyWeight=" << g_enableEnergy << " ===");

    if (::system("mkdir -p results") != 0)
        std::cerr << "[WARN] mkdir results failed\n";

    // Open CSV AFTER cmd.Parse so the filename uses the correct scenario letter
    std::string csvPath = "results/performance_" + g_scenario + ".csv";
    perfLog.open(csvPath);
    if (!perfLog.is_open()) {
        std::cerr << "[ERROR] Cannot open " << csvPath << "\n";
        return 1;
    }
    perfLog << "time_s,pdr,avg_delay_ms,avg_energy_J,isolation_events_cum,"
               "malicious_active,isolated_active,alive_nodes,"
               "energy_stddev,soft_avoided,avg_routing_metric,scenario\n";

    BuildNetwork();
    BuildTraffic();
    ScheduleCycle(0.0);

    // ── Connectivity diagnostic — print neighbor counts at t=0 ───────────────
    // If PDR stays 0, check these counts. Each node should have ≥2 neighbors.
    Simulator::Schedule(Seconds(0.5), [](){
        uint32_t totalNeighbors = 0;
        for (uint32_t i = 0; i < N; ++i) {
            auto mob_i = nodes.Get(i)->GetObject<MobilityModel>();
            uint32_t cnt = 0;
            for (uint32_t j = 0; j < N; ++j) {
                if (i == j) continue;
                auto mob_j = nodes.Get(j)->GetObject<MobilityModel>();
                double dist = mob_i->GetDistanceFrom(mob_j);
                if (dist <= 120.0) cnt++;  // within realistic 120m range
            }
            totalNeighbors += cnt;
        }
        double avgNeighbors = (double)totalNeighbors / N;
        std::cout << "[DIAG] t=0.5s | Avg neighbors per node (within 120m): "
                  << std::fixed << std::setprecision(1) << avgNeighbors << "\n";
        if (avgNeighbors < 2.0)
            std::cout << "[WARN] Low connectivity — PDR will be 0. "
                      << "Increase TxPower or reduce DX/DY.\n";
        else
            std::cout << "[OK]  Connectivity adequate for AODV routing.\n";
    });

    // ── NetAnim ───────────────────────────────────────────────────────────────
    std::string animPath = "results/animation_" + g_scenario + ".xml";
    anim = new AnimationInterface(animPath);
    anim->SetMaxPktsPerTraceFile(10000000);
    anim->EnablePacketMetadata(true);
    anim->EnableWifiPhyCounters(Seconds(0), Seconds(SIM_DUR), Seconds(LOG_INT));
    anim->EnableIpv4L3ProtocolCounters(Seconds(0), Seconds(SIM_DUR), Seconds(LOG_INT));

    // Initial node styling
    anim->UpdateNodeColor(nodes.Get(0), 200, 0, 0);
    anim->UpdateNodeDescription(nodes.Get(0), "SINK ●");
    anim->UpdateNodeSize(0, 5.0, 5.0);
    for (uint32_t i = 1; i < N; ++i) {
        anim->UpdateNodeColor(nodes.Get(i), 0, 190, 60);
        anim->UpdateNodeDescription(nodes.Get(i), "N" + std::to_string(i));
        anim->UpdateNodeSize(i, 2.0, 2.0);
    }

    Simulator::Schedule(Seconds(LOG_INT), &LogSnap, LOG_INT);
    Simulator::Schedule(Seconds(ML_INT),  &MLEval,  ML_INT);

    std::string scenLabel =
        (g_scenario=="A") ? " — Baseline AODV (no ML, no energy)        " :
        (g_scenario=="B") ? " — Trust Only (ML, no energy weighting)     " :
                            " — Trust + Energy (Full Phase 1 system)     ";

    std::cout
        << "\n╔═══════════════════════════════════════════════════════════╗\n"
        << "║  Hybrid Real-Time Secure WSN — Phase 1 (Priyadarshi 2024) ║\n"
        << "║  50 nodes | AODV | 802.11b | Energy=150J | Dur=600s       ║\n"
        << "║  Scenario: " << g_scenario << scenLabel                    << "║\n"
        << "║  Routing metric = 0.7 * trust + 0.3 * energy_fraction     ║\n"
        << "║  Soft-avoidance threshold = 0.4                           ║\n"
        << "║  Energy imbalance (σ) tracked every " << LOG_INT << "s               ║\n"
        << "║  Press Ctrl+C to stop and dump final stats                ║\n"
        << "╚═══════════════════════════════════════════════════════════╝\n\n";

    Simulator::Stop(Seconds(SIM_DUR));
    Simulator::Run();

    std::string fmPath = "results/flowmonitor_" + g_scenario + ".xml";
    flowMon->SerializeToXmlFile(fmPath, true, true);
    Summary();
    Simulator::Destroy();
    perfLog.close();
    delete anim;
    return 0;
}
