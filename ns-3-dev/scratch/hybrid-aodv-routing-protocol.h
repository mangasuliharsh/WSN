/*
 * hybrid-aodv-routing-protocol.h  [v5.2 — build-error-free]
 *
 * Hybrid AODV + ML Controlled Routing — Phase 2
 * =================================================
 *
 * Changes from v5.1:
 *  [C-1] m_hybridIpv4 added — our own Ptr<Ipv4> because the parent's
 *        m_ipv4 is private and inaccessible from derived classes.
 *  [C-2] RecvAodv() declaration removed — parent method is private,
 *        cannot be overridden.
 *  [C-3] m_hybridTable(Seconds(10)) — RoutingTable has no default ctor.
 *  [C-4] LookupSinkRoute const_cast fix documented in .cc.
 *  [FIX-STALE] m_mlRoutes removed; replaced by m_mlNextHop + m_mlHopCount
 *              so RouteOutput always builds a fresh Ptr<Ipv4Route>.
 *  [FIX-LOOPBACK] InjectMLRoute and RouteOutput reject 127.0.0.1/0.0.0.0.
 */

#ifndef HYBRID_AODV_ROUTING_PROTOCOL_H
#define HYBRID_AODV_ROUTING_PROTOCOL_H

#include "ns3/aodv-routing-protocol.h"
#include "ns3/aodv-rtable.h"
#include "ns3/ipv4-routing-helper.h"
#include "ns3/object-factory.h"

#include <map>
#include <vector>
#include <set>
#include <string>
#include <functional>

namespace ns3
{

// ---------------------------------------------------------------------------
// CandidateRoute — one entry per RREP received for a destination
// ---------------------------------------------------------------------------
struct CandidateRoute
{
    Ipv4Address destination;
    Ipv4Address nextHop;
    Ipv4Address sender;
    uint8_t     hopCount       = 0;
    uint32_t    seqNo          = 0;
    double      residualEnergy = 1.0;
    double      trustScore     = 0.7;
    double      score          = 0.0;
    Time        lifetime;

    bool operator==(const CandidateRoute& o) const
    {
        return destination == o.destination && nextHop == o.nextHop;
    }
};

// Callback: void cb(uint32_t nodeId, const CandidateRoute& c)
typedef std::function<void(uint32_t, const CandidateRoute&)> CandidateCallback;

// ---------------------------------------------------------------------------
// HybridAodvRoutingProtocol
// ---------------------------------------------------------------------------
class HybridAodvRoutingProtocol : public aodv::RoutingProtocol
{
  public:
    static TypeId GetTypeId();

    HybridAodvRoutingProtocol();
    ~HybridAodvRoutingProtocol() override;

    // ── External API (called by wsn.cc) ────────────────────────────────────

    void SetCandidateCallback(CandidateCallback cb) { m_candidateCb = cb; }

    std::vector<CandidateRoute> GetCandidates(Ipv4Address dst) const;
    void ClearCandidates(Ipv4Address dst);

    /**
     * Inject an ML-selected route and lock it.
     * Stores only the next-hop address; a fresh Ipv4Route is built on
     * every RouteOutput() call to avoid stale interface references.
     */
    void InjectMLRoute(Ipv4Address dst,
                       Ipv4Address nextHop,
                       uint8_t     hopCount,
                       Time        lifetime);

    void UnlockRoute(Ipv4Address dst);
    bool IsMLLocked(Ipv4Address dst) const;

    /**
     * Look up the current AODV/shadow route for dst.
     * Returns true if a VALID route was found and fills entry.
     */
    bool LookupSinkRoute(Ipv4Address dst, aodv::RoutingTableEntry& entry) const;

    /** Directly append a candidate (called by wsn.cc harvest loop). */
    void AppendCandidate(const CandidateRoute& c);

    /**
     * Called from an external RREP trace if wired by wsn.cc.
     * Parses the RREP header and adds a candidate entry.
     */
    void HookRecvReply(Ptr<Packet> p, Ipv4Address receiver, Ipv4Address sender);

    // ── Override ────────────────────────────────────────────────────────────
    Ptr<Ipv4Route> RouteOutput(Ptr<Packet> p,
                               const Ipv4Header& header,
                               Ptr<NetDevice> oif,
                               Socket::SocketErrno& sockerr) override;

  private:
    // ── Helpers ─────────────────────────────────────────────────────────────
    void     DoInitialize() override;
    void     SetIpv4(Ptr<Ipv4> ipv4) override;
    uint32_t FindUpInterface() const;   // first UP non-loopback iface index

    // ── State ────────────────────────────────────────────────────────────────

    /** Shadow routing table — stores ML-injected entries with real hop counts. */
    aodv::RoutingTable m_hybridTable;   // [C-3] initialised with Seconds(10) in ctor

    /** [C-1] Our own Ipv4 pointer — parent's m_ipv4 is private. */
    Ptr<Ipv4> m_hybridIpv4;

    /** Per-destination candidate lists. */
    std::map<Ipv4Address, std::vector<CandidateRoute>> m_candidates;

    /** Locked destinations. */
    std::set<Ipv4Address> m_mlLocked;

    /** [FIX-STALE] Next-hop address per locked destination.
     *  RouteOutput() builds a fresh Ipv4Route from this each call. */
    std::map<Ipv4Address, Ipv4Address> m_mlNextHop;
    std::map<Ipv4Address, uint8_t>     m_mlHopCount;

    CandidateCallback m_candidateCb;
    uint32_t          m_nodeId = UINT32_MAX;
};

// ---------------------------------------------------------------------------
// Helper (mirrors AodvHelper pattern)
// ---------------------------------------------------------------------------
class HybridAodvHelper : public Ipv4RoutingHelper
{
  public:
    HybridAodvHelper();
    HybridAodvHelper* Copy() const override;
    Ptr<Ipv4RoutingProtocol> Create(Ptr<Node> node) const override;
    void Set(std::string name, const AttributeValue& value);

  private:
    ObjectFactory m_factory;
};

} // namespace ns3

#endif /* HYBRID_AODV_ROUTING_PROTOCOL_H */
