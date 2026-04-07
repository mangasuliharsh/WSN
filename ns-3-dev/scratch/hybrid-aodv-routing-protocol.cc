/*
 * hybrid-aodv-routing-protocol.cc  [v5.2 — build-error-free]
 *
 * KEY DESIGN CONSTRAINTS (from aodv-routing-protocol.h inspection):
 *
 *  [C-1] aodv::RoutingProtocol::m_ipv4  is PRIVATE.
 *        We cache our own  Ptr<Ipv4> m_hybridIpv4  in SetIpv4().
 *
 *  [C-2] aodv::RoutingProtocol::RecvAodv()  is PRIVATE.
 *        Cannot be called or overridden.  The RecvAodv() member is
 *        removed entirely from this class.  Candidate harvest is done
 *        by periodic polling (LookupSinkRoute) in wsn.cc every ML_INT s.
 *
 *  [C-3] aodv::RoutingTable  has no default constructor — needs Time arg.
 *        m_hybridTable is initialised with Seconds(10) in ctor init-list.
 *
 *  [C-4] LookupSinkRoute() is const but must call non-const RouteOutput().
 *        Use const_cast<> on this where required.
 *
 * ROUTE INJECTION STRATEGY:
 *   InjectMLRoute() stores ONLY the next-hop Ipv4Address in m_mlNextHop[].
 *   RouteOutput() builds a FRESH Ptr<Ipv4Route> on every call, avoiding
 *   stale source/device entries that arise with SafeDown/SafeUp and
 *   RandomWaypoint mobility (Scenario D).
 */

#include "hybrid-aodv-routing-protocol.h"

#include "ns3/aodv-packet.h"
#include "ns3/aodv-rtable.h"
#include "ns3/ipv4-route.h"
#include "ns3/ipv4.h"
#include "ns3/log.h"
#include "ns3/node.h"
#include "ns3/simulator.h"
#include "ns3/socket.h"

#include <algorithm>

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("HybridAodvRoutingProtocol");
NS_OBJECT_ENSURE_REGISTERED(HybridAodvRoutingProtocol);

// ---------------------------------------------------------------------------
// TypeId
// ---------------------------------------------------------------------------
TypeId
HybridAodvRoutingProtocol::GetTypeId()
{
    static TypeId tid =
        TypeId("ns3::HybridAodvRoutingProtocol")
            .SetParent<aodv::RoutingProtocol>()
            .SetGroupName("Aodv")
            .AddConstructor<HybridAodvRoutingProtocol>();
    return tid;
}

// ---------------------------------------------------------------------------
// Constructor / Destructor
// [C-3] RoutingTable requires a Time argument — use Seconds(10).
// ---------------------------------------------------------------------------
HybridAodvRoutingProtocol::HybridAodvRoutingProtocol()
    : m_hybridTable(Seconds(10)),
      m_hybridIpv4(nullptr),
      m_candidateCb(nullptr),
      m_nodeId(UINT32_MAX)
{
}

HybridAodvRoutingProtocol::~HybridAodvRoutingProtocol()
{
}

// ---------------------------------------------------------------------------
// SetIpv4
// [C-1] Cache our own Ptr<Ipv4> because the parent's m_ipv4 is private.
// ---------------------------------------------------------------------------
void
HybridAodvRoutingProtocol::SetIpv4(Ptr<Ipv4> ipv4)
{
    aodv::RoutingProtocol::SetIpv4(ipv4);
    m_hybridIpv4 = ipv4;
    m_nodeId     = ipv4->GetObject<Node>()->GetId();
    NS_LOG_INFO("[HybridAODV] N" << m_nodeId << " SetIpv4 done");
}

// ---------------------------------------------------------------------------
// DoInitialize
// ---------------------------------------------------------------------------
void
HybridAodvRoutingProtocol::DoInitialize()
{
    aodv::RoutingProtocol::DoInitialize();
    NS_LOG_INFO("[HybridAODV] N" << m_nodeId << " initialized");
}

// ---------------------------------------------------------------------------
// FindUpInterface — returns first UP non-loopback interface index or UINT32_MAX
// ---------------------------------------------------------------------------
uint32_t
HybridAodvRoutingProtocol::FindUpInterface() const
{
    if (!m_hybridIpv4) return UINT32_MAX;
    for (uint32_t i = 1; i < m_hybridIpv4->GetNInterfaces(); ++i)
        if (m_hybridIpv4->IsUp(i)) return i;
    return UINT32_MAX;
}

// ---------------------------------------------------------------------------
// RouteOutput — ML-locked route has priority; builds FRESH route every call
// ---------------------------------------------------------------------------
Ptr<Ipv4Route>
HybridAodvRoutingProtocol::RouteOutput(Ptr<Packet>          p,
                                       const Ipv4Header&     header,
                                       Ptr<NetDevice>        oif,
                                       Socket::SocketErrno&  sockerr)
{
    Ipv4Address dst = header.GetDestination();

    if (IsMLLocked(dst))
    {
        auto nhIt = m_mlNextHop.find(dst);
        if (nhIt != m_mlNextHop.end())
        {
            Ipv4Address nextHop = nhIt->second;

            if (nextHop != Ipv4Address("127.0.0.1") &&
                nextHop != Ipv4Address("0.0.0.0")   &&
                nextHop != Ipv4Address())
            {
                uint32_t outIface = FindUpInterface();
                if (outIface != UINT32_MAX)
                {
                    Ptr<Ipv4Route> rt = Create<Ipv4Route>();
                    rt->SetDestination(dst);
                    rt->SetGateway(nextHop);
                    rt->SetSource(m_hybridIpv4->GetAddress(outIface, 0).GetLocal());
                    rt->SetOutputDevice(m_hybridIpv4->GetNetDevice(outIface));

                    NS_LOG_INFO("[HybridAODV] N" << m_nodeId
                                << " RouteOutput ML-locked to " << dst
                                << " via " << nextHop);
                    sockerr = Socket::ERROR_NOTERROR;
                    return rt;
                }
            }
            NS_LOG_WARN("[HybridAODV] N" << m_nodeId
                        << " RouteOutput ML nextHop " << nextHop
                        << " unusable, falling back to AODV");
        }
    }

    return aodv::RoutingProtocol::RouteOutput(p, header, oif, sockerr);
}

// ---------------------------------------------------------------------------
// Candidate API
// ---------------------------------------------------------------------------
std::vector<CandidateRoute>
HybridAodvRoutingProtocol::GetCandidates(Ipv4Address dst) const
{
    auto it = m_candidates.find(dst);
    if (it == m_candidates.end()) return {};
    return it->second;
}

void
HybridAodvRoutingProtocol::ClearCandidates(Ipv4Address dst)
{
    m_candidates.erase(dst);
}

// ---------------------------------------------------------------------------
// AppendCandidate — called by wsn.cc's neighbour-harvest loop (FIX-1).
// Deduplicates by next-hop address; updates hop count if entry already exists.
// ---------------------------------------------------------------------------
void
HybridAodvRoutingProtocol::AppendCandidate(const CandidateRoute& c)
{
    auto& list = m_candidates[c.destination];
    for (auto& ex : list)
    {
        if (ex.nextHop == c.nextHop)
        {
            // Refresh hop count and lifetime from fresher data
            ex.hopCount = c.hopCount;
            ex.lifetime = c.lifetime;
            return;
        }
    }
    list.push_back(c);
    NS_LOG_INFO("[HybridAODV] N" << m_nodeId
                << " AppendCandidate dst=" << c.destination
                << " via=" << c.nextHop
                << " hops=" << (int)c.hopCount);
}

// ---------------------------------------------------------------------------
// InjectMLRoute
// [C-1] Uses m_hybridIpv4 instead of private m_ipv4.
// [FIX-STALE] Stores only next-hop address; fresh route built in RouteOutput.
// ---------------------------------------------------------------------------
void
HybridAodvRoutingProtocol::InjectMLRoute(Ipv4Address dst,
                                          Ipv4Address nextHop,
                                          uint8_t     hopCount,
                                          Time        lifetime)
{
    if (!m_hybridIpv4)
    {
        NS_LOG_WARN("[HybridAODV] N" << m_nodeId << " InjectMLRoute: no Ipv4");
        return;
    }

    if (nextHop == Ipv4Address("127.0.0.1") ||
        nextHop == Ipv4Address("0.0.0.0")   ||
        nextHop == Ipv4Address())
    {
        NS_LOG_WARN("[HybridAODV] N" << m_nodeId
                    << " InjectMLRoute: refusing bad nextHop " << nextHop);
        return;
    }

    uint32_t outIface = FindUpInterface();
    if (outIface == UINT32_MAX)
    {
        NS_LOG_WARN("[HybridAODV] N" << m_nodeId << " InjectMLRoute: no UP interface");
        return;
    }

    m_mlNextHop[dst]  = nextHop;
    m_mlHopCount[dst] = hopCount;
    m_mlLocked.insert(dst);

    // Update shadow table so LookupSinkRoute returns a real hop count next cycle
    aodv::RoutingTableEntry newEntry(
        m_hybridIpv4->GetNetDevice(outIface),
        dst,
        /*validSeqNo=*/ true,
        /*seqNo=*/      0,
        m_hybridIpv4->GetAddress(outIface, 0),
        hopCount,
        nextHop,
        lifetime);
    newEntry.SetFlag(aodv::VALID);

    if (!m_hybridTable.Update(newEntry))
        m_hybridTable.AddRoute(newEntry);

    NS_LOG_INFO("[HybridAODV] N" << m_nodeId
                << " ML-INJECT dst=" << dst
                << " via=" << nextHop
                << " hops=" << (int)hopCount
                << " lifetime=" << lifetime.GetSeconds() << "s [LOCKED]");
}

// ---------------------------------------------------------------------------
// UnlockRoute
// ---------------------------------------------------------------------------
void
HybridAodvRoutingProtocol::UnlockRoute(Ipv4Address dst)
{
    m_mlLocked.erase(dst);
    m_mlNextHop.erase(dst);
    m_mlHopCount.erase(dst);
    NS_LOG_INFO("[HybridAODV] N" << m_nodeId << " UNLOCK dst=" << dst);
}

bool
HybridAodvRoutingProtocol::IsMLLocked(Ipv4Address dst) const
{
    return m_mlLocked.count(dst) > 0;
}

// ---------------------------------------------------------------------------
// LookupSinkRoute
// [C-1] Use m_hybridIpv4.
// [C-4] RouteOutput is non-const — call via non-const pointer.
// ---------------------------------------------------------------------------
bool
HybridAodvRoutingProtocol::LookupSinkRoute(Ipv4Address              dst,
                                            aodv::RoutingTableEntry& entry) const
{
    // 1. Shadow table first (has real hop counts from previous inject cycles)
    if (const_cast<aodv::RoutingTable&>(m_hybridTable).LookupValidRoute(dst, entry))
        return true;

    // 2. Probe AODV's table via parent RouteOutput.
    //    RouteOutput is non-const so we need a mutable this pointer. [C-4]
    if (!m_hybridIpv4) return false;

    Ptr<Packet>         dummyPkt = Create<Packet>();
    Ipv4Header          hdr;
    hdr.SetDestination(dst);
    Socket::SocketErrno err = Socket::ERROR_NOTERROR;

    // Call the AODV base RouteOutput directly (bypasses our ML-lock check).
    // We must cast away const because RouteOutput is non-const. [C-4]
    auto* mutableThis = const_cast<HybridAodvRoutingProtocol*>(this);
    Ptr<Ipv4Route> route =
        mutableThis->aodv::RoutingProtocol::RouteOutput(dummyPkt, hdr, nullptr, err);

    if (!route || err != Socket::ERROR_NOTERROR) return false;

    Ipv4Address gw = route->GetGateway();
    if (gw == Ipv4Address("127.0.0.1") ||
        gw == Ipv4Address("0.0.0.0")   ||
        gw == Ipv4Address())
        return false;

    Ptr<NetDevice> dev   = route->GetOutputDevice();
    uint32_t       ifIdx = m_hybridIpv4->GetInterfaceForDevice(dev);

    // hop count from RouteOutput is not exposed; use 1 as floor.
    // The next InjectMLRoute() call will update the shadow table with the real value.
    entry = aodv::RoutingTableEntry(
        dev,
        dst,
        /*validSeqNo=*/ false,
        /*seqNo=*/      0,
        m_hybridIpv4->GetAddress(ifIdx, 0),
        /*hops=*/       1,
        gw,
        Simulator::Now() + Seconds(2));
    entry.SetFlag(aodv::VALID);
    return true;
}

// ---------------------------------------------------------------------------
// HookRecvReply  (called from wsn.cc trace if wired up)
// ---------------------------------------------------------------------------
void
HybridAodvRoutingProtocol::HookRecvReply(Ptr<Packet>  p,
                                          Ipv4Address  /*receiver*/,
                                          Ipv4Address  sender)
{
    aodv::RrepHeader rrepHdr;
    Ptr<Packet> copy = p->Copy();
    copy->RemoveHeader(rrepHdr);

    Ipv4Address dst = rrepHdr.GetDst();
    if (dst == rrepHdr.GetOrigin()) return;   // Hello message — skip

    uint8_t hops = rrepHdr.GetHopCount() + 1;

    CandidateRoute c;
    c.destination = dst;
    c.nextHop     = sender;
    c.sender      = sender;
    c.hopCount    = hops;
    c.seqNo       = rrepHdr.GetDstSeqno();
    c.lifetime    = rrepHdr.GetLifeTime();

    auto& list = m_candidates[dst];
    for (auto& ex : list)
    {
        if (ex.nextHop == c.nextHop)
        {
            ex.hopCount = hops;
            ex.seqNo    = c.seqNo;
            ex.lifetime = c.lifetime;
            if (m_candidateCb) m_candidateCb(m_nodeId, c);
            return;
        }
    }
    list.push_back(c);
    NS_LOG_INFO("[HybridAODV] N" << m_nodeId
                << " NEW candidate " << dst << " via " << sender
                << " hops=" << (int)hops);
    if (m_candidateCb) m_candidateCb(m_nodeId, c);
}

// ---------------------------------------------------------------------------
// HybridAodvHelper
// ---------------------------------------------------------------------------
HybridAodvHelper::HybridAodvHelper()
{
    m_factory.SetTypeId("ns3::HybridAodvRoutingProtocol");
}

HybridAodvHelper*
HybridAodvHelper::Copy() const
{
    return new HybridAodvHelper(*this);
}

Ptr<Ipv4RoutingProtocol>
HybridAodvHelper::Create(Ptr<Node> node) const
{
    Ptr<HybridAodvRoutingProtocol> agent =
        m_factory.Create<HybridAodvRoutingProtocol>();
    node->AggregateObject(agent);   // ← AODV needs GetNode() to be valid
    return agent;
}

void
HybridAodvHelper::Set(std::string name, const AttributeValue& value)
{
    m_factory.Set(name, value);
}

} // namespace ns3
