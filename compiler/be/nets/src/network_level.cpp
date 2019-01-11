#include <string>
#include <vector>
#include <set>
#include <map>
#include <limits>


#include "utils/inc/asserter.hpp"
#include "utils/inc/types.hpp"
#include "utils/inc/passkey.hpp"

#include "arch/inc/arch.hpp"


#include "wave/inc/waveedge.hpp"
#include "wave/inc/matmulwaveop.hpp"
#include "wave/inc/nopwaveop.hpp"

#include "nets/inc/network.hpp"

namespace kcc {
namespace nets {

enum {
    MM_REWIRE_DEBUG = 0
};

/****************************************************************
 *                                                              *
 ****************************************************************/
//--------------------------------------------------------
void
Network::levelizeByLongestPath()
{
    for (auto waveop : m_WaveOps) {
        waveop->rLevel(-LevelDelta);
    }
    const kcc_int32 numWaveops = m_WaveOps.size();

    // move forward
    for (kcc_int32 waveopIdx = 0; waveopIdx < numWaveops; ++waveopIdx) {
        const auto waveop = m_WaveOps[waveopIdx];
        kcc_int32 maxLevel = -LevelDelta;
        for (auto prevWaveop : waveop->gPrevWaveops()) {
            const auto prevLevel = prevWaveop->gLevel();
            Assert(prevLevel >= 0, "Prev waveop " , prevWaveop->gName(), " has negative level");
            if (prevLevel > maxLevel) {
                maxLevel = prevLevel;
            }
        }
        waveop->rLevel(maxLevel + LevelDelta);
    }
}

/****************************************************************
 *                                                              *
 ****************************************************************/
void
Network::RewireMultiOutEdgesOfMatMults()
{
    std::vector<wave::WaveOp*> newWaveops;
    levelizeByLongestPath();
    const kcc_int32 numWaveops = m_WaveOps.size();
    size_t numNops = 0;

    std::map<wave::WaveOp*, wave::NopWaveOp*> waveop2nop;
    std::set<wave::NopWaveOp*> processedNops;

    for (kcc_int32 waveopIdx = 0; waveopIdx < numWaveops; ++waveopIdx) {
        const auto waveop = m_WaveOps[waveopIdx];

        const auto mapIt = waveop2nop.find(waveop);
        if (mapIt != waveop2nop.end()) {
            // This is a succ of a Nop, add Nop first if not added already
            auto nop = (*mapIt).second;
            const auto foundIt = processedNops.find(nop);
            if (foundIt == processedNops.end() ) {
                processedNops.insert(nop);
                newWaveops.push_back(nop);
            }
        }

        //==========
        const auto matmulWaveop = dynamic_cast<wave::MatMulWaveOp*>(waveop);
        if (! matmulWaveop) {
            newWaveops.push_back(waveop);
            continue;
        }
        kcc_int32 numSyncedEdges = 0;
        for (auto outEdge : matmulWaveop->gSuccWaveEdges()) {
            if (outEdge->qNeedToSync()) {
                numSyncedEdges++;
            }
        }
        if (numSyncedEdges <= 1) {
            newWaveops.push_back(waveop);
            continue;
        }
        //----------
        auto nopWaveop = rewireMultiOutEdgesOfOneMatMul(matmulWaveop);
        ++numNops;
        newWaveops.push_back(matmulWaveop);
        //==========

        // Remember Nop to add just before its earliest successor (ALAP)
        for (auto succ : nopWaveop->gSuccWaveops()) {
            waveop2nop[succ] = nopWaveop;
        }
    }

    Assert(numNops == processedNops.size(),
        "MM rewiring: number of created and processed Nops should be equal");
    std::swap(newWaveops, m_WaveOps);
}

/****************************************************************
 *                                                              *
 ****************************************************************/
wave::NopWaveOp*
Network::rewireMultiOutEdgesOfOneMatMul(wave::MatMulWaveOp* matmulWaveop)
{
    Assert(matmulWaveop->gSuccWaveEdges().size() > 1,
        "MatMults with 1 successor should be skipped");
    std::vector<wave::WaveEdge*> syncedEdges;
    for (auto succEdge : matmulWaveop->gSuccWaveEdges()) {
        if (succEdge->qNeedToSync()) {
            syncedEdges.push_back(succEdge);
        }
    }
    Assert(syncedEdges.size() >= 2, "Rewiring MatMul ", matmulWaveop->gName(),
        " has fewer than 2 outgoing synced edges");

    // Min level and min ord are not necessarily the same Waveop:
    // A->B->C->D   E
    // +------------^
    // lev(D)=3 > lev(E)=1  but ord(D)=3 < ord(E)=4 
    kcc_int32 minSuccLevel = std::numeric_limits<decltype(minSuccLevel)>::max();
    wave::WaveEdge* minSuccEdge = nullptr;
    kcc_int32 minWopOrd = std::numeric_limits<decltype(minWopOrd)>::max();
    wave::WaveEdge* minOrdSuccEdge = nullptr;

    for (auto succEdge : syncedEdges) {
        const auto succWaveop = succEdge->gToOp();
        if (! minSuccEdge || succWaveop->gLevel() < minSuccLevel) {
            minSuccEdge = succEdge;
            minSuccLevel = succWaveop->gLevel();
        }
        if (!minOrdSuccEdge || succWaveop->gOrder() < minWopOrd) {
            minOrdSuccEdge = succEdge;
            minWopOrd = succWaveop->gOrder();
        }
    }
    const EngineId engId = minOrdSuccEdge->gToOp()->gEngineId();
    Assert(minSuccEdge, "Must have min succ edge for rewiring");


    // Disconnect succ edges from matmul, say MM->A, MM->B, MM->C
    for (auto succSyncedEdge : syncedEdges) {
        matmulWaveop->DisconnectSuccEdge(utils::Passkey<Network>(), succSyncedEdge);
    }

    // New Nop on PoolEng. This will create edge MatMul->Nop
    wave::NopWaveOp::Params params;
    params.m_WaveOpName = (std::string("nop-") + matmulWaveop->gName());
    std::vector<wave::WaveOp*> prevWaveOps;
    prevWaveOps.push_back(matmulWaveop);
    auto nopWaveop = new wave::NopWaveOp(params, prevWaveOps, engId,
                                         events::EventId_Invalid(),
                                         wave::NopWaveOp::NopType::Broadcast);
    nopWaveop->rLevel( (matmulWaveop->gLevel() + minSuccLevel) / 2);  // middle

    // Connect edges Nop->A, Nop->B, Nop->C
    for (auto succSyncedEdge : syncedEdges) {
        nopWaveop->addSuccWaveEdge(succSyncedEdge);
    }

    if (MM_REWIRE_DEBUG) {
        std::cout << "Rewired MatMul: " << matmulWaveop->gName() << "\n";
    }

    return nopWaveop;
}

}}


