#include <string>
#include <vector>
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

    for (kcc_int32 waveopIdx = 0; waveopIdx < numWaveops; ++waveopIdx) {
        const auto waveop = m_WaveOps[waveopIdx];
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

        auto nopWaveop = rewireMultiOutEdgesOfOneMatMul(matmulWaveop);
        newWaveops.push_back(matmulWaveop);
        newWaveops.push_back(nopWaveop);
    }

    std::swap(newWaveops, m_WaveOps);
}

/****************************************************************
 *                                                              *
 ****************************************************************/
wave::NopWaveOp*
Network::rewireMultiOutEdgesOfOneMatMul(wave::MatMulWaveOp* matmulWaveop)
{
    Assert(matmulWaveop->gSuccWaveEdges().size() > 1, "MatMults with 1 successor should be skipped");
    std::vector<wave::WaveEdge*> syncedEdges;
    for (auto succEdge : matmulWaveop->gSuccWaveEdges()) {
        if (succEdge->qNeedToSync()) {
            syncedEdges.push_back(succEdge);
        }
    }
    Assert(syncedEdges.size() >= 2, "Rewiring MatMul ", matmulWaveop->gName(), " has fewer than 2 outgoing synced edges");
    kcc_int32 minSuccLevel = std::numeric_limits<decltype(minSuccLevel)>::max();
    wave::WaveEdge* minSuccEdge = nullptr;


    for (auto succEdge : syncedEdges) {
        const auto succWaveop = succEdge->gToOp();
        if (! minSuccEdge) {
            minSuccEdge = succEdge;
            minSuccLevel = succWaveop->gLevel();
        } else if (succWaveop->gLevel() < minSuccLevel) {
            minSuccEdge = succEdge;
            minSuccLevel = succWaveop->gLevel();
        }
    }
    Assert(minSuccEdge, "Must have min succ edge for rewiring");

    const EngineId engId = EngineId::Pooling;

    // Disconnect succ edges from matmul, say MM->A, MM->B, MM->C
    for (auto succSyncedEdge : syncedEdges) {
        matmulWaveop->DisconnectSuccEdge(utils::Passkey<Network>(), succSyncedEdge);
    }

    // New Nop on PoolEng. This will create edge MatMul->Nop
    wave::NopWaveOp::Params params;
    params.m_WaveOpName = (std::string("nop-") + matmulWaveop->gName());
    std::vector<wave::WaveOp*> prevWaveOps;
    prevWaveOps.push_back(matmulWaveop);
    auto nopWaveop = new wave::NopWaveOp(params, prevWaveOps, engId, events::EventId_Invalid(),
                                         wave::NopWaveOp::NopType::Broadcast);
    nopWaveop->rLevel( (matmulWaveop->gLevel() + minSuccLevel) / 2);  // middle

    // Connect edges Nop->A, Nop->B, Nop->C
    for (auto succSyncedEdge : syncedEdges) {
        nopWaveop->addSuccWaveEdge(succSyncedEdge);
    }

    std::cout << "Rewired MatMul: " << matmulWaveop->gName() << "\n";

    return nopWaveop;
}

}}


