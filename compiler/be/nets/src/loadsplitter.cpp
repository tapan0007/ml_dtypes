#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <array>
#include <map>


#include "utils/inc/asserter.hpp"
#include "arch/inc/arch.hpp"


#include "wave/inc/sbatomloadwaveop.hpp"
#include "wave/inc/sbatomsavewaveop.hpp"
#include "wave/inc/tpbcopywaveop.hpp"
#include "wave/inc/waveedge.hpp"

#include "nets/inc/network.hpp"
#include "nets/inc/loadsplitter.hpp"

namespace kcc {
namespace nets {

//--------------------------------------------------------
/*
**********************************************************
*  From:                                                 *
*  W0 S0  W1 S1  W2 S2  W3 S3  W4 S4                     *
*   \ /    \ /    \ /    \ /    \ /                      *
*   LD0 -> LD1 -> LD2 -> LD3 -> LD4                      *
*                                                        *
*  To:                                                   *
*  W0 S0  W1 S1  W2 S2  W3 S3  W4 S4                     *
*   \ /    \ /    \ /    \ /    \ /                      *
*   LD0 -> LD1 -> LD2 -> LD3 -> LD4                      *
*      \      \      \      \                            *
*       \      \      \      \                           *
*        \      \      \      \                          *
*         CP1 -> CP2 -> CP3 -> CP4                       *
*         /  \   / \    / \    / \                       *
*         W1  S1 W2 S2  W3 S3  W4 S4                     *
**********************************************************
*
*   Original dependencies
*   W[i]->LD[i]->S[i]   i >= 0
*   LD[i]->LD[i+1]      i >= 0
*   where
*      W[i] are all predecessors of LD[i] except LD[i-1]
*      S[i] are all successors of LD[i] except of LD[i+1]
*
*   After splitting LD[i] into LD[i] and CP[i] for i > 0
*   I.e., LD0 is not split.
*
*   original dependencies
*   W[i]->LD[i]->S[i]    i >= 0
*   LD[i]->LD[i+1]       i >= 0
*
*   dependencies for CP[i]
*   CP[i]->CP[i+1]       i > 0
*   W[i]->CP[i]->S[i]    i > 0
*   LD[i]->CP[i+1]       i >= 0
*
*
*  Split Ifmap SbLoads when replicated into:
*  1. Copying of info that already exists in SBUF (from previous loads)
*  2. Loading of new info
*
*  In a series of replicated ifmap loads (not weights, replicated)
*  the first one must be loaded fully, but subsequent ones may have
*  at least some of the data in the SBUF already
*
*  Condition for replication:
*  1. Previous load replicated
*  2. This load replicated
*/

//--------------------------------------------------------
LoadSplitter::LoadSplitter(Network& network)
    : m_Network(network)
{ }



//--------------------------------------------------------
wave::SbAtomLoadWaveOp*
LoadSplitter::findPrevReplicatedLoad(wave::SbAtomLoadWaveOp* loadWaveop)
{
    for (auto prevWaveop : loadWaveop->gPrevWaveops()) {
        if (auto prevReplicatedLoad = dynamic_cast<wave::SbAtomLoadWaveOp*>(prevWaveop)) {
            return prevReplicatedLoad;
        }
    }
    return nullptr;
}

//--------------------------------------------------------
// The new copy has the following dependencies:
// 1. follows prev replicated load
// 2. follows prev copy (if it exists, L1 does not have one)
// 3. follows the same waveops as current load
// 4. precedes the same waveops as current load
wave::TpbCopyWaveOp*
LoadSplitter::splitOneReplicatedLoad(
    wave::SbAtomLoadWaveOp* prevReplicatedLoad,
    wave::SbAtomLoadWaveOp* loadWaveop)
{
    Assert(prevReplicatedLoad, "Missing prev replicated load");
    auto prevCopyWaveop = prevReplicatedLoad->gPairCopyWaveOp();

    std::vector<wave::WaveOp*> prevWaveops;
    //prevWaveops.push_back(prevReplicatedLoad); // (1. LD[i-1])
    if (prevCopyWaveop) {
        prevWaveops.push_back(prevCopyWaveop); // (2. CP[i-1])
    }
    for (auto prevWaveop : loadWaveop->gPrevWaveops()) { // (1,3.LD[i-1], W[i])
        prevWaveops.push_back(prevWaveop);
    }
    wave::TpbCopyWaveOp::Params tpbcopyParams;
    tpbcopyParams.m_WaveOpName      = loadWaveop->gName() + "_copy";
    tpbcopyParams.m_LayerName       = loadWaveop->gLayerName();
    tpbcopyParams.m_PairLoadWaveOp  = loadWaveop;
    tpbcopyParams.m_PrevCopyWaveOp  = prevCopyWaveop;
    tpbcopyParams.m_SrcSbAddress    = 0;
    tpbcopyParams.m_DstSbAddress    = 0;
    tpbcopyParams.m_SizeInBytes     = 0;

    auto newCopyWaveop = new wave::TpbCopyWaveOp(tpbcopyParams, prevWaveops);
    loadWaveop->rPairCopyWaveOp(newCopyWaveop);
    newCopyWaveop->rEngineId(loadWaveop->gEngineId());

    for (auto succWaveop : loadWaveop->gSuccWaveops()) { // (4. S[i], do not include LD[i+1])
        if (auto succLoadWaveop = dynamic_cast<wave::SbAtomLoadWaveOp*>(succWaveop)) {
            if (loadWaveop == findPrevReplicatedLoad(succLoadWaveop)) {
                continue;
            }
        }
        auto edge = new wave::WaveEdge(newCopyWaveop, succWaveop);
        newCopyWaveop->addSuccWaveEdge(edge);
        succWaveop->addPrevWaveEdge(edge);
    }
    return newCopyWaveop;
}

//--------------------------------------------------------
void
LoadSplitter::SplitReplicatedLoads()
{
    auto& waveops(m_Network.gWaveOps());
    const kcc_int32 numWaveops = waveops.size();
    std::vector<wave::WaveOp*> newWaveops;

    for (kcc_int32 widx = 0; widx < numWaveops; ++widx) {
        const auto waveop = waveops[widx];
        auto loadWaveop = dynamic_cast<wave::SbAtomLoadWaveOp*>(waveop);
        if (!loadWaveop || loadWaveop->gIfmapReplicationResolution() <= 0) {
            m_Network.replaceWaveops(newWaveops, false);
            continue;
        }
        if (loadWaveop->qContainWeights()) {
            m_Network.replaceWaveops(newWaveops, false);
            continue;
        }

        wave::SbAtomLoadWaveOp* prevReplicatedLoad = nullptr;
        prevReplicatedLoad = findPrevReplicatedLoad(loadWaveop);
        if (!prevReplicatedLoad) { // previous load not replicated
            m_Network.replaceWaveops(newWaveops, false);
            continue;
        }

        if (loadWaveop->gIfmapReplicationResolution() <= 0) { // this load not replicated
            m_Network.replaceWaveops(newWaveops, false);
            continue;
        }


        wave::TpbCopyWaveOp* copyWaveop = splitOneReplicatedLoad(
                                            prevReplicatedLoad,
                                            loadWaveop);
        // copy should go first since waveop (load) will reset address map
        // Processing of waveops during code generation should be
        // L0, C1,L1, C2,L2
        newWaveops.push_back(copyWaveop);
        newWaveops.push_back(waveop);
    }
    m_Network.replaceWaveops(newWaveops, false);
}

}}


