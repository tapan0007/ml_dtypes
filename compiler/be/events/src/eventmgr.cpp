
#include "utils/inc/asserter.hpp"

#include "wave/inc/waveop.hpp"
#include "wave/inc/matmulwaveop.hpp"
#include "wave/inc/sbatomfilewaveop.hpp"
#include "wave/inc/sbatomsavewaveop.hpp"
#include "wave/inc/activationwaveop.hpp"
#include "wave/inc/poolwaveop.hpp"

#include "nets/inc/network.hpp"

#include "events/inc/eventmgr.hpp"


namespace kcc {
namespace events {


EventMgr::EventMgr(const nets::Network& network)
    : m_Network(network)
{
}

// Predecessor of Loading Weights can only be:
// Another MatMul which 
void
EventMgr::processMatMult(const wave::MatMulWaveOp* matmulWaveop)
{
    const EngineId engineId = matmulWaveop->gEngineId();

    for (auto prevWaveop : matmulWaveop->gPrevWaveOps()) {
        if (prevWaveop->gEngineId() == engineId) {
            continue; // when two waveops execute on the same engine, no need for sync
        }

        if (auto pWaveOp = dynamic_cast<wave::SbAtomFileWaveOp*>(prevWaveop)) {
            pWaveOp = nullptr;
            continue;
        }
        if (auto pWaveOp = dynamic_cast<wave::PoolWaveOp*>(prevWaveop)) {
            pWaveOp = nullptr;
            continue;
        }
        if (auto pWaveOp = dynamic_cast<wave::ActivationWaveOp*>(prevWaveop)) {
            pWaveOp = nullptr;
            continue;
        }
        Assert(false,
                "Predecessors of MatMult waveop must be one of DramLoad, Pool, Activation: ",
                prevWaveop->gTypeStr());
    }
}


void
EventMgr::processWaveops()
{
    for (auto waveOp : m_Network.gWaveOps()) {
        if (auto matmulWaveop = dynamic_cast<wave::MatMulWaveOp*>(waveOp)) {
            processMatMult(matmulWaveop);
            continue;
        }
    }
}
    


}}

