
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

/***************************************************************
Predecessor of Loading Weights can only be:
Another MatMul which
***************************************************************/
void
EventMgr::processMatMult(const wave::MatMulWaveOp* matmulWaveop)
{
    const EngineId engineId = matmulWaveop->gEngineId();

    int numPrevAtomFile = 0;
    int numPrevPool = 0;
    int numPrevActivation = 0;

    for (auto prevWaveop : matmulWaveop->gPrevWaveOps()) {
        if (prevWaveop->gEngineId() == engineId) {
            continue; // when two waveops execute on the same engine, no need for sync
        }

        if (auto sbatomFileWaveOp = dynamic_cast<wave::SbAtomFileWaveOp*>(prevWaveop)) {
            ++numPrevAtomFile;
            sbatomFileWaveOp = nullptr;
            continue;
        }
        if (auto poolWaveOp = dynamic_cast<wave::PoolWaveOp*>(prevWaveop)) {
            ++numPrevPool;
            poolWaveOp = nullptr;
            continue;
        }
        if (auto activationWaveOp = dynamic_cast<wave::ActivationWaveOp*>(prevWaveop)) {
            ++numPrevActivation;
            activationWaveOp = nullptr;
            continue;
        }
        Assert(false,
                "Predecessors of MatMult waveop must be one of DramLoad, Pool, Activation: ",
                prevWaveop->gTypeStr());
    }
    if (matmulWaveop->qStartTensorCalc()) {
        Assert(numPrevAtomFile + numPrevPool + numPrevActivation >= 1,
            "MatMul waveop starting tensor calc should have at least one predecessor from another engine.");
    }

    for (auto succWaveop : matmulWaveop->gSuccWaveOps()) {
        if (succWaveop->gEngineId() == engineId) {
            continue; // when two waveops execute on the same engine, no need for sync
        }
    }
}

/***************************************************************
***************************************************************/
void
EventMgr::processPool(const wave::PoolWaveOp* poolWaveop)
{
    const EngineId engineId = poolWaveop->gEngineId();

    for (auto prevWaveop : poolWaveop->gPrevWaveOps()) {
        if (prevWaveop->gEngineId() == engineId) {
            continue; // when two waveops execute on the same engine, no need for sync
        }
        if (auto pWaveOp = dynamic_cast<wave::SbAtomFileWaveOp*>(prevWaveop)) {
            pWaveOp = nullptr;
            continue;
        }
    }
}


/***************************************************************
***************************************************************/
void
EventMgr::processActivation(const wave::ActivationWaveOp* activationWaveop)
{
    const EngineId engineId = activationWaveop->gEngineId();

    for (auto prevWaveop : activationWaveop->gPrevWaveOps()) {
        if (prevWaveop->gEngineId() == engineId) {
            continue; // when two waveops execute on the same engine, no need for sync
        }
    }
}


/***************************************************************
***************************************************************/
void
EventMgr::processWaveops()
{
    for (auto waveOp : m_Network.gWaveOps()) {
        if (auto matmulWaveop = dynamic_cast<wave::MatMulWaveOp*>(waveOp)) {
            processMatMult(matmulWaveop);
            continue;
        }
        if (auto poolWaveop = dynamic_cast<wave::PoolWaveOp*>(waveOp)) {
            processPool(poolWaveop);
            continue;
        }
        if (auto activationWaveop = dynamic_cast<wave::ActivationWaveOp*>(waveOp)) {
            processActivation(activationWaveop);
            continue;
        }
    }
}



}}

