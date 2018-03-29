#pragma once

#include "shared/inc/uarch_cfg.hpp"

#ifndef KCC_EVENTS_EVENTMGR_H
#define KCC_EVENTS_EVENTMGR_H

namespace kcc {

namespace wave {
class MatMulWaveOp;
class PoolWaveOp;
class ActivationWaveOp;
class SbAtomLoadWaveOp;
class SbAtomSaveWaveOp;
class ResAddWaveOp;
}


namespace events {

class EventMgr {
public:
    EventMgr(const nets::Network& network);
    void processWaveops();

private:
    void processMatMult(wave::MatMulWaveOp* matmulWaveop);
#if 0
    void processPool(wave::PoolWaveOp* poolWaveop);
    void processActivation(wave::ActivationWaveOp* activationWaveop);
    void processSbAtomSave(wave::SbAtomSaveWaveOp* sbatomSaveWaveop);
    void processSbAtomLoad(wave::SbAtomLoadWaveOp* sbatomLoadWaveop);
    void processResAdd(wave::ResAddWaveOp* resaddWaveop);
#endif
    void processWaveop(wave::WaveOp* waveop);

    EventId getLocalEventId(const wave::WaveEdge* edge);

    

private:
    const nets::Network& m_Network;
    EventId m_EventId;
};

}}

#endif // KCC_EVENTS_EVENTMGR_H

