#pragma once

#ifndef KCC_EVENTS_EVENTMGR_H
#define KCC_EVENTS_EVENTMGR_H

namespace kcc {
//enum class EventWaitMode;
//enum class EventSetMode;

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
    void processPool(wave::PoolWaveOp* poolWaveop);
    void processActivation(wave::ActivationWaveOp* activationWaveop);
    void processSbAtomSave(wave::SbAtomSaveWaveOp* sbatomSaveWaveop);
    void processSbAtomLoad(wave::SbAtomLoadWaveOp* sbatomLoadWaveop);
    void processResAdd(wave::ResAddWaveOp* resaddWaveop);

    EventId getLocalEventId(const wave::WaveEdge* edge);

private:
    const nets::Network& m_Network;
    EventId m_EventId;
};

}}

#endif // KCC_EVENTS_EVENTMGR_H

