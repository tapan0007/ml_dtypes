#pragma once

#ifndef KCC_EVENTS_EVENTMGR_H
#define KCC_EVENTS_EVENTMGR_H

namespace kcc {
//enum class EventWaitMode;
//enum class EventSetMode;

namespace wave {
class MatMulWaveOp;
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

    kcc_int32 getLocalEventId(const wave::WaveOp* from, const wave::WaveOp* to);

private:
    const nets::Network& m_Network;
    kcc_int32 m_EventId;
};

}}

#endif // KCC_EVENTS_EVENTMGR_H

