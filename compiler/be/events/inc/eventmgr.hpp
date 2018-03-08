#pragma once

#ifndef KCC_EVENTS_EVENTMGR_H
#define KCC_EVENTS_EVENTMGR_H

namespace kcc {
namespace wave {
class MatMulWaveOp;
}


namespace events {

class EventMgr {
public:
    EventMgr(const nets::Network& network);
    void processWaveops();

private:
    void processMatMult(const wave::MatMulWaveOp* matmulWaveop);
    void processPool(const wave::PoolWaveOp* poolWaveop);
    void processActivation(const wave::ActivationWaveOp* activationWaveop);

private:
    const nets::Network& m_Network;
};

}}

#endif // KCC_EVENTS_EVENTMGR_H

