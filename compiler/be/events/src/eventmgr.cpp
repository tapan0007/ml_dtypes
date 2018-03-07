
#include "wave/inc/waveop.hpp"
#include "nets/inc/network.hpp"


namespace kcc {
namespace events {


EventMgr::EventMgr(const nets::Network& network)
    : m_Network(network)
{
}

void
EventMgr::processLoadWeights(const wave::MatMulWaveOp* matmulWaveop)
{
    for (auto prevWaveop : matmulWaveop->gPrevWaveOps()) {
    }
}

void
EventMgr::processMatMult(const wave::MatMulWaveOp* matmulWaveop)
{
}
    


}}

