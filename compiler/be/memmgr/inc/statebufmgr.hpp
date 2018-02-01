#pragma once

#ifndef KCC_MEMMGR_STATEBUFMGR_H
#define KCC_MEMMGR_STATEBUFMGR_H 1

#include "types.hpp"
#include "arch.hpp"

using namespace kcc::utils;

namespace kcc {
namespace arch {
    class Arch;
    class StateBuffer;
}
namespace layers {
    class Layer;
}
namespace nets {
    class Network;
}



namespace memmgr {


//--------------------------------------------------------
class StateBufferMgr {
public:
    //----------------------------------------------------------------
    StateBufferMgr(const arch::Arch& arch, nets::Network* ntwk);


    //----------------------------------------------------------------
    StateBufferAddress calcOneLayerFmapMemSizePerPartition(layers::Layer* layer);

    //----------------------------------------------------------------
    void freeLayerMem(layers::Layer* layer);

    //----------------------------------------------------------------
    void calcOneLayerFmapAddresses(layers::Layer* layer);

    //----------------------------------------------------------------
    void calcLayerFmapAddresses();

private:
    nets::Network* const      m_Network;
    const arch::Arch&         m_Arch;
    const arch::StateBuffer&  m_StateBuffer;

    StateBufferAddress  m_OfmapAddress;
    StateBufferAddress  m_IfmapAddress;
    StateBufferAddress  m_FirstSbAddress;
    StateBufferAddress  m_FirstFreeStart;
    StateBufferAddress  m_MaxNumberWeightsPerPart;
    StateBufferAddress  m_PartitionSize;
};


}}

#endif // KCC_MEMMGR_STATEBUFMGR_H

