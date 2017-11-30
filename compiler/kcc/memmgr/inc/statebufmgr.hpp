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
using kcc::arch::Arch;
using kcc::arch::StateBuffer;
using kcc::layers::Layer;
using kcc::nets::Network;


//--------------------------------------------------------
class StateBufferMgr {
private:
    const StateBufferAddress InvalidFmapAddress = -1L;

public:
    //----------------------------------------------------------------
    StateBufferMgr(Arch* arch, Network* ntwk);


    //----------------------------------------------------------------
    StateBufferAddress calcOneLayerFmapMemSizePerPartition(Layer* layer);

    //----------------------------------------------------------------
    void freeLayerMem(Layer* layer);

    //----------------------------------------------------------------
    void calcOneLayerFmapAddresses(Layer* layer);

    //----------------------------------------------------------------
    void calcLayerFmapAddresses();

private:
    Network* m_Network;
    Arch* m_Arch;
    StateBuffer* m_StateBuffer;

    StateBufferAddress m_OfmapAddress;
    StateBufferAddress m_IfmapAddress;
    StateBufferAddress m_FirstSbAddress;
    StateBufferAddress m_FirstFreeStart;
    StateBufferAddress m_MaxNumberWeightsPerPart;
    StateBufferAddress m_PartitionSize;
};


}}

#endif // KCC_MEMMGR_STATEBUFMGR_H

