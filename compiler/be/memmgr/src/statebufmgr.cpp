#include "statebuffer.hpp"
#include "layer.hpp"
#include "network.hpp"
#include "statebufmgr.hpp"

namespace kcc{
using layers::Layer;

namespace memmgr {

//--------------------------------------------------------
StateBufferMgr::StateBufferMgr(arch::Arch* arch, Network* ntwk)
    : m_Network(ntwk)
    , m_Arch(arch)
    , m_StateBuffer(arch->gStateBuffer())
{

    m_PartitionSize = m_StateBuffer->gPartitionSizeInBytes();
    m_FirstSbAddress = m_StateBuffer->gFirstAddressInBytes();

    m_FirstFreeStart = m_FirstSbAddress;
}

//--------------------------------------------------------
void
StateBufferMgr::freeLayerMem(Layer* layer)
{
    assert(layer->qStoreInSB());
}

//--------------------------------------------------------
StateBufferAddress
StateBufferMgr::calcOneLayerFmapMemSizePerPartition(layers::Layer* layer)
{
    const StateBufferAddress outSbMemBatch  = layer->gOutputStateMemWithBatching();
    const StateBufferAddress resSbMemBatch  = layer->gResMemWithBatching();
    const StateBufferAddress totSbMemBatch  = outSbMemBatch + resSbMemBatch;
    const kcc_int32 numOfmaps      = layer->gNumOfmaps();
    assert(numOfmaps > 0);
    const StateBufferAddress numPeArrayRows = m_Arch->gNumberPeArrayRows();

    const StateBufferAddress sbMemPerOfmap  = totSbMemBatch / numOfmaps;
    const StateBufferAddress maxNumOfmapsPerRow = 1 + ((numOfmaps - 1) / numPeArrayRows);

    const StateBufferAddress ofmapMemPerPart = sbMemPerOfmap * maxNumOfmapsPerRow;
    return ofmapMemPerPart;
}



//--------------------------------------------------------
void
StateBufferMgr::calcOneLayerFmapAddresses(layers::Layer* layer)
{
    if (layer->qStoreInSB()) {
        for (auto prevSbLayer : layer->gPrevSbLayers()) {
            prevSbLayer->changeRefCount(-1);
            if (prevSbLayer->gRefCount() == 0) {
                freeLayerMem(prevSbLayer);
            }
        }

        assert(layer->gRefCount() == 0);
        layer->changeRefCount(layer->gNumNextSbLayers());

        StateBufferAddress ifmapAddress = StateBufferAddress_Invalid;
        StateBufferAddress ofmapAddress = StateBufferAddress_Invalid;
        const StateBufferAddress prevOfmapAddress = m_OfmapAddress;
        const StateBufferAddress prevIfmapAddress = m_IfmapAddress;

        if (layer->qInputLayer()) {
            assert(prevIfmapAddress == StateBufferAddress_Invalid &&
                   prevOfmapAddress == StateBufferAddress_Invalid);
            ifmapAddress = StateBufferAddress_Invalid;
            ofmapAddress = m_FirstSbAddress +
                           (layer->gDataType().gSizeInBytes() * m_MaxNumberWeightsPerPart);
        } else {
            assert(prevOfmapAddress != StateBufferAddress_Invalid);
            ifmapAddress = prevOfmapAddress;

            const StateBufferAddress ofmapSizePerPart = calcOneLayerFmapMemSizePerPartition(layer);

            if (prevIfmapAddress == StateBufferAddress_Invalid) {
                //         Weights | prevOfmap | ... | ...
                // need to get batching memory per partition
                ofmapAddress = m_FirstSbAddress + m_PartitionSize - ofmapSizePerPart;
            } else if (prevIfmapAddress < prevOfmapAddress) {
                //     Weights | prevIfmap | ... | prevOfmap
                ofmapAddress = m_FirstSbAddress +
                               (layer->gDataType().gSizeInBytes() * m_MaxNumberWeightsPerPart);
            } else {
                //     Weights | prevOfmap | ... | prevIfmap
                //                             | Ofmap
                ofmapAddress = m_FirstSbAddress + m_PartitionSize - ofmapSizePerPart;
            }
        }

        layer->rIfmapAddress(ifmapAddress);
        layer->rOfmapAddress(ofmapAddress);
        m_OfmapAddress = ofmapAddress;
        m_IfmapAddress = ifmapAddress;
    }

    if (layer->qConvLayer()) {
        layer->rWeightAddress(m_FirstSbAddress);
    }
}


//--------------------------------------------------------
void
StateBufferMgr::calcLayerFmapAddresses()
{
    for (auto layer : m_Network->gLayers()) {
        layer->changeRefCount(-layer->gRefCount());
        assert(layer->gRefCount() == 0);
    }

    StateBufferAddress maxNumWeightsPerPart = 0;
    for (auto layer : m_Network->gLayers()) {
        const StateBufferAddress numWeights = layer->gNumberWeightsPerPartition();
        if (numWeights > maxNumWeightsPerPart) {
            maxNumWeightsPerPart = numWeights;
        }
    }

    m_MaxNumberWeightsPerPart = maxNumWeightsPerPart;


    m_OfmapAddress = m_IfmapAddress = StateBufferAddress_Invalid;

    for (auto layer : m_Network->gLayers()) {
        calcOneLayerFmapAddresses(layer);
    }
}

}}

