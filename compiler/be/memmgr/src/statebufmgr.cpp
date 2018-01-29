#include "statebuffer.hpp"
#include "convlayer.hpp"
#include "constlayer.hpp"
#include "network.hpp"
#include "statebufmgr.hpp"

namespace kcc{

namespace memmgr {

//--------------------------------------------------------
StateBufferMgr::StateBufferMgr(const arch::Arch& arch, nets::Network* ntwk)
    : m_Network(ntwk)
    , m_Arch(arch)
    , m_StateBuffer(arch.gStateBuffer())
{

    m_PartitionSize = m_StateBuffer.gPartitionSizeInBytes();
    m_FirstSbAddress = m_StateBuffer.gFirstAddressInBytes();

    m_FirstFreeStart = m_FirstSbAddress;
}

//--------------------------------------------------------
void
StateBufferMgr::freeLayerMem(layers::Layer* layer)
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
    assert(numOfmaps > 0 && "Layer has no Ofmaps");
    const StateBufferAddress numPeArrayRows = m_Arch.gNumberPeArrayRows();

    const StateBufferAddress sbMemPerOfmap  = totSbMemBatch / numOfmaps;
    const StateBufferAddress maxNumOfmapsPerRow = 1 + ((numOfmaps - 1) / numPeArrayRows);

    const StateBufferAddress ofmapMemPerPart = sbMemPerOfmap * maxNumOfmapsPerRow;
    return ofmapMemPerPart;
}



//--------------------------------------------------------
void
StateBufferMgr::calcOneLayerFmapAddresses(layers::Layer* layer)
{
    if (layer->qConstLayer()) {
        const auto const_layer = dynamic_cast<layers::ConstLayer*>(layer);
        const_layer->rOfmapAddress(m_FirstSbAddress);
        return;
    }

    if (layer->qStoreInSB()) {
        for (auto prevSbLayer : layer->gPrevSbLayers()) {
            prevSbLayer->changeRefCount(-1);
            if (prevSbLayer->gRefCount() == 0) {
                freeLayerMem(prevSbLayer);
            }
        }

        assert(layer->gRefCount() == 0 && "New layer should start with zero back reference count" );
        layer->changeRefCount(layer->gNumNextSbLayers());

        StateBufferAddress ifmapAddress = StateBufferAddress_Invalid;
        StateBufferAddress ofmapAddress = StateBufferAddress_Invalid;
        const StateBufferAddress prevOfmapAddress = m_OfmapAddress;
        const StateBufferAddress prevIfmapAddress = m_IfmapAddress;

        if (layer->qInputLayer()) {
            assert(prevIfmapAddress == StateBufferAddress_Invalid &&
                   prevOfmapAddress == StateBufferAddress_Invalid &&
                   "Input layer should start with invalid FMAP addresses");
            ifmapAddress = StateBufferAddress_Invalid;
            ofmapAddress = m_FirstSbAddress +
                           (layer->gDataType().gSizeInBytes() * m_MaxNumberWeightsPerPart);
        } else {
            assert(prevOfmapAddress != StateBufferAddress_Invalid &&
                    "Non-input layers should start valid IFMAP address");
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
        const auto conv_layer = dynamic_cast<layers::ConvLayer*>(layer);
        conv_layer->rWeightAddress(m_FirstSbAddress);
    }
}


//--------------------------------------------------------
void
StateBufferMgr::calcLayerFmapAddresses()
{
    for (auto layer : m_Network->gLayers()) {
        layer->changeRefCount(-layer->gRefCount());
        assert(layer->gRefCount() == 0 && "Layer should have zero reference count");
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

