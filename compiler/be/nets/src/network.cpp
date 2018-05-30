#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <array>
#include <map>


#include "utils/inc/asserter.hpp"
#include "arch/inc/arch.hpp"

#include "layers/inc/layer.hpp"
#include "layers/inc/inputlayer.hpp"
#include "layers/inc/constlayer.hpp"
#include "layers/inc/convlayer.hpp"
#include "layers/inc/relulayer.hpp"
#include "layers/inc/tanhlayer.hpp"
#include "layers/inc/maxpoollayer.hpp"
#include "layers/inc/avgpoollayer.hpp"
#include "layers/inc/resaddlayer.hpp"
#include "layers/inc/biasaddlayer.hpp"

#include "wave/inc/sbatomloadwaveop.hpp"
#include "wave/inc/sbatomsavewaveop.hpp"

#include "nets/inc/network.hpp"
#include "nets/inc/network_load.hpp"
#include "nets/inc/network_save.hpp"

namespace kcc {

/*
namespace wave {
    class SbAtomLoadWaveOp;
    class SbAtomSaveWaveOp;
    class MatMulWaveOp;
}
*/

namespace nets {

//--------------------------------------------------------
Network::Network(const arch::Arch& arch)
    : m_Arch(arch)
    , m_DataType(nullptr)
    , m_Name()
    , m_DoBatching(false)
    , m_Load(std::make_unique<Load>(*this))
    , m_Save(std::make_unique<Save>(*this))
{}


Network::~Network() = default;

//--------------------------------------------------------
void
Network::addLayer(layers::Layer* layer)
{
    m_Layers.push_back(layer);
}



//--------------------------------------------------------
void
Network::SchedLayerForwRevIter::operator++()
{
    layers::Layer* const currLayer = m_CurrLayer;
    Assert(currLayer, "Layer iterator in Network: Invalid current layer");
    layers::Layer* nextLayer;

    if (m_Forw) {
        nextLayer = currLayer->gNextSchedLayer();
    } else {
        nextLayer = currLayer->gPrevSchedLayer();
    }

    m_CurrLayer = nextLayer;
}

//--------------------------------------------------------
Network::SchedForwLayers
Network::gSchedForwLayers() const
{
    return SchedForwLayers(m_Layers);
}

//--------------------------------------------------------
Network::SchedRevLayers
Network::gSchedRevLayers()
{
    return SchedRevLayers(m_Layers);
}

//--------------------------------------------------------
layers::Layer*
Network::findLayer(const std::string& layerName)
{
    layers::Layer* layer = m_Name2Layer[layerName];
    Assert(layer, "Could not find layer ", layerName);
    return layer;
}

//--------------------------------------------------------
wave::WaveOp*
Network::findWaveOp(const std::string& waveOpName)
{
    wave::WaveOp* waveOp = m_Name2WaveOp[waveOpName];
    Assert(waveOp, "Could not find WaveOp ", waveOpName);
    return waveOp;
}

//--------------------------------------------------------
void
Network::revertSavedWaveops()
{
    std::swap(m_WaveOps, m_SaveWaveOps);
}

//--------------------------------------------------------
void
Network::replaceWaveops(std::vector<wave::WaveOp*>& newWaveops)
{
    const kcc_int32 numWaveops = newWaveops.size();
    for (kcc_int32 k = 0; k < numWaveops; ++k) {
        newWaveops[k]->rOrder(k);
    }
    //m_WaveOps.clear();
    std::swap(m_WaveOps, m_SaveWaveOps);
    std::swap(m_WaveOps, newWaveops);
    //std::copy(newWaveops.begin(), newWaveops.end(), m_WaveOps.begin());
}

//--------------------------------------------------------
const std::string&
Network::gInTensorFormat() const
{
    static const std::string emptyStr;

    for (auto waveop : m_WaveOps) {
        const auto sbLoadWaveop =
                dynamic_cast<const wave::SbAtomLoadWaveOp*>(waveop);
        if (sbLoadWaveop && ! sbLoadWaveop->qContainWeights()) {
            return sbLoadWaveop->gRefFileFormat();
        }
    }
    Assert(false, "Network::gTensorFormat: did not find IFMAP Load waveop");
    return emptyStr;
}

//--------------------------------------------------------
const std::array<kcc_int32, 4>&
Network::gInTensorDimensions() const
{
    static const std::array<kcc_int32, 4> badDim = {{ -1, -1, -1, -1 }};

    for (auto waveop : m_WaveOps) {
        const auto sbLoadWaveop =
                dynamic_cast<const wave::SbAtomLoadWaveOp*>(waveop);
        if (sbLoadWaveop && ! sbLoadWaveop->qContainWeights()) {
            return sbLoadWaveop->gRefFileShape();
        }
    }
    Assert(false, "Network::gTensorDimensions: did not find IFMAP Load waveop");
    return badDim;
}

//--------------------------------------------------------
kcc_int32
Network::gInLayerStride() const
{
    for (auto waveop : m_WaveOps) {
        const auto sbLoadWaveop =
                dynamic_cast<const wave::SbAtomLoadWaveOp*>(waveop);
        if (sbLoadWaveop && ! sbLoadWaveop->qContainWeights()) {
            return sbLoadWaveop->gSrcStepElem();
        }
    }
    Assert(false, "Network::gInLayerStride: did not find IFMAP Load waveop");
    return -1;
}

//--------------------------------------------------------
kcc_int32
Network::gInDataSizeInBytes() const
{
    kcc_int64 inSizeInBytes = gDataType().gSizeInBytes();
    const auto& refShape(gInTensorDimensions());

    for (auto n : refShape) {
        inSizeInBytes *= n;
    }
    return inSizeInBytes;
}

//--------------------------------------------------------
kcc_int32
Network::gOutDataSizeInBytes() const
{
    for (auto waveop : m_WaveOps) {
        const auto sbSaveWaveop =
                dynamic_cast<const wave::SbAtomSaveWaveOp*>(waveop);
        if (sbSaveWaveop) {
            kcc_int64 outSizeInBytes = gDataType().gSizeInBytes();
            for (auto n : sbSaveWaveop->gRefFileShape()) {
                outSizeInBytes *= n;
            }
            return outSizeInBytes;
        }
    }
    Assert(false, "Network::gOutDataSizeInBytes: did not find IFMAP Load waveop");
    return -1;
}

}}


