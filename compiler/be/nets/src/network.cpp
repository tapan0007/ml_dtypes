#include <string>
#include <fstream>
#include <iostream>
#include <vector>
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

#include "nets/inc/network.hpp"
#include "nets/inc/network_load.hpp"
#include "nets/inc/network_save.hpp"

namespace kcc {

/*
namespace wave {
    class SbAtomFileWaveOp;
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

wave::WaveOp*
Network::findWaveOp(const std::string& waveOpName)
{
    wave::WaveOp* waveOp = m_Name2WaveOp[waveOpName];
    Assert(waveOp, "Could not find WaveOp ", waveOpName);
    return waveOp;
}


void
Network::createSuccWaveOps()
{
    for (auto succWaveop : m_WaveOps) {
        for (auto prevWaveop : succWaveop->gPrevWaveOps()) {
            prevWaveop->addSuccWaveop(succWaveop);
        }
    }
}

}}


