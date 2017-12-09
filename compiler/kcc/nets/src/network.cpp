

#include "layer.hpp"
#include "network.hpp"

namespace kcc {

namespace nets {

Network::Network(const DataType* dataType, const string& netName)
    : m_DataType(dataType)
    , m_Name(netName)
{
}

void
Network::addLayer(Layer* layer)
{
    m_Layers.push_back(layer);
}



//--------------------------------------------------------
void 
Network::SchedLayerForwRevIter::operator++()
{
    Layer* const currLayer = m_CurrLayer;
    assert(currLayer);
    Layer* nextLayer;

    if (m_Forw) {
        nextLayer = currLayer->gNextSchedLayer();
    } else {
        nextLayer = currLayer->gPrevSchedLayer();
    }

    m_CurrLayer = nextLayer;
}

//--------------------------------------------------------
Network::SchedForwLayers
Network::gSchedForwLayers()
{
    return SchedForwLayers(m_Layers);
}

//--------------------------------------------------------
Network::SchedRevLayers
Network::gSchedRevLayers()
{
    return SchedRevLayers(m_Layers);
}

}}


