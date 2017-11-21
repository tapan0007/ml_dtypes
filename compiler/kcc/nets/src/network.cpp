

#include "network.h"

namespace kcc {

namespace nets {

Network::Network(const DataType& dataType, const string& netName)
    : m_DataType(dataType)
    , m_Name(netName)
{
}

void
Network::addLayer(Layer* layer)
{
    m_Layers.push_back(layer);
}

}}

