#include <sstream>

#include "layers/inc/reshapelayer.hpp"


namespace kcc {

namespace nets {
    class Network;
}

namespace layers {

//--------------------------------------------------------
ReshapeLayer::ReshapeLayer(const ReshapeLayer::Params& params, Layer* prev_layer,
        const FmapDesc& fmapDesc)
    : Layer(params, fmapDesc, std::vector<Layer*>(1, prev_layer))
{
}


//----------------------------------------------------------------
const char*
ReshapeLayer::gTypeStr() const
{
    return TypeStr();
}
    
std::string
ReshapeLayer::gString() const 
{
    return TypeStr();
}

bool
ReshapeLayer::verify() const
{
    if (! this->Layer::verify()) {
        return false;
    }
    return true;
}


ReshapeLayer::Params::Params(const Layer::Params& params)
    : Layer::Params(params)
{
    m_LayerName     = params.m_LayerName;
    m_Network       = params.m_Network;
}


}}


