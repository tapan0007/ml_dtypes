#include <sstream>

#include "utils/inc/datatype.hpp"
#include "nets/inc/network.hpp"
#include "layers/inc/sqrtlayer.hpp"



namespace kcc {
namespace layers {

//--------------------------------------------------------
SqrtLayer::SqrtLayer(const Params& params, Layer* prev_layer)
    : ActivLayer(params, prev_layer)
{}


//--------------------------------------------------------
std::string
SqrtLayer::gString() const
{
    std::string baseLayerStr = gBaseLayerStr();
    std::string retVal;
    retVal = retVal + gTypeStr() + " " + baseLayerStr + gStateSizesStr();
    return retVal;
}


//--------------------------------------------------------
const char*
SqrtLayer::gTypeStr() const
{
    return "Sqrt";
}

//--------------------------------------------------------
bool
SqrtLayer::qPassThrough() const
{
    return false;
}

//--------------------------------------------------------
bool
SqrtLayer::qSqrtLayer() const
{
    return true;
}

} // namespace layers
} // namespace kcc


