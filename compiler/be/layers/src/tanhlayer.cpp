#include <sstream>

#include "datatype.hpp"
#include "network.hpp"
#include "tanhlayer.hpp"



namespace kcc {
namespace layers {

//--------------------------------------------------------
TanhLayer::TanhLayer(const Params& params, Layer* prev_layer)
    : ActivLayer(params, prev_layer)
{}


//--------------------------------------------------------
std::string
TanhLayer::gString() const
{
    std::string baseLayerStr = gBaseLayerStr();
    std::string retVal;
    retVal = retVal + gTypeStr() + " " + baseLayerStr + gStateSizesStr();
    return retVal;
}


//--------------------------------------------------------
const char*
TanhLayer::gTypeStr() const
{
    return "Tanh";
}

//--------------------------------------------------------
bool
TanhLayer::qPassThrough() const
{
    return false;
}

//--------------------------------------------------------
bool
TanhLayer::qTanhLayer() const
{
    return true;
}

} // namespace layers
} // namespace kcc


