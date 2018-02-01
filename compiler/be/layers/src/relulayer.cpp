#include <sstream>

#include "datatype.hpp"
#include "network.hpp"
#include "relulayer.hpp"



namespace kcc {
namespace layers {

//--------------------------------------------------------
ReluLayer::ReluLayer(const Params& params, Layer* prev_layer)
    : ActivLayer(params, prev_layer)
{}



//--------------------------------------------------------
std::string
ReluLayer::gString() const
{
    std::string baseLayerStr = gBaseLayerStr();
    std::string retVal;
    retVal = retVal + gTypeStr() + " " + baseLayerStr + gStateSizesStr();
    return retVal;
}


//--------------------------------------------------------
const char*
ReluLayer::gTypeStrStatic()
{
    return "Relu";
}

//--------------------------------------------------------
const char*
ReluLayer::gTypeStr() const
{
    return gTypeStrStatic();
}

//--------------------------------------------------------
bool
ReluLayer::qPassThrough() const
{
    return false;
}

//--------------------------------------------------------
bool
ReluLayer::qReluLayer() const
{
    return true;
}

} // namespace layers
} // namespace kcc


