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

#if 0
//--------------------------------------------------------
ReluLayer::gJson()
{
    x = super().gJson()
    return x
}

//--------------------------------------------------------
ReluLayer* 
ReluLayer::constructFromJson(cls, layerDict, nn)
{
        layerName = Layer.gLayerNameFromJson(layerDict)
        ofmapDesc = Layer.gOfmapDescFromJson(layerDict, nn)
        batch = 1
        param = Layer.Param(layerName, batch, nn)
        prevLayers = Layer.gPrevLayersFromJson(layerDict, nn)
        assert isinstance(prevLayers, list) and len(prevLayers)==1
        layer = ReluLayer(param, prevLayers[0])
        return layer
}
#endif


//--------------------------------------------------------
string
ReluLayer::gString() const
{
    string baseLayerStr = gBaseLayerStr();
    string retVal;
    retVal = retVal + gTypeStr() + " " + baseLayerStr + gStateSizesStr();
    return retVal;
}


const char*
ReluLayer::gTypeStrStatic()
{
    return "Relu";
}

const char*
ReluLayer::gTypeStr() const
{
    return gTypeStrStatic();
}

bool
ReluLayer::qPassThrough() const
{
    return false;
}

bool 
ReluLayer::qReluLayer() const
{
    return true;
}

} // namespace layers
} // namespace kcc


