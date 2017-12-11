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

#if 0
//--------------------------------------------------------
TanhLayer::gJson()
{
    x = super().gJson()
    return x
}

//--------------------------------------------------------
TanhLayer* 
TanhLayer::constructFromJson(cls, layerDict, nn)
{
        layerName = Layer.gLayerNameFromJson(layerDict)
        ofmapDesc = Layer.gOfmapDescFromJson(layerDict, nn)
        batch = 1
        param = Layer.Param(layerName, batch, nn)
        prevLayers = Layer.gPrevLayersFromJson(layerDict, nn)
        assert isinstance(prevLayers, list) and len(prevLayers)==1
        layer = TanhLayer(param, prevLayers[0])
        return layer
}
#endif


//--------------------------------------------------------
string
TanhLayer::gString() const
{
    string baseLayerStr = gBaseLayerStr();
    string retVal;
    retVal = retVal + gTypeStr() + " " + baseLayerStr + gStateSizesStr();
    return retVal;
}


const char*
TanhLayer::gTypeStr() const
{
    return "Tanh";
}

bool
TanhLayer::qPassThrough() const
{
    return false;
}

bool 
TanhLayer::qTanhLayer() const
{
    return true;
}

} // namespace layers
} // namespace kcc


