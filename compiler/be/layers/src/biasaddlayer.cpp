
#include "biasaddlayer.hpp"

namespace kcc {
namespace layers {

//----------------------------------------------------------------
// Represents input FMAP. The input FMAP is stored as the *OUTPUT*
// of the input layer.
//--------------------------------------------------------
BiasAddLayer::BiasAddLayer(const Params& param, const FmapDesc& fmap_desc,
                           const std::vector<Layer*>& prevLayers)
    : SubClass(param, fmap_desc, prevLayers)
{
    for (auto prevLayer : prevLayers) {
        if (!prevLayer->qConstLayer()) {
            continue;
        }
        assert(prevLayer->gOfmapWidth() == 1 &&
               "Const layer feeding BiasAdd must have width == 1");
        assert(prevLayer->gOfmapHeight() == 1 &&
               "Const layer feeding BiasAdd must have height == 1");
    }
}

//----------------------------------------------------------------
std::string
BiasAddLayer::gString() const
{
    std::string baseLayer = gBaseLayerStr();
    return (gName() + baseLayer + gStateSizesStr());
}


//----------------------------------------------------------------
bool
BiasAddLayer::verify() const
{
    return true;
}

}}


