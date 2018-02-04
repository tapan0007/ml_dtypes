#include "layers/inc/combinelayer.hpp"


namespace kcc {

namespace nets {
    class Network;
}

namespace layers {

//--------------------------------------------------------
CombineLayer::CombineLayer(const Params& params, Layer* prev_layer, Layer* earlier_layer, kcc_int32 num_ofmaps)
    : Layer(params,
            FmapDesc(num_ofmaps, prev_layer->gOfmapHeight(), prev_layer->gOfmapWidth()),
            mkLayerVector2(prev_layer, earlier_layer))

{
    assert(prev_layer->gOfmapWidth() == earlier_layer->gOfmapWidth() && "Combine layer: Input image widths not identical");
    assert(prev_layer->gOfmapHeight() == earlier_layer->gOfmapHeight() && "Combine layer: Input image heights not identical");
}

//----------------------------------------------------------------
bool
CombineLayer::qPassThrough() const
{
    return false;
}

//----------------------------------------------------------------
bool
CombineLayer::qCombineLayer() const
{
    return true;
}

} // namespace layers
} // namespace kcc

