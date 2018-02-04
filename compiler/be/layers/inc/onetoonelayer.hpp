#pragma once

#ifndef KCC_LAYERS_ONETOONELAYER_H
#define KCC_LAYERS_ONETOONELAYER_H

#include "utils/inc/types.hpp"
#include "utils/inc/datatype.hpp"
#include "utils/inc/fmapdesc.hpp"
#include "layers/inc/layer.hpp"


namespace kcc {

namespace nets {
    class Network;
}

namespace layers {

//--------------------------------------------------------
// Base class of layers that have the same OFMAP tensor shaper
// as the IFMAP tensor shape. Example: Activations.
//--------------------------------------------------------
class OneToOneLayer : public Layer {
private:
    using SubClass = Layer;
public:
    //----------------------------------------------------------------
    OneToOneLayer(const Params& params, Layer* prev_layer)
        : SubClass(params, prev_layer->gOfmapDesc(),
            std::vector<Layer*>(1, prev_layer))
    { }

    //----------------------------------------------------------------
    bool verify() const override {
        assert(1 == gNumPrevLayers() && "1-1 layer: Number of previous layers not 1");
        const Layer* prev_layer = gPrevLayer(0);
        assert(prev_layer->gOfmapDesc() == gOfmapDesc() && "1-1 layer: Input and output OFMAP descs are different");
        return true;
    }

    bool qOneToOneLayer() const override {
        return true;
    }
};

} // namespace layers
} // namespace kcc

#endif // KCC_LAYERS_ONETOONELAYER_H

