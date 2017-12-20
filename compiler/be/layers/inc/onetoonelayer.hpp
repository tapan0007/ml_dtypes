#pragma once

#ifndef KCC_LAYERS_ONETOONELAYER_H
#define KCC_LAYERS_ONETOONELAYER_H

#include "types.hpp"
#include "datatype.hpp"
#include "fmapdesc.hpp"
#include "layer.hpp"


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
    typedef Layer SubClass;
public:
    //----------------------------------------------------------------
    OneToOneLayer(const Params& params, Layer* prev_layer)
        : Layer(params,
            prev_layer->gOfmapDesc(), prev_layer->gDataTensorDimSemantics(),
            vector<Layer*>(1, prev_layer))
    { }

    //----------------------------------------------------------------
    bool verify() const override {
        assert(1 == gNumPrevLayers());
        const Layer* prev_layer = gPrevLayer(0);
        assert(prev_layer->gOfmapDesc() == gOfmapDesc());
        return true;
    }

    bool qOneToOneLayer() const override {
        return true;
    }
};

} // namespace layers
} // namespace kcc

#endif // KCC_LAYERS_ONETOONELAYER_H

