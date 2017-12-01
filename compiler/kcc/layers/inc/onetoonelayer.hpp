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
class OneToOneLayer : public Layer {
public:
    //----------------------------------------------------------------
    OneToOneLayer(const Params& params, Layer* prev_layer)
        : Layer(params, prev_layer->gOfmapDesc(), vector<Layer*>(1, prev_layer))
    { }

    //----------------------------------------------------------------
    bool verify() const {
        assert(1 == gNumPrevLayers());
        const Layer* prev_layer = gPrevLayer(0);
        assert(prev_layer->gOfmapDesc() == gOfmapDesc());
        return true;
    }

    bool qOneToOneLayer() const {
        return true;
    }
};

} // namespace layers
} // namespace kcc

#endif // KCC_LAYERS_ONETOONELAYER_H

