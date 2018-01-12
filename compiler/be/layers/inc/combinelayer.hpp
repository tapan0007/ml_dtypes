#pragma once

#ifndef KCC_LAYERS_COMBINELAYER_H
#define KCC_LAYERS_COMBINELAYER_H

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
// Base layer for all layer that combine multiple earlier layer
// (multi-input layer). Examples: ResNet's residue add, DenseNet's concatenation
//--------------------------------------------------------
class CombineLayer : public Layer {
private:
    using SubClass = Layer;
public:
    //----------------------------------------------------------------
    CombineLayer(const Params& param, Layer* prev_layer, Layer* earlier_layer, kcc_int32 num_ofmaps);

    //----------------------------------------------------------------
    bool qPassThrough() const override;

    //----------------------------------------------------------------
    bool qCombineLayer() const override;
};

} // namespace layers
} // namespace kcc

#endif // KCC_LAYERS_COMBINELAYER_H

