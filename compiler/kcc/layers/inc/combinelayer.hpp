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
class CombineLayer : public Layer {
private:
    typedef Layer SubClass;
public:
    //----------------------------------------------------------------
    CombineLayer(const Params& param, Layer* prev_layer, Layer* earlier_layer, int num_ofmaps);

    //----------------------------------------------------------------
    bool qPassThrough() const override;

    //----------------------------------------------------------------
    bool qCombineLayer() const override;
};

} // namespace layers
} // namespace kcc

#endif // KCC_LAYERS_COMBINELAYER_H

