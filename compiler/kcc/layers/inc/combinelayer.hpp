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
    //----------------------------------------------------------------
    CombineLayer(const Params& param, Layer* prev_layer, Layer* earlier_layer, int num_ofmaps);

#if 0
    //----------------------------------------------------------------
    def gJson(self):
        x = super().gJson()
        return x
#endif

    //----------------------------------------------------------------
    bool qPassThrough() const;

    //----------------------------------------------------------------
    bool qCombineLayer() const;
};

} // namespace layers
} // namespace kcc

#endif // KCC_LAYERS_COMBINELAYER_H

