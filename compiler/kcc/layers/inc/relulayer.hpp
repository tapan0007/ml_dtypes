#pragma once

#ifndef KCC_LAYERS_RELULAYER_H
#define KCC_LAYERS_RELULAYER_H

#include "types.hpp"
#include "datatype.hpp"
#include "fmapdesc.hpp"
#include "activlayer.hpp"


namespace kcc {

namespace nets {
    class Network;
}

namespace layers {

using nets::Network;
using namespace utils;

//--------------------------------------------------------
class ReluLayer : public ActivLayer {
public:
    //----------------------------------------------------------------
    ReluLayer(const Params& params, Layer* prev_layer);

#if 0
    //----------------------------------------------------------------
    gJson();

    //----------------------------------------------------------------
    static ReluLayer* constructFromJson(cls, layerDict, nn);
#endif


    //----------------------------------------------------------------
    string gString() const override;

    //----------------------------------------------------------------
    static const char* gTypeStrStatic();
    const char* gTypeStr() const override;

    //----------------------------------------------------------------
    bool qPassThrough() const override;

    //----------------------------------------------------------------
    bool qReluLayer() const override;
};

} // namespace layers
} // namespace kcc

#endif // KCC_LAYERS_RELULAYER_H

