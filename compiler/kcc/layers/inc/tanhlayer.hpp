#pragma once

#ifndef KCC_LAYERS_TANHLAYER_H
#define KCC_LAYERS_TANHLAYER_H

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
class TanhLayer : public ActivLayer {
public:
    //----------------------------------------------------------------
    TanhLayer(const Params& params, Layer* prev_layer);

#if 0
    //----------------------------------------------------------------
    gJson();

    //----------------------------------------------------------------
    static TanhLayer* constructFromJson(cls, layerDict, nn);
#endif


    //----------------------------------------------------------------
    string gString() const;

    //----------------------------------------------------------------
    const char* gTypeStr() const;

    //----------------------------------------------------------------
    bool qPassThrough() const override;

    //----------------------------------------------------------------
    bool qTanhLayer() const override;
};

} // namespace layers
} // namespace kcc

#endif // KCC_LAYERS_TANHLAYER_H

