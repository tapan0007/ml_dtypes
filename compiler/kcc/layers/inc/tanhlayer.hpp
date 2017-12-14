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
private:
    typedef ActivLayer SubClass;
public:
    //----------------------------------------------------------------
    TanhLayer(const Params& params, Layer* prev_layer);


    //----------------------------------------------------------------
    string gString() const override;

    //----------------------------------------------------------------
    const char* gTypeStr() const override;

    //----------------------------------------------------------------
    bool qPassThrough() const override;

    //----------------------------------------------------------------
    bool qTanhLayer() const override;
};

} // namespace layers
} // namespace kcc

#endif // KCC_LAYERS_TANHLAYER_H

