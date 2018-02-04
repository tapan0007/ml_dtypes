#pragma once

#ifndef KCC_LAYERS_RELULAYER_H
#define KCC_LAYERS_RELULAYER_H

#include "utils/inc/types.hpp"
#include "utils/inc/datatype.hpp"
#include "utils/inc/fmapdesc.hpp"
#include "layers/inc/activlayer.hpp"


namespace kcc {

namespace nets {
    class Network;
}

namespace layers {

using namespace utils;

//--------------------------------------------------------
// Relu activation
//--------------------------------------------------------
class ReluLayer : public ActivLayer {
private:
    using SubClass = ActivLayer;
public:
    //----------------------------------------------------------------
    ReluLayer(const Params& params, Layer* prev_layer);


    //----------------------------------------------------------------
    std::string gString() const override;

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

