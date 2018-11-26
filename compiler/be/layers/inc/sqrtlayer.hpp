#pragma once

#ifndef KCC_LAYERS_SQRTLAYER_H
#define KCC_LAYERS_SQRTLAYER_H


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
// Sqrt activation
//--------------------------------------------------------
class SqrtLayer : public ActivLayer {
private:
    using SubClass = ActivLayer;
public:
    //----------------------------------------------------------------
    SqrtLayer(const Params& params, Layer* prev_layer);


    //----------------------------------------------------------------
    std::string gString() const override;

    //----------------------------------------------------------------
    const char* gTypeStr() const override;

    //----------------------------------------------------------------
    bool qPassThrough() const override;

    //----------------------------------------------------------------
    bool qSqrtLayer() const override;
};

} // namespace layers
} // namespace kcc

#endif // KCC_LAYERS_SqrtLayer_H

