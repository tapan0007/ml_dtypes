#pragma once

#ifndef KCC_LAYERS_ACTIVLAYER_H
#define KCC_LAYERS_ACTIVLAYER_H

#include "types.hpp"
#include "datatype.hpp"
#include "fmapdesc.hpp"
#include "onetoonelayer.hpp"


namespace kcc {

namespace nets {
    class Network;
}

namespace layers {

using namespace utils;

//--------------------------------------------------------
// Base layer for all activation layers;
//--------------------------------------------------------
class ActivLayer : public OneToOneLayer {
private:
    using  SubClass = OneToOneLayer;
protected:
    //----------------------------------------------------------------
    ActivLayer(const Params& params, Layer* prev_layer);

public:
    //----------------------------------------------------------------
    bool qActivLayer() const override;
};

} // namespace layers
} // namespace kcc

#endif // KCC_LAYERS_ACTIVLAYER_H

