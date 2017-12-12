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

using nets::Network;
using namespace utils;

//--------------------------------------------------------
class ActivLayer : public OneToOneLayer {
protected:
    //----------------------------------------------------------------
    ActivLayer(const Params& params, Layer* prev_layer);

public:
    //----------------------------------------------------------------
    bool qActivLayer() const override;
    long gNumberWeightsPerPartition() const override {
        return 0L;
    }
};

} // namespace layers
} // namespace kcc

#endif // KCC_LAYERS_ACTIVLAYER_H

