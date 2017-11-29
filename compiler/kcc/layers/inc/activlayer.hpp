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

    //----------------------------------------------------------------
    ActivLayer(const Params& params, Layer* prev_layer);

    //----------------------------------------------------------------
    bool qActivLayer() const;
};

} // namespace layers
} // namespace kcc

#endif

