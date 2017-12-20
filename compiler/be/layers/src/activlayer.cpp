#include <sstream>

#include "datatype.hpp"
#include "network.hpp"
#include "activlayer.hpp"



namespace kcc {
namespace layers {


//----------------------------------------------------------------
ActivLayer::ActivLayer(const Params& params, Layer* prev_layer)
    : OneToOneLayer(params, prev_layer)
{}

//----------------------------------------------------------------
bool
ActivLayer::qActivLayer() const
{
    return true;
}

} // namespace layers
} // namespace kcc

