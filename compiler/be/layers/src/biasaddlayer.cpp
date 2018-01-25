
#include "biasaddlayer.hpp"

namespace kcc {
namespace layers {

//----------------------------------------------------------------
std::string
BiasAddLayer::gString() const
{
    string baseLayer = gBaseLayerStr();
    return (gName() + baseLayer + gStateSizesStr());
}


//----------------------------------------------------------------
bool
BiasAddLayer::verify() const
{
    return true;
}

}}


