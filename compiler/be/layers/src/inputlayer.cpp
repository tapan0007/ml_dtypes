
#include "inputlayer.hpp"

namespace kcc {
namespace layers {

//----------------------------------------------------------------
std::string
InputLayer::gString() const
{
    std::string baseLayer = gBaseLayerStr();
    return (gName() + baseLayer + gStateSizesStr());
}


}}

