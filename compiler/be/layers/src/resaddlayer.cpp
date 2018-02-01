#include "resaddlayer.hpp"

namespace kcc {
namespace layers {

//----------------------------------------------------------------
std::string
ResAddLayer::gString() const
{
    std::string baseLayer = gBaseLayerStr();
    return (gName() + baseLayer + gStateSizesStr());
}


//----------------------------------------------------------------
bool
ResAddLayer::verify() const
{
    return true;
}

}}

