
#include "constlayer.hpp"

namespace kcc {
namespace layers {

//----------------------------------------------------------------
std::string
ConstLayer::gString() const
{
    std::string baseLayer = gBaseLayerStr();
    return (gName() + baseLayer + gStateSizesStr());
}


}}


