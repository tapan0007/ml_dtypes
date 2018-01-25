
#include "constlayer.hpp"

namespace kcc {
namespace layers {

//----------------------------------------------------------------
std::string
ConstLayer::gString() const
{
    string baseLayer = gBaseLayerStr();
    return (gName() + baseLayer + gStateSizesStr());
}


}}


