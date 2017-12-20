
#include "maxpoollayer.hpp"

namespace kcc {
namespace layers {

//----------------------------------------------------------------
bool
MaxPoolLayer::verify() const
{
    assert(gNumPrevLayers() == 1);
    assert((gPrevLayer(0)->gOfmapWidth()  / gStrideLeftRight()) == gOfmapWidth());
    assert((gPrevLayer(0)->gOfmapHeight() / gStrideTopBottom()) == gOfmapHeight());
    return this->SubClass::verify();
}


}}

