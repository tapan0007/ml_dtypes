
#include "maxpoollayer.hpp"

namespace kcc {
namespace layers {

//----------------------------------------------------------------
bool
MaxPoolLayer::verify() const
{
    assert(gNumPrevLayers() == 1);
    assert((gPrevLayer(0)->gOfmapWidth()  / gStrideLR()) == gOfmapWidth());
    assert((gPrevLayer(0)->gOfmapHeight() / gStrideBT()) == gOfmapHeight());
    return this->SubClass::verify();
}


}}

