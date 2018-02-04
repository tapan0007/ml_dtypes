
#include "layers/inc/constlayer.hpp"

namespace kcc {
namespace layers {

//----------------------------------------------------------------
std::string
ConstLayer::gString() const
{
    std::string baseLayer = gBaseLayerStr();
    return (gName() + baseLayer + gStateSizesStr());
}


//--------------------------------------------------------
kcc_int64
ConstLayer::gNumberWeightsPerPartition() const
{
    assert(gNumPrevLayers() == 0 && "Const layer: number previous layers not 0");

    //----------------------------------------------------------------
    const StateBufferAddress ofmapSize = gNumOfmaps();
    return ofmapSize;
}


}}


