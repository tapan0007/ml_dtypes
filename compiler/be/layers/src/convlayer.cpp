#include <sstream>

#include "convlayer.hpp"


namespace kcc {

namespace nets {
    class Network;
}

namespace layers {

//--------------------------------------------------------
ConvLayer::ConvLayer(const Params& params, Layer* prev_layer,
        const FmapDesc& fmapDesc,
        const std::tuple<kcc_int32,kcc_int32>& stride, const std::tuple<kcc_int32,kcc_int32>& kernel,
        const std::tuple<kcc_int32,kcc_int32,kcc_int32,kcc_int32>& padding,
        const char* filterFileName, const char* filterTensorFormat)
    : SubSampleLayer(params, prev_layer, fmapDesc, stride, kernel, padding)
{
    m_FilterFileName           = filterFileName;
    m_FilterTensorFormat       = filterTensorFormat;
    m_WeightAddress            = StateBufferAddress_Invalid;
}

//--------------------------------------------------------
std::string
ConvLayer::gString() const
{
    std::stringstream ss;
    const kcc_int32 kh = gKernelHeight();
    const kcc_int32 kw = gKernelWidth();
    ss << gName() << gBaseLayerStr() << ", kernel=" << kh << "x" << kw
       << ",stride=" << gStrideTopBottom() << "/" << gStrideLeftRight() << gStateSizesStr();
    return ss.str();
}

//--------------------------------------------------------
bool
ConvLayer::verify() const
{
    assert(gNumPrevLayers() == 1 && " Conv layer: number of previous layers not 1");
    const bool ok = this->SubClass::verify();
    const Layer* prevLayer = gPrevLayer(0);
    const kcc_int32 prevMapWidth = prevLayer->gOfmapWidth();
    const kcc_int32 prevMapHeight = prevLayer->gOfmapHeight();
    assert(prevMapWidth ==  gStrideLeftRight() * gOfmapWidth() && "Conv layer: Ifmap width != stride * Ofmap width");
    assert(prevMapHeight == gStrideTopBottom() * gOfmapHeight() && "Conv layer: Ifmap height != stride * Ofmap height");
    return ok;
}

//--------------------------------------------------------
kcc_int32
ConvLayer::gNumberWeights() const
{
    assert(gNumPrevLayers() == 1 && "Conv layer: number previous layers not 1");
    const kcc_int32 num_ifmaps = gPrevLayer(0)->gNumOfmaps();
    return num_ifmaps * gNumberWeightsPerPartition();
}

//--------------------------------------------------------
kcc_int64
ConvLayer::gNumberWeightsPerPartition() const
{
    assert(gNumPrevLayers() == 1 && "Conv layer: number previous layers not 1");
    const kcc_int32 kw = gKernelWidth();
    const kcc_int32 kh = gKernelHeight();
    const kcc_int32 num_ofmaps = gNumOfmaps();
    return kw*kh * num_ofmaps;
}



} // namespace layers
} // namespace kcc


