#include <sstream>

#include "convlayer.hpp"


namespace kcc {

namespace nets {
    class Network;
}

namespace layers {

//--------------------------------------------------------
ConvLayer::ConvLayer(const Params& params, Layer* prev_layer,
        int num_ofmaps, const string& dataTensorSemantics,
        const std::tuple<int,int>& stride, const std::tuple<int,int>& kernel,
        const char* filterFileName, const char* filterTensorDimSemantics)
    : SubSampleLayer(params, prev_layer,
                num_ofmaps, dataTensorSemantics,
                stride, kernel)
{
        m_FilterFileName = filterFileName;
        m_FilterTensorDimSemantics = filterTensorDimSemantics;
}

//--------------------------------------------------------
string
ConvLayer::gString() const
{
    std::stringstream ss;
    const int kh = gKernelHeight();
    const int kw = gKernelWidth();
    ss << gName() << gBaseLayerStr() << ", kernel=" << kh << "x" << kw
       << ",stride=" << gStrideBT() << "/" << gStrideLR() << gStateSizesStr();
    return ss.str();
}

//--------------------------------------------------------
bool
ConvLayer::verify() const
{
    assert(gNumPrevLayers() == 1);
    const bool ok = this->SubSampleLayer::verify();
    const Layer* prevLayer = gPrevLayer(0);
    const int prevMapWidth = prevLayer->gOfmapWidth();
    const int prevMapHeight = prevLayer->gOfmapHeight();
    assert(prevMapWidth ==  gStrideLR() * gOfmapWidth());
    assert(prevMapHeight == gStrideBT() * gOfmapHeight());
    return ok;
}

//--------------------------------------------------------
int 
ConvLayer::gNumberWeights() const
{
    assert(gNumPrevLayers() == 1);
    const int num_ifmaps = gPrevLayer(0)->gNumOfmaps();
    return num_ifmaps * gNumberWeightsPerPartition();
}

//--------------------------------------------------------
long
ConvLayer::gNumberWeightsPerPartition() const
{
    assert(gNumPrevLayers() == 1);
    const int kw = gKernelWidth();
    const int kh = gKernelHeight();
    const int num_ofmaps = gNumOfmaps();
    return kw*kh * num_ofmaps;
}



} // namespace layers
} // namespace kcc


