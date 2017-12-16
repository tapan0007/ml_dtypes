#include <sstream>

#include "datatype.hpp"
#include "network.hpp"
#include "subsamplelayer.hpp"



namespace kcc {
namespace layers {


//--------------------------------------------------------
SubSampleLayer::SubSampleLayer (const Params& param, Layer* prev_layer,
         int num_ofmaps, const string& dataTensorSemantics,
         const std::tuple<int,int>& stride,
         const std::tuple<int,int>& kernel)
    : Layer(param,
            FmapDesc(
                (num_ofmaps >= 1 ? num_ofmaps : prev_layer->gNumOfmaps()),
                prev_layer->gOfmapHeight() / std::get<0>(stride),
                prev_layer->gOfmapWidth() / std::get<1>(stride)),
            dataTensorSemantics,
            vector<Layer*>(1, prev_layer))
{
    // Stride(2) is larger than kernel(1*1) for some layers in ResNet.
    // That seems to be a wrong way of aggregating information, but
    // it does happen.
    // assert(stride <= kernel)


    // Stride
    m_StrideBT = std::get<0>(stride);
    m_StrideLR = std::get<1>(stride);

    m_KernelHeight = std::get<0>(kernel);
    m_KernelWidth  = std::get<1>(kernel);

    // Padding
    const int kh = gKernelHeight();
    if (kh % 2 == 1) {
        m_PaddingBottom = m_PaddingTop = (kh - 1) / 2;
    } else {
        m_PaddingBottom = 0;
        m_PaddingTop    = kh / 2;
    }

    const int kw = gKernelWidth();
    if (kw % 2 == 1) {
        m_PaddingRight  = m_PaddingLeft  = (kw - 1) / 2;
    } else {
        m_PaddingLeft   = 0;
        m_PaddingRight  = kw / 2;
    }
}


//--------------------------------------------------------
bool
SubSampleLayer::verify() const
{
    assert(gPrevLayer(0)->gOfmapWidth()  == gStrideLR() * gOfmapWidth());
    assert(gPrevLayer(0)->gOfmapHeight() == gStrideBT() * gOfmapHeight());
    return this->SubClass::verify();
}

} // namespace layers
} // namespace kcc

