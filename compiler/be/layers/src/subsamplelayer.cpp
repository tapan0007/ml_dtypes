#include <sstream>

#include "datatype.hpp"
#include "network.hpp"
#include "subsamplelayer.hpp"



namespace kcc {
namespace layers {


//--------------------------------------------------------
SubSampleLayer::SubSampleLayer (const Params& param, Layer* prev_layer,
         kcc_int32 num_ofmaps, const string& dataTensorSemantics,
         const std::tuple<kcc_int32,kcc_int32>& stride,
         const std::tuple<kcc_int32,kcc_int32>& kernel,
         const std::tuple<kcc_int32,kcc_int32,kcc_int32,kcc_int32>& padding)
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
    m_Stride.m_TopBottom = std::get<StrideIndex_TopBottom>(stride);
    m_Stride.m_LeftRight = std::get<StrideIndex_LeftRight>(stride);

    m_Kernel.m_Height = std::get<0>(kernel);
    m_Kernel.m_Width  = std::get<1>(kernel);

    m_Padding.m_Top    = std::get<PaddingIndex_Top>(padding);
    m_Padding.m_Bottom = std::get<PaddingIndex_Bottom>(padding);
    m_Padding.m_Left   = std::get<PaddingIndex_Left>(padding);
    m_Padding.m_Right  = std::get<PaddingIndex_Right>(padding);
}


//--------------------------------------------------------
bool
SubSampleLayer::verify() const
{
    assert(gPrevLayer(0)->gOfmapWidth()  == gStrideLeftRight() * gOfmapWidth());
    assert(gPrevLayer(0)->gOfmapHeight() == gStrideTopBottom() * gOfmapHeight());
    return this->SubClass::verify();
}

} // namespace layers
} // namespace kcc

