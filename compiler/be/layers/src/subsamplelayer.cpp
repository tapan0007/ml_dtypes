#include <sstream>

#include "utils/inc/datatype.hpp"
#include "nets/inc/network.hpp"
#include "layers/inc/subsamplelayer.hpp"



namespace kcc {
namespace layers {


//--------------------------------------------------------
SubSampleLayer::SubSampleLayer (const Params& param, Layer* prev_layer,
         const FmapDesc& fmapDesc,
         const std::tuple<kcc_int32,kcc_int32>& stride,
         const std::tuple<kcc_int32,kcc_int32>& kernel,
         const std::tuple<kcc_int32,kcc_int32,kcc_int32,kcc_int32>& padding)
    : Layer(param, fmapDesc, std::vector<Layer*>(1, prev_layer))
{
    // Stride(2) is larger than kernel(1*1) for some layers in ResNet.
    // That seems to be a wrong way of aggregating information, but
    // it does happen.
    // assert(stride <= kernel && "SubSampleLayer: stride > kernel");


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
    assert(gPrevLayer(0)->gOfmapWidth()  == gStrideLeftRight() * gOfmapWidth() &&
            "SubSampleLayer: IFMAP width != stride * OFMAP width");
    assert(gPrevLayer(0)->gOfmapHeight() == gStrideTopBottom() * gOfmapHeight() &&
            "SubSampleLayer: IFMAP height != stride * OFMAP height");
    return this->SubClass::verify();
}

} // namespace layers
} // namespace kcc

