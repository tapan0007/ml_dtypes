#include "serialize/inc/serlayer.hpp"

namespace kcc {
namespace serialize {

SerLayer::SerLayer()
{
    m_OfmapShape.resize(4);
    m_KernelShape.resize(4);   // conv,pool
    m_Batching.resize(4);
    m_Stride.resize(4);
    m_Padding.resize(4);
    for (auto& v : m_Padding) {
        v.resize(2);
    }

    m_Batching[FmapIndex_N]     = 1;
    m_Batching[FmapIndex_C]     = 1;
    m_Batching[FmapIndex_H]     = 1;
    m_Batching[FmapIndex_W]     = 1;

    m_Stride[FilterIndex_M]     = 1;
    m_Stride[FilterIndex_C]     = 1;
    m_Stride[FilterIndex_R]     = 1;
    m_Stride[FilterIndex_S]     = 1;

    m_Padding[FmapIndex_N][0]   = 0;
    m_Padding[FmapIndex_N][1]   = 0;
    m_Padding[FmapIndex_C][0]   = 0;
    m_Padding[FmapIndex_C][1]   = 0;
    m_Padding[FmapIndex_H][0]   = 0; // Top
    m_Padding[FmapIndex_H][1]   = 0; // Bottom
    m_Padding[FmapIndex_W][0]   = 0; // Left
    m_Padding[FmapIndex_W][1]   = 0; // Right

    /*
    m_BatchingInWave  = -1;
    */
}

//----------------------------------------------------------------
const std::string&
SerLayer::gName() const
{
    return m_LayerName;
}

//----------------------------------------------------------------

void
SerLayer::rStride(const StrideType stride)
{
    for (kcc_int32 i = 0; i < FILTER_TENSOR_RANK; ++i) {
        m_Stride[i] = stride[i];
    }
}


void
SerLayer::rKernelShape(const KernelShapeType  kernelShape)
{
    for (kcc_int32 i = 0; i < FILTER_TENSOR_RANK; ++i) {
        m_KernelShape[i] = kernelShape[i];
    }
}

void
SerLayer::rPadding(const PaddingType padding)
{
    for (kcc_int32 i0 = 0; i0 < FMAP_TENSOR_RANK; ++i0) {
        for (kcc_int32 i1 = 0; i1 < 2; ++i1) { // 1 for before, 1 for after
            m_Padding[i0][i1] = padding[i0][i1];
        }
    }
}


//----------------------------------------------------------------
void
SerLayer::rOfmapShape(const OfmapShapeType ofmapShape)
{
    for (kcc_int32 i = 0; i < FMAP_TENSOR_RANK; ++i) {
        m_OfmapShape[i] = ofmapShape[i];
    }
}



} // namespace serialize
} // namespace kcc

