#pragma once

#ifndef KCC_SERIALIZE_SERLAYER_H
#define KCC_SERIALIZE_SERLAYER_H


#include <string>
#include <vector>
#include <assert.h>

#include <cereal/types/memory.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/map.hpp>


using std::string;
using std::vector;


#include "debug.hpp"
#include "consts.hpp"
#include "types.hpp"
#include "datatype.hpp"


namespace kcc {
using  namespace utils;

namespace serialize {


//--------------------------------------------------------
class SerLayer {
    //----------------------------------------------------
public:
    SerLayer()
    {
        m_Batching[FmapIndex_N]     = 1;
        m_Batching[FmapIndex_C]     = 1;
        m_Batching[FmapIndex_H]     = 1;
        m_Batching[FmapIndex_W]     = 1;

        m_Stride[FilterIndex_M]     = 0;
        m_Stride[FilterIndex_C]     = 0;
        m_Stride[FilterIndex_R]     = 0;
        m_Stride[FilterIndex_S]     = 0;

        m_Padding[FmapIndex_N][0]   = 0;
        m_Padding[FmapIndex_N][1]   = 0;
        m_Padding[FmapIndex_C][0]   = 0;
        m_Padding[FmapIndex_C][1]   = 0;
        m_Padding[FmapIndex_H][0]   = 0; // Top
        m_Padding[FmapIndex_H][1]   = 0; // Bottom
        m_Padding[FmapIndex_W][0]   = 0; // Left
        m_Padding[FmapIndex_W][1]   = 0; // Right
    }

    SerLayer(const SerLayer&) = default;

public:
    template<typename Archive>
    void serialize(Archive & archive)
    {
        archive(cereal::make_nvp(Key_LayerType, m_LayerType));
        archive(cereal::make_nvp(Key_LayerName, m_LayerName));
        archive(cereal::make_nvp(Key_PrevLayers, m_PrevLayers));
        archive(cereal::make_nvp(Key_OfmapShape, m_OfmapShape));
        archive(cereal::make_nvp(Key_OfmapFormat, m_OfmapFormat));
        archive(cereal::make_nvp(Key_RefFile, m_RefFile));

        if (m_LayerType == TypeStr_Input) {
        } else if (m_LayerType == TypeStr_Conv) {
            archive(cereal::make_nvp(Key_KernelShape, m_KernelShape));
            archive(cereal::make_nvp(Key_KernelFile, m_KernelFile));
            archive(cereal::make_nvp(Key_KernelFormat, m_KernelFormat));
            archive(cereal::make_nvp(Key_Stride, m_Stride));
            archive(cereal::make_nvp(Key_Padding, m_Padding));
        } else if (m_LayerType == TypeStr_Tanh) {
            // nothing specific to Tanh
        } else if (m_LayerType == TypeStr_Relu) {
            // nothing specific to Relu
        } else {
            assert(false);
        }
    }

public:
    const string& gTypeStr() const {
        return m_LayerType;
    };

    void rLayerType(const string& t) {
        m_LayerType = t;
    }

    const string& gLayerName() const {
        return m_LayerName;
    }

    void rLayerName(const string& nm) {
        m_LayerName = nm;
    }

    //----------------------------------------------------------------
    kcc_int32 gBatchFactor() const {
        return m_Batching[0];
    }

    //----------------------------------------------------------------
    const vector<string>& gPrevLayers() const {
        return m_PrevLayers;
    }

    //----------------------------------------------------------------
    void addPrevLayer(const string& prevLayer) {
        m_PrevLayers.push_back(prevLayer);
    }

    //----------------------------------------------------------------
    const std::string& gPrevLayer(kcc_int32 idx) const {
        assert(0 <= idx and idx < gNumPrevLayers());
        return m_PrevLayers[idx];
    }

    //----------------------------------------------------------------
    kcc_int32 gNumPrevLayers() const {
        return m_PrevLayers.size();
    }


    //----------------------------------------------------------------
    void rOfmapShape(const OfmapShapeType ofmapShape) {
        for (kcc_int32 i = 0; i < FMAP_TENSOR_RANK; ++i) {
            m_OfmapShape[i] = ofmapShape[i];
        }
    }

    //----------------------------------------------------------------
    kcc_int32 gOfmapWidth() const {
        return m_OfmapShape[FmapIndex_W];
    }

    //----------------------------------------------------------------
    kcc_int32 gOfmapHeight() const {
        return m_OfmapShape[FmapIndex_H];
    }


    //----------------------------------------------------------------
    kcc_int32 gNumOfmaps() const {
        return m_OfmapShape[FmapIndex_C];
    }

    const std::string& gOfmapFormat() const {
        return m_OfmapFormat;   // input,conv
    }

    void rOfmapFormat(const std::string& fmt) {
        m_OfmapFormat = fmt;   // input,conv
    }

    //----------------------------------------------------------------
    const string& gName() const;

    const std::string& gRefFile() {
        return m_RefFile;       // input
    }
    void rRefFile(const std::string& f) {
        m_RefFile = f;       // input
    }

    void rStride(const StrideType stride) {        // conv,pool
        for (kcc_int32 i = 0; i < FILTER_TENSOR_RANK; ++i) {
            m_Stride[i] = stride[i];
        }
    }

    kcc_int32 gStrideVertical () const {
        return m_Stride[FilterIndex_R];        // conv,pool
    }

    kcc_int32 gStrideHorizontal () const {
        return m_Stride[FilterIndex_S];        // conv,pool
    }
    void rKernelShape(const KernelShapeType  kernelShape) {//conv,pool
        for (kcc_int32 i = 0; i < FILTER_TENSOR_RANK; ++i) {
            m_KernelShape[i] = kernelShape[i];
        }
    }
    kcc_int32 gKernelHeight() const {
        return m_KernelShape[FilterIndex_R];   // conv,pool
    }
    kcc_int32 gKernelWidth() const {
        return m_KernelShape[FilterIndex_S];   // conv,pool
    }

    const std::string& gKernelFile() const {   // input(data), conv(weights)
        return m_KernelFile;
    }
    void rKernelFile(const std::string& kfil) {   // input(data), conv(weights)
        m_KernelFile = kfil;
    }

    const std::string& gKernelFormat() const {   // conv, pool
        return m_KernelFormat;
    }
    void rKernelFormat(const std::string&  fmt) {   // conv, pool
        m_KernelFormat = fmt;
    }

    void rPadding(const PaddingType padding) {     // conv,pool
        for (kcc_int32 i0 = 0; i0 < FMAP_TENSOR_RANK; ++i0) {
            for (kcc_int32 i1 = 0; i1 < 2; ++i1) { // 1 for before, 1 for after
                m_Padding[i0][i1] = padding[i0][i1];
            }
        }
    }

    //----------------------------------------------------------------

private:
    std::string                 m_LayerType;
    std::string                 m_LayerName;
    vector<std::string>         m_PrevLayers;
    OfmapShapeType              m_OfmapShape;

    std::string                 m_OfmapFormat;   // input,conv
    std::string                 m_RefFile;       // input

    std::string                 m_KernelFile;    // input(data), conv(weights)
    std::string                 m_KernelFormat;  // conv, pool
    KernelShapeType             m_KernelShape;   // conv,pool
    StrideType                  m_Stride;        // conv,pool
    PaddingType                 m_Padding;       // conv,pool
    BatchingType                m_Batching;
}; // class Layer



} // namespace serialize
} // namespace kcc

#endif // KCC_SERIALIZE_SERLAYER_H

