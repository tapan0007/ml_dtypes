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




#include "utils/inc/debug.hpp"
#include "utils/inc/consts.hpp"
#include "utils/inc/types.hpp"
#include "utils/inc/datatype.hpp"


namespace kcc {
using  namespace utils;

namespace serialize {


//--------------------------------------------------------
class SerLayer {
    //----------------------------------------------------
public:
    SerLayer();

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
            // nothing specific to Input layer
        } else if (m_LayerType == TypeStr_Const) {
            // nothing specific to Const layer
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
        } else if (m_LayerType == TypeStr_MaxPool || m_LayerType == TypeStr_AvgPool) {
            archive(cereal::make_nvp(Key_KernelShape, m_KernelShape));
            archive(cereal::make_nvp(Key_Stride, m_Stride));
            archive(cereal::make_nvp(Key_Padding, m_Padding));
        } else if (m_LayerType == TypeStr_BiasAdd) {
            // nothing specific to BiasAdd layer
        } else if (m_LayerType == TypeStr_ResAdd) {
            // nothing specific to ResAdd layer
        } else {
            assert(false && "Serialization: unsupported layer");
        }
    }

public:
    const std::string& gTypeStr() const {
        return m_LayerType;
    };


    void rLayerType(const std::string& t) {
        m_LayerType = t;
    }

    const std::string& gLayerName() const {
        return m_LayerName;
    }

    void rLayerName(const std::string& nm) {
        m_LayerName = nm;
    }

    //----------------------------------------------------------------
    kcc_int32 gBatchFactor() const {
        return m_Batching[0];
    }

    //----------------------------------------------------------------
    const std::vector<std::string>& gPrevLayers() const {
        return m_PrevLayers;
    }

    //----------------------------------------------------------------
    void addPrevLayer(const std::string& prevLayer) {
        m_PrevLayers.push_back(prevLayer);
    }

    //----------------------------------------------------------------
    const std::string& gPrevLayer(kcc_int32 idx) const {
        assert(0 <= idx and idx < gNumPrevLayers() && "Previous layer index out of range");
        return m_PrevLayers[idx];
    }

    //----------------------------------------------------------------
    kcc_int32 gNumPrevLayers() const {
        return m_PrevLayers.size();
    }


    //----------------------------------------------------------------
    void rOfmapShape(const OfmapShapeType ofmapShape);

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
    const std::string& gName() const;

    const std::string& gRefFile() {
        return m_RefFile;       // input
    }
    void rRefFile(const std::string& f) {
        m_RefFile = f;       // input
    }

    void rStride(const StrideType stride);

    kcc_int32 gStrideVertical () const {
        return m_Stride[FmapIndex_H];        // conv,pool
    }

    kcc_int32 gStrideHorizontal () const {
        return m_Stride[FmapIndex_W];        // conv,pool
    }

    void rKernelShape(const KernelShapeType  kernelShape);

    kcc_int32 gConvFilterHeight() const {
        return m_KernelShape[FilterIndex_R];   // conv,pool
    }
    kcc_int32 gConvFilterWidth() const {
        return m_KernelShape[FilterIndex_S];   // conv,pool
    }

    kcc_int32 gPoolKernelHeight() const {
        return m_KernelShape[FmapIndex_H];   // conv,pool
    }
    kcc_int32 gPoolKernelWidth() const {
        return m_KernelShape[FmapIndex_W];   // conv,pool
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

    void rPadding(const PaddingType padding);

    kcc_int32 gPaddingTop() const {
        return m_Padding[FmapIndex_H][0];
    }

    kcc_int32 gPaddingBottom() const {
        return m_Padding[FmapIndex_H][1];
    }

    kcc_int32 gPaddingLeft() const {
        return m_Padding[FmapIndex_W][0];
    }

    kcc_int32 gPaddingRight() const {
        return m_Padding[FmapIndex_W][1];
    }

    //----------------------------------------------------------------

private:
    std::string                 m_LayerType;
    std::string                 m_LayerName;
    std::vector<std::string>    m_PrevLayers;

    //OfmapShapeType            m_OfmapShape;
    std::vector<int>            m_OfmapShape;

    std::string                 m_OfmapFormat;   // input,conv
    std::string                 m_RefFile;       // input

    std::string                 m_KernelFile;    // input(data), conv(weights)
    std::string                 m_KernelFormat;  // conv, pool

    //KernelShapeType             m_KernelShape;   // conv,pool
    std::vector<int>                 m_KernelShape;   // conv,pool

    //StrideType                  m_Stride;        // conv,pool
    std::vector<int>                 m_Stride;        // conv,pool

    //PaddingType                 m_Padding;       // conv,pool
    std::vector<std::vector<int> >        m_Padding;       // conv,pool

    //BatchingType                m_Batching;
    std::vector<int>                 m_Batching;
}; // class SerLayer



} // namespace serialize
} // namespace kcc

#endif // KCC_SERIALIZE_SERLAYER_H

