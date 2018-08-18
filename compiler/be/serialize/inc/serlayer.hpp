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




// Layer keys

constexpr static const char* LayerKey_LayerType         = "layer_type";
constexpr static const char* LayerKey_LayerName         = "layer_name";
constexpr static const char* LayerKey_PrevLayers        = "previous_layers";
constexpr static const char* LayerKey_OfmapShape        = "ofmap_shape";

constexpr static const char* LayerKey_OfmapFormat       = "ofmap_format";
constexpr static const char* LayerKey_RefFile           = "ref_file";

constexpr static const char* LayerKey_KernelFile        = "kernel_file";
constexpr static const char* LayerKey_KernelFormat      = "kernel_format";
constexpr static const char* LayerKey_KernelShape       = "kernel_shape";
constexpr static const char* LayerKey_Stride            = "stride";
constexpr static const char* LayerKey_Padding           = "padding";
constexpr static const char* LayerKey_Batching          = "batching";



#define KCC_ARCHIVE(X) archive(cereal::make_nvp(KCC_CONCAT(LayerKey_,X), KCC_CONCAT(m_,X)))

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
        KCC_ARCHIVE(LayerType);
        KCC_ARCHIVE(LayerName);
        KCC_ARCHIVE(PrevLayers);
        KCC_ARCHIVE(OfmapShape);
        KCC_ARCHIVE(OfmapFormat);
        KCC_ARCHIVE(RefFile);

        if (m_LayerType == LayerTypeStr_Input) {
            // nothing specific to Input layer
        } else if (m_LayerType == LayerTypeStr_Const) {
            // nothing specific to Const layer
        } else if (m_LayerType == LayerTypeStr_Conv) {
            KCC_ARCHIVE(KernelShape);
            KCC_ARCHIVE(KernelFile);
            KCC_ARCHIVE(KernelFormat);
            KCC_ARCHIVE(Stride);
            KCC_ARCHIVE(Padding);

            /*
            KCC_ARCHIVE(BatchingInWave);
            */
        } else if (m_LayerType == LayerTypeStr_Matmul) {
            KCC_ARCHIVE(KernelShape);
            KCC_ARCHIVE(KernelFile);
            KCC_ARCHIVE(KernelFormat);
        } else if (m_LayerType == LayerTypeStr_Reshape) {
            // nothing specific to Reshape
        } else if (m_LayerType == LayerTypeStr_Tanh) {
            // nothing specific to Tanh
        } else if (m_LayerType == LayerTypeStr_Relu) {
            // nothing specific to Relu
        } else if (m_LayerType == LayerTypeStr_Sigmoid) {
            // nothing specific to Relu
        } else if (m_LayerType == LayerTypeStr_MaxPool || m_LayerType == LayerTypeStr_AvgPool) {
            KCC_ARCHIVE(KernelShape);
            KCC_ARCHIVE(Stride);
            KCC_ARCHIVE(Padding);
        } else if (m_LayerType == LayerTypeStr_BiasAdd) {
            // nothing specific to BiasAdd layer
        } else if (m_LayerType == LayerTypeStr_ResAdd || m_LayerType == LayerTypeStr_Multiply) {
            // nothing specific to ResAdd layer
        } else if (m_LayerType == LayerTypeStr_StridedSlice || m_LayerType == LayerTypeStr_Unstack || m_LayerType == LayerTypeStr_Concat) {
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

    const std::string& gRefFile() const {
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

/*
    kcc_int32 gBatchingInWave () const {
        return m_BatchingInWave;
    }
    void rBatchingInWave (kcc_int32 batchingInWave) {
        m_BatchingInWave = batchingInWave;
    }
*/
    //----------------------------------------------------------------

private:
    std::string                 m_LayerType;
    std::string                 m_LayerName;
    std::vector<std::string>    m_PrevLayers;

    std::vector<int>            m_OfmapShape;

    std::string                 m_OfmapFormat;   // input,conv
    std::string                 m_RefFile;       // input

    std::string                 m_KernelFile;    // input(data), conv(weights)
    std::string                 m_KernelFormat;  // conv, pool

    std::vector<int>            m_KernelShape;   // conv,pool

    std::vector<int>            m_Stride;        // conv,pool

    std::vector<std::vector<int> >        m_Padding;       // conv,pool

    std::vector<int>            m_Batching;
    /*
    int                         m_BatchingInWave  = -1;
    */
}; // class SerLayer

#undef KCC_ARCHIVE


} // namespace serialize
} // namespace kcc

#endif // KCC_SERIALIZE_SERLAYER_H

