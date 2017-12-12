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
namespace serialize {


//--------------------------------------------------------
class SerLayer { 
    //----------------------------------------------------
public:
    SerLayer()
    {
        m_Batching[0] = m_Batching[1] = m_Batching[2] = m_Batching[3] = 1;
        m_Stride[0]   = m_Stride[1]   = m_Stride[2]   = m_Stride[3]   = 0;
        m_Padding[0][0]      = m_Padding[0][1]  = 
            m_Padding[1][0]  = m_Padding[1][1]  =
            m_Padding[2][0]  = m_Padding[2][1]  = 
            m_Padding[3][0]  = m_Padding[3][1]  = 0;
    }

    template<typename Archive>
    void serialize(Archive & archive)
    {
        utils::breakFunc(33);
        archive(cereal::make_nvp(utils::Key_LayerType, m_LayerType));
        archive(cereal::make_nvp(utils::Key_LayerName, m_LayerName));
        archive(cereal::make_nvp(utils::Key_PrevLayers, m_PrevLayers));
        archive(cereal::make_nvp(utils::Key_OfmapShape, m_OfmapShape));
        archive(cereal::make_nvp(utils::Key_OfmapFormat, m_OfmapFormat));
        archive(cereal::make_nvp(utils::Key_RefFile, m_RefFile));

        if (m_LayerType == utils::TypeStr_Input) {
        } else if (m_LayerType == utils::TypeStr_Conv) {
            archive(cereal::make_nvp(utils::Key_KernelShape, m_KernelShape));
            archive(cereal::make_nvp(utils::Key_KernelFile, m_KernelFile));
            archive(cereal::make_nvp(utils::Key_KernelFormat, m_KernelFormat));
            archive(cereal::make_nvp(utils::Key_Stride, m_Stride));
            archive(cereal::make_nvp(utils::Key_Padding, m_Padding));
        } else if (m_LayerType == utils::TypeStr_Tanh) {
            // nothing specific to Tanh
        } else if (m_LayerType == utils::TypeStr_Relu) {
            // nothing specific to Relu
        } else {
            assert(false);
        }
    }

protected:
    //----------------------------------------------------------------

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
    int gBatchFactor() const {
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
    const std::string& gPrevLayer(int idx) const {
        assert(0 <= idx and idx < gNumPrevLayers());
        return m_PrevLayers[idx];
    }

    //----------------------------------------------------------------
    int gNumPrevLayers() const {
        return m_PrevLayers.size();
    }


    //----------------------------------------------------------------
    void rOfmapShape(const utils::OfmapShapeType ofmapShape) {
        for (int i = 0; i < 4; ++i) {
            m_OfmapShape[i] = ofmapShape[i];
        }
    }

    //----------------------------------------------------------------
    int gOfmapWidth() const {
        return m_OfmapShape[3];
    }

    //----------------------------------------------------------------
    int gOfmapHeight() const {
        return m_OfmapShape[2];
    }


    //----------------------------------------------------------------
    int gNumOfmaps() const {
        return m_OfmapShape[1];
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

    void rStride(const utils::StrideType stride) {        // conv,pool
        for (int i = 0; i < 4; ++i) {
            m_Stride[i] = stride[i];
        }
    }

    int gStrideVertical () const {
        return m_Stride[2];        // conv,pool
    }

    int gStrideHorizontal () const {
        return m_Stride[3];        // conv,pool
    }
    void rKernelShape(const utils::KernelShapeType  kernelShape) {//conv,pool
        for (int i = 0; i < 4; ++i) {
            m_KernelShape[i] = kernelShape[i];
        }
    }
    int gKernelHeight() const {
        return m_KernelShape[2];   // conv,pool
    }
    int gKernelWidth() const {
        return m_KernelShape[3];   // conv,pool
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

    void rPadding(const utils::PaddingType padding) {     // conv,pool
        for (int i0 = 0; i0 < 4; ++i0) {
            for (int i1 = 0; i1 < 2; ++i1) {
                m_Padding[i0][i1] = padding[i0][i1];
            }
        }
    }

    //----------------------------------------------------------------

private:
    std::string                 m_LayerType;
    std::string                 m_LayerName;
    vector<std::string>         m_PrevLayers;
    utils::OfmapShapeType       m_OfmapShape;

    std::string                 m_OfmapFormat;   // input,conv
    std::string                 m_RefFile;       // input

    std::string                 m_KernelFile;    // input(data), conv(weights)
    std::string                 m_KernelFormat;  // conv, pool
    utils::KernelShapeType      m_KernelShape;   // conv,pool
    utils::StrideType           m_Stride;        // conv,pool
    utils::PaddingType          m_Padding;       // conv,pool
    utils::BatchingType         m_Batching;
}; // class Layer



} // namespace serialize
} // namespace kcc

#endif // KCC_SERIALIZE_SERLAYER_H

