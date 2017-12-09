#pragma once

#ifndef KCC_SERIALIZE_SERLAYER_H
#define KCC_SERIALIZE_SERLAYER_H


#include <string>
#include <vector>
#include <assert.h>

#include <cereal/archives/json.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/map.hpp>


using std::string;
using std::vector;


#include "consts.hpp"
#include "types.hpp"
#include "datatype.hpp"


namespace kcc {
namespace serialize {


//--------------------------------------------------------
class SerLayer { 
    //----------------------------------------------------
public:
    template<class Archive>
    void serialize(Archive & archive)
    {
        archive(cereal::make_nvp(utils::Key_LayerType, m_LayerType));
        archive(cereal::make_nvp(utils::Key_LayerName, m_LayerName));
        archive(cereal::make_nvp(utils::Key_PrevLayers, m_PrevLayers));
        archive(cereal::make_nvp(utils::Key_OfmapShape, m_OfmapShape));

        if (m_LayerType == utils::TypeStr_Input) {
            archive(cereal::make_nvp(utils::Key_RefFile, m_RefFile));
            archive(cereal::make_nvp(utils::Key_OfmapFormat, m_OfmapFormat));
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
    const char* gTypeStr() const {
        return m_LayerType.c_str();
    };

    //----------------------------------------------------------------
    int gBatchFactor() const {
        return m_Batching[0];
    }

    //----------------------------------------------------------------
    const vector<string>& gPrevLayers() const {
        return m_PrevLayers;
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

    //----------------------------------------------------------------
    const string& gName() const;
    const std::string& gRefFile() {
        return m_RefFile;       // input
    }

    int gStrideVertical () const {
        return m_Stride[2];        // conv,pool
    }

    int gStrideHorizontal () const {
        return m_Stride[3];        // conv,pool
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
    const std::string& gKernelFormat() const {   // conv, pool
        return m_KernelFormat;
    }

    //----------------------------------------------------------------

private:
    std::string         m_LayerType;
    std::string         m_LayerName;
    vector<std::string> m_PrevLayers;
    utils::OfmapShapeType      m_OfmapShape;

    std::string         m_OfmapFormat;   // input,conv
    std::string         m_RefFile;       // input

    std::string         m_KernelFile;    // input(data), conv(weights)
    std::string         m_KernelFormat;  // conv, pool
    utils::KernelShapeType     m_KernelShape;   // conv,pool
    utils::StrideType          m_Stride;        // conv,pool
    utils::PaddingType         m_Padding;       // conv,pool
    utils::BatchingType        m_Batching;
}; // class Layer



#if 0

    #-----------------------------------------------------------------
    def combineJson(it):
        x = {}
        for y in it:
            x.update(y)
            #x = { **x, **y }
        return x

    static
    def gOfmapDescFromJson(klass, layerDict, nn):
        if nn.gUseDimList():
            of = layerDict[Layer.ofmap_key] ##  : [1, self.gNumOfmaps(), self.gOfmapHeight(), self.gOfmapWidth()]
            return OfmapDesc(of[1], (of[2], of[3]) )
        else:
            nOfmaps = layerDict[Layer.number_ofmaps_key]
            ofmapH = layerDict[Layer.ofmap_height_key]
            return OfmapDesc(nOfmaps, (ofmapW, ofmapH))

    static
    def gLayerNameFromJson(klass, layerDict):
        layerName = layerDict[Layer.layer_name_key]
        return layerName

    static
    def gPrevLayersFromJson(klass, layerDict, nn):
        prevLayers = []
        prevLayersNames = layerDict[Layer.prev_layers_key]
        for prevLayerName in prevLayersNames:
            prevLayers.append(nn.gLayerByName(prevLayerName))
        return prevLayers

#endif


} // namespace serialize
} // namespace kcc

#endif // KCC_SERIALIZE_SERLAYER_H

