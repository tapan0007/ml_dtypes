#pragma once

#ifndef KCC_NETS_NETWORK_H
#define KCC_NETS_NETWORK_H


#include <string>
#include <vector>
#include <assert.h>

using std::string;
using std::vector;


#include "consts.hpp"
#include "types.hpp"
#include "datatype.hpp"
#include "fmapdesc.hpp"

#include "inputlayer.hpp"
#include "convlayer.hpp"
#include "relulayer.hpp"
#include "tanhlayer.hpp"

#include "serlayer.hpp"


namespace kcc {

namespace layers {
    class Layer;
}

namespace nets {

using namespace utils;
using layers::Layer;

//--------------------------------------------------------
class Network {
public:
    static const char* const net_name_key;
    static const char* const data_type_key;
    static const char* const layers_key;
    class SchedForwLayers;
    class SchedRevLayers;
private:
    class SchedLayerForwRevIter;

public:
    template<class Archive>
    void save(Archive & archive)
    {
        archive(cereal::make_nvp(utils::Key_NetName, m_Name));
        archive(cereal::make_nvp(utils::Key_DataType, m_DataType->gName()));

        vector<std::unique_ptr<Layer> > Ulayers;
        for (unsigned i = 0; i < m_Layers.size(); ++i) {
            Ulayers.push_back(std::move(std::unique_ptr<Layer>(m_Layers[i])));
        }
        archive(cereal::make_nvp(utils::Key_Layers, Ulayers));
    }

    template<class Archive>
    void load(Archive & archive)
    {
        archive(cereal::make_nvp(utils::Key_NetName, m_Name));
        string netName;
        archive(cereal::make_nvp(utils::Key_DataType, netName));
        if (netName == DataTypeInt8::gNameStatic()) {
            m_DataType = new DataTypeInt8();
        } else if (netName==DataTypeInt16::gNameStatic()) {
            m_DataType = new DataTypeInt16();
        } else if (netName == DataTypeFloat16::gNameStatic()) {
            m_DataType = new DataTypeFloat16();
        } else {
            assert(0);
        }
         
        vector<std::unique_ptr<serialize::SerLayer> > serLayers;
        archive(cereal::make_nvp(utils::Key_Layers, serLayers));
        for (auto it : serLayers) {
            serialize::SerLayer& serLayer(*it);
            Layer::Params params;
            params.m_LayerName = serLayer.gName();
            params.m_BatchFactor = serLayer.gBatchFactor();
            params.m_Network = this;

            Layer* layer;
            if (serLayer.gTypeStr() == TypeStr_Input) {
                FmapDesc fmap_desc;
                const string inputDataFileName = serLayer.gRefFile();
                const string dataTensorDimSemantics = serLayer.gOfmapFormat();
                layer = new layers::InputLayer(params, fmap_desc, 
                            inputDataFileName.c_str(), dataTensorDimSemantics.c_str());
            } else if (serLayer.gTypeStr() == TypeStr_Conv) {
                const string& prevLayerName = serLayer.gPrevLayer(0);
                Layer* prevLayer = findLayer(prevLayerName);
                const int num_ofmaps = serLayer.gNumOfmaps();
                std::tuple<int,int> stride = std::make_tuple(serLayer.gStrideVertical(), serLayer.gStrideHorizontal());
                std::tuple<int,int> kernel = std::make_tuple(serLayer.gStrideVertical(), serLayer.gStrideHorizontal());
                const string filterFileName = serLayer.gKernelFile();
                const string filterTensorDimSemantics = serLayer.gKernelFormat();

                layer = new layers::ConvLayer(params, prevLayer, num_ofmaps,
                                                    stride, kernel,
                                                    filterFileName.c_str(),
                                                    filterTensorDimSemantics.c_str());
            }
            m_Layers.push_back(std::move(layer));
        }
    }
    
    Layer* findLayer(const string& prevLayerName);

public:
    //----------------------------------------------------------------
    Network(const DataType* dataType, const string& netName);

    bool qDoBatching() const {
        return m_DoBatching;
    }
    void rDoBatching(bool doBatch) {
        m_DoBatching = doBatch;
    }

    std::vector<Layer*>& gLayers() {
        return m_Layers;
    }

    Layer* gLayer(int idx) const {
        return m_Layers[idx];
    }

    int gNumberLayers() const {
        return m_Layers.size();
    }

    const DataType* gDataType() const {
        return m_DataType;
    }

    void addLayer(Layer* layer);
    
    SchedForwLayers gSchedForwLayers();
    SchedRevLayers gSchedRevLayers();

private:
    const DataType* m_DataType;
    string m_Name;
    vector<Layer*> m_Layers;
    bool m_DoBatching;
}; // class Layer




//----------------------------------------------------------------
class Network::SchedLayerForwRevIter {
public:
    SchedLayerForwRevIter(Layer* startLayer, bool forw)
        : m_CurrLayer(startLayer)
        , m_Forw(forw)
    { }

    bool operator!= (const SchedLayerForwRevIter& rhs) const {
        return m_CurrLayer != rhs.m_CurrLayer;
    }

    Layer* operator* () const {
        return m_CurrLayer;
    }

    void operator++();
private:
    Layer* m_CurrLayer;
    const bool m_Forw;
};

//--------------------------------------------------------
class Network::SchedForwLayers {
public:
    SchedForwLayers(std::vector<Layer*>& layers)
        : m_Layers(layers)
    { }
    SchedLayerForwRevIter begin() const {
        return SchedLayerForwRevIter(m_Layers[0], true);
    }
    SchedLayerForwRevIter end() const {
        return SchedLayerForwRevIter(nullptr, true);
    }
private:
    vector<Layer*>& m_Layers;
};

//--------------------------------------------------------
class Network::SchedRevLayers {
public:
    SchedRevLayers(std::vector<Layer*>& layers)
        : m_Layers(layers)
    { }
    SchedLayerForwRevIter begin() const {
        return SchedLayerForwRevIter(m_Layers[m_Layers.size()-1], false);
    }
    SchedLayerForwRevIter end() const {
        return SchedLayerForwRevIter(nullptr, true);
    }
private:
    vector<Layer*>& m_Layers;
};

} // namespace nets
} // namespace kcc

#endif // KCC_NETS_NETWORK_H

