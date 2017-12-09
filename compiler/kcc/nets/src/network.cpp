#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <map>

#include <cereal/types/memory.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/map.hpp>



#include "layer.hpp"
#include "network.hpp"

namespace kcc {

namespace nets {

//--------------------------------------------------------
Network::Network(const DataType* dataType, const string& netName)
    : m_DataType(dataType)
    , m_Name(netName)
{
}

//--------------------------------------------------------
void
Network::addLayer(Layer* layer)
{
    m_Layers.push_back(layer);
}



//--------------------------------------------------------
void 
Network::SchedLayerForwRevIter::operator++()
{
    Layer* const currLayer = m_CurrLayer;
    assert(currLayer);
    Layer* nextLayer;

    if (m_Forw) {
        nextLayer = currLayer->gNextSchedLayer();
    } else {
        nextLayer = currLayer->gPrevSchedLayer();
    }

    m_CurrLayer = nextLayer;
}

//--------------------------------------------------------
Network::SchedForwLayers
Network::gSchedForwLayers()
{
    return SchedForwLayers(m_Layers);
}

//--------------------------------------------------------
Network::SchedRevLayers
Network::gSchedRevLayers()
{
    return SchedRevLayers(m_Layers);
}

//--------------------------------------------------------
template<>
void
Network::save<cereal::JSONOutputArchive>(cereal::JSONOutputArchive& archive)
{
    archive(cereal::make_nvp(utils::Key_NetName, m_Name));
    archive(cereal::make_nvp(utils::Key_DataType, 
                            std::string(m_DataType->gName())));

    vector<std::unique_ptr<Layer> > Ulayers;
    for (unsigned i = 0; i < m_Layers.size(); ++i) {
        Ulayers.push_back(std::move(std::unique_ptr<Layer>(m_Layers[i])));
    }
    archive(cereal::make_nvp(utils::Key_Layers, Ulayers));
}

//--------------------------------------------------------
template<>
void
Network::load<cereal::JSONInputArchive>(cereal::JSONInputArchive& archive)
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
    for (unsigned i = 0; i < serLayers.size(); ++i) {
        serialize::SerLayer& serLayer(*serLayers[i]);
        Layer::Params params;
        params.m_LayerName = serLayer.gName();
        params.m_BatchFactor = serLayer.gBatchFactor();
        params.m_Network = this;

        Layer* layer;
        if (serLayer.gTypeStr() == TypeStr_Input) {
            FmapDesc fmap_desc(
                        serLayer.gNumOfmaps(),
                        serLayer.gOfmapHeight(),
                        serLayer.gOfmapWidth());
            const string inputDataFileName = serLayer.gRefFile();
            const string dataTensorDimSemantics = serLayer.gOfmapFormat();
            layer = new layers::InputLayer(params, fmap_desc, 
                        inputDataFileName.c_str(), dataTensorDimSemantics.c_str());
        } else if (serLayer.gTypeStr() == TypeStr_Conv) {
            const string& prevLayerName = serLayer.gPrevLayer(0);
            Layer* prevLayer = findLayer(prevLayerName);
            assert(prevLayer != nullptr);
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

//--------------------------------------------------------
Layer*
Network::findLayer(const string& layerName)
{
    Layer* layer = m_Name2Layer[layerName];
    assert(layer);
    return layer;
}

}}


