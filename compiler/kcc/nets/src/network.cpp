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
#include "inputlayer.hpp"
#include "convlayer.hpp"
#include "relulayer.hpp"
#include "tanhlayer.hpp"
#include "network.hpp"
#include "serlayer.hpp"

namespace kcc {
using layers::Layer;
using layers::InputLayer;
using layers::ConvLayer;
using layers::ReluLayer;
using layers::TanhLayer;

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
Network::save<cereal::JSONOutputArchive>(cereal::JSONOutputArchive& archive) const
{
    archive(cereal::make_nvp(utils::Key_NetName, m_Name));
    archive(cereal::make_nvp(utils::Key_DataType, 
                            std::string(m_DataType->gName())));

    // Temporary to vector of unique_ptrs for Cereal
    //std::vector<std::unique_ptr<serialize::SerLayer> > serLayers(m_Layers.size());
    std::vector<serialize::SerLayer> serLayers(m_Layers.size());
    for (unsigned i = 0; i < m_Layers.size(); ++i) {
        Layer* layer = m_Layers[i];
        //serLayers[i].reset(new serialize::SerLayer());
        serialize::SerLayer& serLayer(serLayers[i]);

        serLayer.rLayerName(layer->gName());
        serLayer.rLayerType(string(layer->gTypeStr()));
        for (auto prevLayer : layer->gPrevLayers()) {
            serLayer.addPrevLayer(prevLayer->gName());
        }
        {
            utils::OfmapShapeType ofmapShape;
            ofmapShape[0] = layer->gBatchFactor();
            ofmapShape[1] = layer->gNumOfmaps();
            ofmapShape[2] = layer->gOfmapHeight();
            ofmapShape[3] = layer->gOfmapWidth();
            serLayer.rOfmapShape(ofmapShape);
        }
        serLayer.rOfmapFormat(layer->gDataTensorDimSemantics());

        if (auto inLayer = dynamic_cast<InputLayer*>(layer)) {
            serLayer.rRefFile(inLayer->gInputDataFileName());
        } else if (auto convLayer = dynamic_cast<ConvLayer*>(layer)) {
            {
                utils::KernelShapeType  kernelShape;   // conv,pool
                kernelShape[0] = 1;
                kernelShape[1] = 1;
                kernelShape[2] = convLayer->gKernelHeight();
                kernelShape[3] = convLayer->gKernelWidth();
                serLayer.rKernelShape(kernelShape);
            }

            serLayer.rKernelFile(convLayer->gFilterFileName());
            serLayer.rKernelFormat(convLayer->gFilterTensorDimSemantics());
            {
                utils::StrideType stride;        // conv,pool
                stride[0] = 1;
                stride[1] = 1;
                stride[2] = convLayer->gStrideBT();
                stride[3] = convLayer->gStrideLR();
                serLayer.rStride(stride);
            }
            {
                utils::PaddingType padding;       // conv,pool
                padding[0][0] = 0; padding[0][1] = 0;
                padding[1][0] = 0; padding[1][1] = 0;
                padding[2][0] = convLayer->gPaddingBottom();
                padding[2][1] = convLayer->gPaddingTop();
                padding[3][0] = convLayer->gPaddingLeft();
                padding[3][1] = convLayer->gPaddingRight();
                serLayer.rPadding(padding);
            }
        } else if (auto tanhLayer = dynamic_cast<TanhLayer*>(layer)) {
            assert(tanhLayer);
        } else if (auto reluLayer = dynamic_cast<ReluLayer*>(layer)) {
            assert(reluLayer);
        } else {
            assert(false);
        }
    }
    archive(cereal::make_nvp(utils::Key_Layers, serLayers));
}

//--------------------------------------------------------
template<>
void
Network::load<cereal::JSONInputArchive>(cereal::JSONInputArchive& archive)
{
    archive(cereal::make_nvp(utils::Key_NetName, m_Name));
    string dataType;
    archive(cereal::make_nvp(utils::Key_DataType, dataType));
    if (dataType == DataTypeInt8::gNameStatic()) {
        m_DataType = new DataTypeInt8();
    } else if (dataType==DataTypeInt16::gNameStatic()) {
        m_DataType = new DataTypeInt16();
    } else if (dataType == DataTypeFloat16::gNameStatic()) {
        m_DataType = new DataTypeFloat16();
    } else {
        assert(0);
    }
     
    //vector<std::unique_ptr<serialize::SerLayer> > serLayers;
    vector<serialize::SerLayer> serLayers;
    archive(cereal::make_nvp(utils::Key_Layers, serLayers));
    kcc::utils::breakFunc(333);
    for (unsigned i = 0; i < serLayers.size(); ++i) {
        serialize::SerLayer& serLayer(serLayers[i]);
        Layer::Params params;
        params.m_LayerName = serLayer.gName();
        params.m_BatchFactor = serLayer.gBatchFactor();
        params.m_Network = this;
        const string refFile = serLayer.gRefFile();

        Layer* layer = nullptr;
        if (serLayer.gTypeStr() == TypeStr_Input) {
            assert(serLayer.gNumPrevLayers() == 0);
            FmapDesc fmap_desc(
                        serLayer.gNumOfmaps(),
                        serLayer.gOfmapHeight(),
                        serLayer.gOfmapWidth());
            const string dataTensorDimSemantics = serLayer.gOfmapFormat();
            layer = new layers::InputLayer(params, fmap_desc, 
                        refFile.c_str(), dataTensorDimSemantics.c_str());
        } else if (serLayer.gTypeStr() == TypeStr_Conv) {
            assert(serLayer.gNumPrevLayers() == 1);
            const string& prevLayerName = serLayer.gPrevLayer(0);
            Layer* prevLayer = findLayer(prevLayerName);
            assert(prevLayer != nullptr);
            const int num_ofmaps = serLayer.gNumOfmaps();
            const string ofmapFormat(serLayer.gOfmapFormat());
            std::tuple<int,int> stride = std::make_tuple(
                                            serLayer.gStrideVertical(),
                                            serLayer.gStrideHorizontal());
            std::tuple<int,int> kernel = std::make_tuple(
                                            serLayer.gKernelHeight(),
                                            serLayer.gKernelWidth());
            const string filterFileName = serLayer.gKernelFile();
            const string filterTensorDimSemantics = serLayer.gKernelFormat();

            layer = new layers::ConvLayer(params, prevLayer,
                                          num_ofmaps, ofmapFormat,
                                          stride, kernel,
                                          filterFileName.c_str(),
                                          filterTensorDimSemantics.c_str());
        } else if (serLayer.gTypeStr() == TypeStr_Relu) {
            assert(serLayer.gNumPrevLayers() == 1);
            const string& prevLayerName = serLayer.gPrevLayer(0);
            Layer* prevLayer = findLayer(prevLayerName);
            assert(prevLayer != nullptr);
            layer = new layers::ReluLayer(params, prevLayer);
        } else if (serLayer.gTypeStr() == TypeStr_Tanh) {
            assert(serLayer.gNumPrevLayers() == 1);
            const string& prevLayerName = serLayer.gPrevLayer(0);
            Layer* prevLayer = findLayer(prevLayerName);
            assert(prevLayer != nullptr);
            layer = new layers::TanhLayer(params, prevLayer);
        } else {
            assert(false);
        }
        layer->rRefFileName(refFile);
        m_Name2Layer[params.m_LayerName] = layer;
    }
    assert(m_Layers.size() == serLayers.size());
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


