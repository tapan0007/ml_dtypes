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
#include "maxpoollayer.hpp"
#include "avgpoollayer.hpp"

#include "network.hpp"
#include "serlayer.hpp"

namespace kcc {

namespace nets {

//--------------------------------------------------------
Network::Network(const DataType* dataType, const string& netName)
    : m_DataType(dataType)
    , m_Name(netName)
    , m_DoBatching(false)
{
}

//--------------------------------------------------------
void
Network::addLayer(layers::Layer* layer)
{
    m_Layers.push_back(layer);
}



//--------------------------------------------------------
void
Network::SchedLayerForwRevIter::operator++()
{
    layers::Layer* const currLayer = m_CurrLayer;
    assert(currLayer);
    layers::Layer* nextLayer;

    if (m_Forw) {
        nextLayer = currLayer->gNextSchedLayer();
    } else {
        nextLayer = currLayer->gPrevSchedLayer();
    }

    m_CurrLayer = nextLayer;
}

//--------------------------------------------------------
Network::SchedForwLayers
Network::gSchedForwLayers() const
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
    archive(cereal::make_nvp(Key_NetName, m_Name));
    archive(cereal::make_nvp(Key_DataType,
                            std::string(m_DataType->gName())));

    // Temporary to vector for Cereal
    std::vector<serialize::SerLayer> serLayers(m_Layers.size());
    for (unsigned i = 0; i < m_Layers.size(); ++i) {
        layers::Layer* layer = m_Layers[i];
        serialize::SerLayer& serLayer(serLayers[i]);

        serLayer.rLayerName(layer->gName());
        serLayer.rLayerType(string(layer->gTypeStr()));
        for (auto prevLayer : layer->gPrevLayers()) {
            serLayer.addPrevLayer(prevLayer->gName());
        }
        {
            OfmapShapeType ofmapShape;
            ofmapShape[FmapIndex_N] = layer->gBatchFactor();
            ofmapShape[FmapIndex_C] = layer->gNumOfmaps();
            ofmapShape[FmapIndex_H] = layer->gOfmapHeight();
            ofmapShape[FmapIndex_W] = layer->gOfmapWidth();
            serLayer.rOfmapShape(ofmapShape);
        }
        serLayer.rOfmapFormat(layer->gDataTensorDimSemantics());

        if (auto inLayer = dynamic_cast<layers::InputLayer*>(layer)) {
            serLayer.rRefFile(inLayer->gInputDataFileName());
        } else if (auto convLayer = dynamic_cast<layers::ConvLayer*>(layer)) {
            assert(convLayer->gPrevLayers().size() == 1U);
            const layers::Layer* prevLayer = convLayer->gPrevLayer(0);
            const int32_t numIfmaps = prevLayer->gNumOfmaps();
            const int32_t batchStride = 1, ifmapStride = 1;
            const int32_t batchPadBefore = 0, batchPadAfter = 0, ifmapPadBefore = 0, ifmapPadAfter = 0;

            {
                KernelShapeType  kernelShape;   // conv,pool
                kernelShape[FilterIndex_M] = convLayer->gNumOfmaps();
                kernelShape[FilterIndex_C] = numIfmaps;
                kernelShape[FilterIndex_R] = convLayer->gKernelHeight();
                kernelShape[FilterIndex_S] = convLayer->gKernelWidth();
                serLayer.rKernelShape(kernelShape);
            }

            serLayer.rKernelFile(convLayer->gFilterFileName());
            serLayer.rKernelFormat(convLayer->gFilterTensorDimSemantics());
            {
                StrideType stride;        // conv,pool
                stride[FmapIndex_N] = batchStride;
                stride[FmapIndex_C] = ifmapStride;
                stride[FmapIndex_H] = convLayer->gStrideTopBottom();
                stride[FmapIndex_W] = convLayer->gStrideLeftRight();
                serLayer.rStride(stride);
            }
            {
                PaddingType padding;       // conv,pool
                padding[FmapIndex_N][0] = batchPadBefore;
                padding[FmapIndex_N][1] = batchPadAfter;
                padding[FmapIndex_C][0] = ifmapPadBefore;
                padding[FmapIndex_C][1] = ifmapPadAfter;
                padding[FmapIndex_H][0] = convLayer->gPaddingTop();
                padding[FmapIndex_H][1] = convLayer->gPaddingBottom();
                padding[FmapIndex_W][0] = convLayer->gPaddingLeft();
                padding[FmapIndex_W][1] = convLayer->gPaddingRight();
                serLayer.rPadding(padding);
            }
        } else if (auto tanhLayer = dynamic_cast<layers::TanhLayer*>(layer)) {
            assert(tanhLayer);
        } else if (auto reluLayer = dynamic_cast<layers::ReluLayer*>(layer)) {
            assert(reluLayer);
        } else if (auto poolLayer = dynamic_cast<layers::PoolLayer*>(layer)) {
            assert(poolLayer);
            assert(poolLayer->gPrevLayers().size() == 1U);
            auto prevLayer = convLayer->gPrevLayer(0);
            const int32_t batchStride = 1, ifmapStride = 1;
            const int32_t batchPadBefore = 0, batchPadAfter = 0, ifmapPadBefore = 0, ifmapPadAfter = 0;

            {
                KernelShapeType  kernelShape;   // conv,pool
                kernelShape[FilterIndex_M] = poolLayer->gNumOfmaps();
                kernelShape[FilterIndex_C] = prevLayer->gNumOfmaps();
                kernelShape[FilterIndex_R] = poolLayer->gKernelHeight();
                kernelShape[FilterIndex_S] = poolLayer->gKernelWidth();
                serLayer.rKernelShape(kernelShape);
            }
            {
                StrideType stride;        // conv,pool
                stride[FmapIndex_N] = batchStride;
                stride[FmapIndex_C] = ifmapStride;
                stride[FmapIndex_H] = poolLayer->gStrideTopBottom();
                stride[FmapIndex_W] = poolLayer->gStrideLeftRight();
                serLayer.rStride(stride);
            }
            {
                PaddingType padding;       // conv,pool
                padding[FmapIndex_N][0] = batchPadBefore;
                padding[FmapIndex_N][1] = batchPadAfter;
                padding[FmapIndex_C][0] = ifmapPadBefore;
                padding[FmapIndex_C][1] = ifmapPadAfter;
                padding[FmapIndex_H][0] = poolLayer->gPaddingTop();
                padding[FmapIndex_H][1] = poolLayer->gPaddingBottom();
                padding[FmapIndex_W][0] = poolLayer->gPaddingLeft();
                padding[FmapIndex_W][1] = poolLayer->gPaddingRight();
                serLayer.rPadding(padding);
            }
            if (dynamic_cast<layers::MaxPoolLayer*>(layer)) {
                serLayer.rLayerType(TypeStr_MaxPool);
            } else {
                serLayer.rLayerType(TypeStr_AvgPool);
            }
        } else {
            assert(false);
        }
    }
    archive(cereal::make_nvp(Key_Layers, serLayers));
}

//--------------------------------------------------------
template<>
void
Network::load<cereal::JSONInputArchive>(cereal::JSONInputArchive& archive)
{
    archive(cereal::make_nvp(Key_NetName, m_Name));
    string dataType;
    archive(cereal::make_nvp(Key_DataType, dataType));
    if (dataType == DataTypeInt8::gNameStatic()) {
        m_DataType = new DataTypeInt8();
    } else if (dataType==DataTypeInt16::gNameStatic()) {
        m_DataType = new DataTypeInt16();
    } else if (dataType == DataTypeFloat16::gNameStatic()) {
        m_DataType = new DataTypeFloat16();
    } else if (dataType == DataTypeFloat32::gNameStatic()) {
        m_DataType = new DataTypeFloat32();
    } else {
        assert(0);
    }

    vector<serialize::SerLayer> serLayers;
    archive(cereal::make_nvp(Key_Layers, serLayers));
    kcc::utils::breakFunc(333);
    for (unsigned i = 0; i < serLayers.size(); ++i) {
        serialize::SerLayer& serLayer(serLayers[i]);
        layers::Layer::Params params;
        params.m_LayerName = serLayer.gName();
        params.m_BatchFactor = serLayer.gBatchFactor();
        params.m_Network = this;
        const string refFile = serLayer.gRefFile();

        FmapDesc fmap_desc(
                    serLayer.gNumOfmaps(),
                    serLayer.gOfmapHeight(),
                    serLayer.gOfmapWidth());
        layers::Layer* layer = nullptr;
        if (serLayer.gTypeStr() == TypeStr_Input) {
            assert(serLayer.gNumPrevLayers() == 0);
            const string dataTensorDimSemantics = serLayer.gOfmapFormat();
            layer = new layers::InputLayer(params, fmap_desc,
                        refFile.c_str(), dataTensorDimSemantics.c_str());
        } else if (serLayer.gTypeStr() == TypeStr_Conv) {
            assert(serLayer.gNumPrevLayers() == 1);
            const string& prevLayerName = serLayer.gPrevLayer(0);
            layers::Layer* prevLayer = findLayer(prevLayerName);
            assert(prevLayer != nullptr);
            const string ofmapFormat(serLayer.gOfmapFormat());
            std::tuple<kcc_int32,kcc_int32> stride = std::make_tuple(
                                            serLayer.gStrideVertical(),
                                            serLayer.gStrideHorizontal());
            std::tuple<kcc_int32,kcc_int32> kernel = std::make_tuple(
                                            serLayer.gKernelHeight(),
                                            serLayer.gKernelWidth());
            std::tuple<kcc_int32,kcc_int32, kcc_int32,kcc_int32> padding = std::make_tuple(
                                            serLayer.gPaddingTop(),
                                            serLayer.gPaddingBottom(),
                                            serLayer.gPaddingLeft(),
                                            serLayer.gPaddingRight());

            const string filterFileName = serLayer.gKernelFile();
            const string filterTensorDimSemantics = serLayer.gKernelFormat();

            layer = new layers::ConvLayer(params, prevLayer,
                                          fmap_desc, ofmapFormat,
                                          stride, kernel, padding,
                                          filterFileName.c_str(),
                                          filterTensorDimSemantics.c_str());
        } else if (serLayer.gTypeStr() == TypeStr_Relu) {
            assert(serLayer.gNumPrevLayers() == 1);
            const string& prevLayerName = serLayer.gPrevLayer(0);
            layers::Layer* prevLayer = findLayer(prevLayerName);
            assert(prevLayer != nullptr);
            layer = new layers::ReluLayer(params, prevLayer);
        } else if (serLayer.gTypeStr() == TypeStr_Tanh) {
            assert(serLayer.gNumPrevLayers() == 1);
            const string& prevLayerName = serLayer.gPrevLayer(0);
            layers::Layer* prevLayer = findLayer(prevLayerName);
            assert(prevLayer != nullptr);
            layer = new layers::TanhLayer(params, prevLayer);
        } else if (serLayer.gTypeStr() == TypeStr_MaxPool || serLayer.gTypeStr() == TypeStr_AvgPool) {
            assert(serLayer.gNumPrevLayers() == 1);
            const string& prevLayerName = serLayer.gPrevLayer(0);
            layers::Layer* prevLayer = findLayer(prevLayerName);
            assert(prevLayer != nullptr);
            const string ofmapFormat(serLayer.gOfmapFormat());
            std::tuple<kcc_int32,kcc_int32> stride = std::make_tuple(
                                            serLayer.gStrideVertical(),
                                            serLayer.gStrideHorizontal());
            std::tuple<kcc_int32,kcc_int32> kernel = std::make_tuple(
                                            serLayer.gKernelHeight(),
                                            serLayer.gKernelWidth());
            std::tuple<kcc_int32,kcc_int32, kcc_int32,kcc_int32> padding = std::make_tuple(
                                            serLayer.gPaddingTop(),
                                            serLayer.gPaddingBottom(),
                                            serLayer.gPaddingLeft(),
                                            serLayer.gPaddingRight());

            if (serLayer.gTypeStr() == TypeStr_MaxPool) {
                layer = new layers::MaxPoolLayer(
                        params,
                        prevLayer,
                        fmap_desc,
                        ofmapFormat,
                        stride,
                        kernel,
                        padding);
            } else {
                layer = new layers::AvgPoolLayer(
                        params,
                        prevLayer,
                        fmap_desc,
                        ofmapFormat,
                        stride,
                        kernel,
                        padding);
            }
        } else {
            assert(false);
        }

        layer->rRefFileName(refFile);
        m_Name2Layer[params.m_LayerName] = layer;
    }
    assert(m_Layers.size() == serLayers.size());
}

//--------------------------------------------------------
layers::Layer*
Network::findLayer(const string& layerName)
{
    layers::Layer* layer = m_Name2Layer[layerName];
    assert(layer);
    return layer;
}

}}


