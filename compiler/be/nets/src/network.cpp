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
#include "constlayer.hpp"
#include "convlayer.hpp"
#include "relulayer.hpp"
#include "tanhlayer.hpp"
#include "maxpoollayer.hpp"
#include "avgpoollayer.hpp"
#include "resaddlayer.hpp"
#include "biasaddlayer.hpp"

#include "network.hpp"
#include "serlayer.hpp"

namespace kcc {

namespace nets {

//--------------------------------------------------------
#if 0
Network::Network(const DataType* dataType, const std::string& netName)
    : m_DataType(dataType)
    , m_Name(netName)
    , m_DoBatching(false)
{
}
#endif

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
    assert(currLayer && "Layer iterator in Network: Invalid current layer");
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
        serLayer.rLayerType(std::string(layer->gTypeStr()));
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
        serLayer.rRefFile(layer->gRefFileName());
        serLayer.rOfmapFormat(layer->gRefFileFormat());

        /* The following series of 'IFs' is not coded as
         * if (inLayer = ) {
         * } else if (convLayer = ) {
         * } else if (tanhLayer = ) {
         * } else {
         * }
         * because inLayer is visibe in the whole sequence of 'else ifs'
         * and convLayer is visible inside tanhLayer's blocks.
         */

        if (const auto convLayer = dynamic_cast<layers::ConvLayer*>(layer)) {
            assert(convLayer->gPrevLayers().size() == 1U && "Convolution layer should have exactly one input layer");
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
            continue;
        }

        if (const auto poolLayer = dynamic_cast<layers::PoolLayer*>(layer)) {
            assert(poolLayer && "Expected Pool layer");
            assert(poolLayer->gPrevLayers().size() == 1U && "Pool layer should have one input");
            auto prevLayer = poolLayer->gPrevLayer(0);
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
            if (const auto maxpoolLayer = dynamic_cast<layers::MaxPoolLayer*>(poolLayer)) {
                assert(maxpoolLayer && "Expected MaxPool layer");
                serLayer.rLayerType(TypeStr_MaxPool);
                continue;
            }
            if (const auto avgpoolLayer = dynamic_cast<layers::AvgPoolLayer*>(poolLayer)) {
                assert(avgpoolLayer && "Expected AvgPool layer");
                serLayer.rLayerType(TypeStr_AvgPool);
                continue;
            }
            continue;
        }

        if (const auto inLayer = dynamic_cast<layers::InputLayer*>(layer)) {
            assert(inLayer && "Expected Input layer");
            continue;
        }

        if (const auto constLayer = dynamic_cast<layers::ConstLayer*>(layer)) {
            assert(constLayer && "Expected Const layer");
            continue;
        }

        if (const auto tanhLayer = dynamic_cast<layers::TanhLayer*>(layer)) {
            assert(tanhLayer && "Expected Tanh layer");
            continue;
        }

        if (const auto reluLayer = dynamic_cast<layers::ReluLayer*>(layer)) {
            assert(reluLayer && "Expected Relu layer");
            continue;
        }

        if (const auto biasAddLayer = dynamic_cast<layers::BiasAddLayer*>(layer)) {
            assert(biasAddLayer && "Expected BiasAdd layer");
            continue;
        }
        if (const auto resAddLayer = dynamic_cast<layers::ResAddLayer*>(layer)) {
            assert(resAddLayer && "Expected ResAdd layer");
            continue;
        }

        assert(false && "Unsupported layer");
    }
    archive(cereal::make_nvp(Key_Layers, serLayers));
}

//--------------------------------------------------------
template<>
void
Network::load<cereal::JSONInputArchive>(cereal::JSONInputArchive& archive)
{
    archive(cereal::make_nvp(Key_NetName, m_Name));
    std::string dataType;
    archive(cereal::make_nvp(Key_DataType, dataType));

    if (dataType == DataTypeInt8::gNameStatic()) {
        m_DataType = std::make_unique<DataTypeInt8>();

    } else if (dataType == DataTypeInt16::gNameStatic()) {
        m_DataType = std::make_unique<DataTypeInt16>();

    } else if (dataType == DataTypeUint8::gNameStatic()) {
        m_DataType = std::make_unique<DataTypeUint8>();

    } else if (dataType == DataTypeUint16::gNameStatic()) {
        m_DataType = std::make_unique<DataTypeUint16>();

    } else if (dataType == DataTypeFloat16::gNameStatic()) {
        m_DataType = std::make_unique<DataTypeFloat16>();

    } else if (dataType == DataTypeFloat32::gNameStatic()) {
        m_DataType = std::make_unique<DataTypeFloat32>();

    } else {
        assert(0 && "Unsupported data type");
    }

    std::vector<serialize::SerLayer> serLayers;
    archive(cereal::make_nvp(Key_Layers, serLayers));
    kcc::utils::breakFunc(333);
    for (unsigned i = 0; i < serLayers.size(); ++i) {
        serialize::SerLayer& serLayer(serLayers[i]);
        layers::Layer::Params params;
        params.m_LayerName = serLayer.gName();
        params.m_BatchFactor = serLayer.gBatchFactor();
        params.m_Network = this;
        params.m_RefFile = serLayer.gRefFile();
        params.m_RefFileFormat = serLayer.gOfmapFormat();

        FmapDesc fmap_desc(serLayer.gNumOfmaps(), serLayer.gOfmapHeight(), serLayer.gOfmapWidth());
        layers::Layer* layer = nullptr;
        if (serLayer.gTypeStr() == TypeStr_Input) {
            assert(serLayer.gNumPrevLayers() == 0 && "Input layer should have zero inputs");
            layer = new layers::InputLayer(params, fmap_desc);
        } else if (serLayer.gTypeStr() == TypeStr_Const) {
            assert(serLayer.gNumPrevLayers() == 0 && "Const layer should have zero inputs");
            layer = new layers::ConstLayer(params, fmap_desc);
        } else if (serLayer.gTypeStr() == TypeStr_Conv) {
            assert(serLayer.gNumPrevLayers() == 1 && "Convolution layer should have one input");
            const std::string& prevLayerName = serLayer.gPrevLayer(0);
            layers::Layer* prevLayer = findLayer(prevLayerName);
            assert(prevLayer != nullptr && "Convolution: Unknown input layer");
            std::tuple<kcc_int32,kcc_int32> stride = std::make_tuple(
                                            serLayer.gStrideVertical(),
                                            serLayer.gStrideHorizontal());
            std::tuple<kcc_int32,kcc_int32> kernel = std::make_tuple(
                                            serLayer.gConvFilterHeight(),
                                            serLayer.gConvFilterWidth());
            std::tuple<kcc_int32,kcc_int32, kcc_int32,kcc_int32> padding = std::make_tuple(
                                            serLayer.gPaddingTop(),
                                            serLayer.gPaddingBottom(),
                                            serLayer.gPaddingLeft(),
                                            serLayer.gPaddingRight());

            const std::string filterFileName = serLayer.gKernelFile();
            const std::string filterTensorDimSemantics = serLayer.gKernelFormat();

            layer = new layers::ConvLayer(params, prevLayer,
                                          fmap_desc,
                                          stride, kernel, padding,
                                          filterFileName.c_str(),
                                          filterTensorDimSemantics.c_str());

        } else if (serLayer.gTypeStr() == TypeStr_MaxPool || serLayer.gTypeStr() == TypeStr_AvgPool) {
            assert(serLayer.gNumPrevLayers() == 1 && "Pool layer: number of inputs not 1");
            const std::string& prevLayerName = serLayer.gPrevLayer(0);
            layers::Layer* prevLayer = findLayer(prevLayerName);
            assert(prevLayer != nullptr && "Pool: Unknown previous layer");
            std::tuple<kcc_int32,kcc_int32> stride = std::make_tuple(
                                            serLayer.gStrideVertical(),
                                            serLayer.gStrideHorizontal());
            std::tuple<kcc_int32,kcc_int32> kernel = std::make_tuple(
                                            serLayer.gPoolKernelHeight(),
                                            serLayer.gPoolKernelWidth());
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
                        stride,
                        kernel,
                        padding);
            } else {
                layer = new layers::AvgPoolLayer(
                        params,
                        prevLayer,
                        fmap_desc,
                        stride,
                        kernel,
                        padding);
            }

        } else if (serLayer.gTypeStr() == TypeStr_Relu) {
            assert(serLayer.gNumPrevLayers() == 1 && "Relu layer: number of inputs not 1");
            const std::string& prevLayerName = serLayer.gPrevLayer(0);
            layers::Layer* prevLayer = findLayer(prevLayerName);
            assert(prevLayer != nullptr && "Relu: Unknown previous layer");
            layer = new layers::ReluLayer(params, prevLayer);
        } else if (serLayer.gTypeStr() == TypeStr_Tanh) {
            assert(serLayer.gNumPrevLayers() == 1 && "Tanh layer: number of inputs not 1");
            const std::string& prevLayerName = serLayer.gPrevLayer(0);
            layers::Layer* prevLayer = findLayer(prevLayerName);
            assert(prevLayer != nullptr && "Tanh: Unknown previous layer");
            layer = new layers::TanhLayer(params, prevLayer);

        } else if (serLayer.gTypeStr() == TypeStr_ResAdd) {
            // TODO: check dimensions and types of inputs
            assert(serLayer.gNumPrevLayers() == 2 && "ResAdd layer should have two inputs");
            std::vector<layers::Layer*> prevLayers;
            for (const auto& prevLayerName : serLayer.gPrevLayers()) {
                layers::Layer* prevLayer = findLayer(prevLayerName);
                assert(prevLayer != nullptr && "RadAdd: Unknown previous layer");
                prevLayers.push_back(prevLayer);
            }
            layer = new layers::ResAddLayer(params, fmap_desc,prevLayers);
        } else if (serLayer.gTypeStr() == TypeStr_BiasAdd) {
            // TODO check dimensions and types of inputs
            assert(serLayer.gNumPrevLayers() == 2 && "BiasAdd layer should have two inputs");
            std::vector<layers::Layer*> prevLayers;
            for (const auto& prevLayerName : serLayer.gPrevLayers()) {
                layers::Layer* prevLayer = findLayer(prevLayerName);
                assert(prevLayer != nullptr && "RadAdd: Unknown previous layer");
                prevLayers.push_back(prevLayer);
            }
            layer = new layers::BiasAddLayer(params, fmap_desc, prevLayers);
        } else {
            assert(false && "Unsuported layer");
        }

        m_Name2Layer[params.m_LayerName] = layer;
    }
    assert(m_Layers.size() == serLayers.size() && "Layer mismatch count after input deserialization" );
}

//--------------------------------------------------------
layers::Layer*
Network::findLayer(const std::string& layerName)
{
    layers::Layer* layer = m_Name2Layer[layerName];
    assert(layer && "Could not find layer");
    return layer;
}

}}


