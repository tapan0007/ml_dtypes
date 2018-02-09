#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <map>

#include <cereal/types/memory.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/map.hpp>


#include "arch/inc/arch.hpp"

#include "layers/inc/layer.hpp"
#include "layers/inc/inputlayer.hpp"
#include "layers/inc/constlayer.hpp"
#include "layers/inc/convlayer.hpp"
#include "layers/inc/relulayer.hpp"
#include "layers/inc/tanhlayer.hpp"
#include "layers/inc/maxpoollayer.hpp"
#include "layers/inc/avgpoollayer.hpp"
#include "layers/inc/resaddlayer.hpp"
#include "layers/inc/biasaddlayer.hpp"

#include "nets/inc/network.hpp"
#include "serialize/inc/serlayer.hpp"
#include "serialize/inc/serwaveop.hpp"

namespace kcc {

/*
namespace wave {
    class SbAtomFileWaveOp;
    class SbAtomSaveWaveOp;
    class MatMulWaveOp;
}
*/

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

const arch::Arch&
Network::gArch() const
{
    return m_Arch;
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
    archive(cereal::make_nvp(NetKey_NetName, m_Name));
    archive(cereal::make_nvp(NetKey_DataType,
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
                KernelShapeType  kernelShape;
                kernelShape[FilterIndex_M] = convLayer->gNumOfmaps();
                kernelShape[FilterIndex_C] = numIfmaps;
                kernelShape[FilterIndex_R] = convLayer->gKernelHeight();
                kernelShape[FilterIndex_S] = convLayer->gKernelWidth();
                serLayer.rKernelShape(kernelShape);
            }

            serLayer.rKernelFile(convLayer->gFilterFileName());
            serLayer.rKernelFormat(convLayer->gFilterTensorDimSemantics());
            {
                StrideType stride;
                stride[FmapIndex_N] = batchStride;
                stride[FmapIndex_C] = ifmapStride;
                stride[FmapIndex_H] = convLayer->gStrideTopBottom();
                stride[FmapIndex_W] = convLayer->gStrideLeftRight();
                serLayer.rStride(stride);
            }
            {
                PaddingType padding;
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
                KernelShapeType  kernelShape;
                kernelShape[FilterIndex_M] = poolLayer->gNumOfmaps();
                kernelShape[FilterIndex_C] = prevLayer->gNumOfmaps();
                kernelShape[FilterIndex_R] = poolLayer->gKernelHeight();
                kernelShape[FilterIndex_S] = poolLayer->gKernelWidth();
                serLayer.rKernelShape(kernelShape);
            }
            {
                StrideType stride;
                stride[FmapIndex_N] = batchStride;
                stride[FmapIndex_C] = ifmapStride;
                stride[FmapIndex_H] = poolLayer->gStrideTopBottom();
                stride[FmapIndex_W] = poolLayer->gStrideLeftRight();
                serLayer.rStride(stride);
            }
            {
                PaddingType padding;
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
                serLayer.rLayerType(LayerTypeStr_MaxPool);
                continue;
            }
            if (const auto avgpoolLayer = dynamic_cast<layers::AvgPoolLayer*>(poolLayer)) {
                assert(avgpoolLayer && "Expected AvgPool layer");
                serLayer.rLayerType(LayerTypeStr_AvgPool);
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
    archive(cereal::make_nvp(NetKey_Layers, serLayers));


    //===========================================================================
    std::vector<serialize::SerWaveOp> serWaveOps(m_WaveOps.size());
    for (unsigned i = 0; i < m_WaveOps.size(); ++i) {
        serialize::SerWaveOp& serWaveOp(serWaveOps[i]);
        wave::WaveOp* waveOp = m_WaveOps[i];
        serWaveOp.rWaveOpName(waveOp->gName());
        serWaveOp.rLayerName(waveOp->gLayer()->gName());
        for (auto prevWaveOp : waveOp->gPrevWaveOps()) {
            serWaveOp.addPreviousWaveOp(prevWaveOp->gName());
        }

        if (const auto sbatomfileWaveOp = dynamic_cast<wave::SbAtomFileWaveOp*>(waveOp)) {
            serWaveOp.rWaveOpType(wave::SbAtomFileWaveOp::gTypeStr());

            serWaveOp.rAtomId(sbatomfileWaveOp->gAtomId());
            serWaveOp.rBatchFoldIdx(sbatomfileWaveOp->gBatchFoldIdx());
            serWaveOp.rLength(sbatomfileWaveOp->gLength());
            serWaveOp.rOffsetInFile(sbatomfileWaveOp->gOffsetInFile());
            serWaveOp.rRefFile(sbatomfileWaveOp->gRefFileName());

            serWaveOp.rIfmapsFoldIdx(sbatomfileWaveOp->gIfmapsFoldIdx());
            serWaveOp.rIfmapsReplicate(sbatomfileWaveOp->qIfmapsReplicate());
            continue;
        }

        if (const auto sbatomsaveWaveOp = dynamic_cast<wave::SbAtomSaveWaveOp*>(waveOp)) {
            serWaveOp.rWaveOpType(wave::SbAtomSaveWaveOp::gTypeStr());

            serWaveOp.rAtomId(sbatomsaveWaveOp->gAtomId());
            serWaveOp.rBatchFoldIdx(sbatomsaveWaveOp->gBatchFoldIdx());
            serWaveOp.rLength(sbatomsaveWaveOp->gLength());
            serWaveOp.rOffsetInFile(sbatomsaveWaveOp->gOffsetInFile());
            serWaveOp.rRefFile(sbatomsaveWaveOp->gRefFileName());

            serWaveOp.rOfmapsFoldIdx(sbatomsaveWaveOp->gOfmapsFoldIdx());
            continue;
        }

        if (const auto matmulWaveOp = dynamic_cast<wave::MatMulWaveOp*>(waveOp)) {
            serWaveOp.rWaveOpType(wave::MatMulWaveOp::gTypeStr());

            serWaveOp.rIfmapTileHeight(matmulWaveOp->gIfmapTileHeight());
            serWaveOp.rIfmapTileWidth(matmulWaveOp->gIfmapTileWidth());
            serWaveOp.rIfmapsAtomId(matmulWaveOp->gIfmapsAtomId());
            serWaveOp.rIfmapsOffsetInAtom(matmulWaveOp->gIfmapsOffsetInAtom());
            serWaveOp.rOfmapTileHeight(matmulWaveOp->gOfmapTileHeight());
            serWaveOp.rOfmapTileWidth(matmulWaveOp->gOfmapTileWidth());
            serWaveOp.rPsumBankId(matmulWaveOp->gPsumBankId());
            serWaveOp.rPsumBankOffset(matmulWaveOp->gPsumBankOffset());
            serWaveOp.rStart(matmulWaveOp->qStart());
            serWaveOp.rWaveIdFormat(matmulWaveOp->gWaveIdFormat());
            serWaveOp.rWeightsAtomId(matmulWaveOp->gWeightsAtomId());
            serWaveOp.rWeightsOffsetInAtom(matmulWaveOp->gWeightsOffsetInAtom());
            serWaveOp.rWaveId(matmulWaveOp->gWaveId());
            continue;
        }
        assert(false && "Unsupported WaveOp");
    }
    archive(cereal::make_nvp(NetKey_WaveOps, serWaveOps));
}

//--------------------------------------------------------
template<>
void
Network::load<cereal::JSONInputArchive>(cereal::JSONInputArchive& archive)
{
    archive(cereal::make_nvp(NetKey_NetName, m_Name));
    std::string dataType;
    archive(cereal::make_nvp(NetKey_DataType, dataType));

    if (dataType == DataTypeUint8::gNameStatic()) {
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
    archive(cereal::make_nvp(NetKey_Layers, serLayers));
    kcc::utils::breakFunc(333);
    for (unsigned i = 0; i < serLayers.size(); ++i) {
        const serialize::SerLayer& serLayer(serLayers[i]);
        layers::Layer::Params params;
        params.m_LayerName = serLayer.gName();
        params.m_BatchFactor = serLayer.gBatchFactor();
        params.m_Network = this;
        params.m_RefFile = serLayer.gRefFile();
        params.m_RefFileFormat = serLayer.gOfmapFormat();

        FmapDesc fmap_desc(serLayer.gNumOfmaps(), serLayer.gOfmapHeight(), serLayer.gOfmapWidth());
        layers::Layer* layer = nullptr;
        if (serLayer.gTypeStr() == LayerTypeStr_Input) {
            assert(serLayer.gNumPrevLayers() == 0 && "Input layer should have zero inputs");
            layer = new layers::InputLayer(params, fmap_desc);
        } else if (serLayer.gTypeStr() == LayerTypeStr_Const) {
            assert(serLayer.gNumPrevLayers() == 0 && "Const layer should have zero inputs");
            layer = new layers::ConstLayer(params, fmap_desc);
        } else if (serLayer.gTypeStr() == LayerTypeStr_Conv) {
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

        } else if (serLayer.gTypeStr() == LayerTypeStr_MaxPool || serLayer.gTypeStr() == LayerTypeStr_AvgPool) {
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

            if (serLayer.gTypeStr() == LayerTypeStr_MaxPool) {
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

        } else if (serLayer.gTypeStr() == LayerTypeStr_Relu) {
            assert(serLayer.gNumPrevLayers() == 1 && "Relu layer: number of inputs not 1");
            const std::string& prevLayerName = serLayer.gPrevLayer(0);
            layers::Layer* prevLayer = findLayer(prevLayerName);
            assert(prevLayer != nullptr && "Relu: Unknown previous layer");
            layer = new layers::ReluLayer(params, prevLayer);
        } else if (serLayer.gTypeStr() == LayerTypeStr_Tanh) {
            assert(serLayer.gNumPrevLayers() == 1 && "Tanh layer: number of inputs not 1");
            const std::string& prevLayerName = serLayer.gPrevLayer(0);
            layers::Layer* prevLayer = findLayer(prevLayerName);
            assert(prevLayer != nullptr && "Tanh: Unknown previous layer");
            layer = new layers::TanhLayer(params, prevLayer);

        } else if (serLayer.gTypeStr() == LayerTypeStr_ResAdd) {
            // TODO: check dimensions and types of inputs
            assert(serLayer.gNumPrevLayers() == 2 && "ResAdd layer should have two inputs");
            std::vector<layers::Layer*> prevLayers;
            for (const auto& prevLayerName : serLayer.gPrevLayers()) {
                layers::Layer* prevLayer = findLayer(prevLayerName);
                assert(prevLayer != nullptr && "RadAdd: Unknown previous layer");
                prevLayers.push_back(prevLayer);
            }
            layer = new layers::ResAddLayer(params, fmap_desc,prevLayers);
        } else if (serLayer.gTypeStr() == LayerTypeStr_BiasAdd) {
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

        assert(m_Name2Layer.find(params.m_LayerName) == m_Name2Layer.end());
        m_Name2Layer[params.m_LayerName] = layer;
    }
    assert(m_Layers.size() == serLayers.size() && "Layer mismatch count after input deserialization" );


    //===========================================================================
    std::vector<serialize::SerWaveOp> serWaveOps;
    archive(cereal::make_nvp(NetKey_WaveOps, serWaveOps));
    for (unsigned i = 0; i < serWaveOps.size(); ++i) {
        const serialize::SerWaveOp& serWaveOp(serWaveOps[i]);

        std::vector<wave::WaveOp*> prevWaveOps;
        wave::WaveOp* waveOp = nullptr;

        auto fillWaveOpParams = [this, &prevWaveOps](
                                    const serialize::SerWaveOp& serWaveOp,
                                    wave::WaveOp::Params& waveOpParams) -> void
        {
            waveOpParams.m_WaveOpName   = serWaveOp.gWaveOpName();
            waveOpParams.m_Layer        = this->findLayer(serWaveOp.gLayerName());
            assert(waveOpParams.m_Layer);
            for (const auto& prevWaveOpName : serWaveOp.gPreviousWaveOps()) {
                prevWaveOps.push_back(findWaveOp(prevWaveOpName));
            }
        };


        if (serWaveOp.gWaveOpType() == wave::SbAtomFileWaveOp::gTypeStr()) {
            wave::SbAtomFileWaveOp::Params sbatomfileParams;
            fillWaveOpParams(serWaveOp, sbatomfileParams);
            sbatomfileParams.m_RefFileName       = serWaveOp.gRefFile();
            sbatomfileParams.m_BatchFoldIdx      = serWaveOp.gBatchFoldIdx();
            sbatomfileParams.m_AtomId            = serWaveOp.gAtomId();
            sbatomfileParams.m_Length            = serWaveOp.gLength();
            sbatomfileParams.m_OffsetInFile      = serWaveOp.gOffsetInFile();

            sbatomfileParams.m_IfmapsFoldIdx     = serWaveOp.gIfmapsFoldIdx();
            sbatomfileParams.m_IfmapsReplicate   = serWaveOp.qIfmapsReplicate();

            waveOp = new wave::SbAtomFileWaveOp(sbatomfileParams, prevWaveOps);
            assert(waveOp->gName() == sbatomfileParams.m_WaveOpName);

        } else if (serWaveOp.gWaveOpType() == wave::SbAtomSaveWaveOp::gTypeStr()) {
            wave::SbAtomSaveWaveOp::Params sbatomsaveParams;
            fillWaveOpParams(serWaveOp, sbatomsaveParams);
            sbatomsaveParams.m_RefFileName       = serWaveOp.gRefFile();
            sbatomsaveParams.m_BatchFoldIdx      = serWaveOp.gBatchFoldIdx();
            sbatomsaveParams.m_AtomId            = serWaveOp.gAtomId();
            sbatomsaveParams.m_Length            = serWaveOp.gLength();
            sbatomsaveParams.m_OffsetInFile      = serWaveOp.gOffsetInFile();

            sbatomsaveParams.m_OfmapsFoldIdx     = serWaveOp.gOfmapsFoldIdx();

            waveOp = new wave::SbAtomSaveWaveOp(sbatomsaveParams, prevWaveOps);
            assert(waveOp->gName() == sbatomsaveParams.m_WaveOpName);

        } else if (serWaveOp.gWaveOpType() == wave::MatMulWaveOp::gTypeStr()) {
            wave::MatMulWaveOp::Params matmulParams;
            fillWaveOpParams(serWaveOp, matmulParams);

            matmulParams.m_IfmapTileHeight      = serWaveOp.gIfmapTileHeight();
            matmulParams.m_IfmapTileWidth      = serWaveOp.gIfmapTileWidth();
            matmulParams.m_IfmapsAtomId         = serWaveOp.gIfmapsAtomId();
            matmulParams.m_IfmapsOffsetInAtom   = serWaveOp.gIfmapsOffsetInAtom();
            matmulParams.m_OfmapTileHeight      = serWaveOp.gOfmapTileHeight();
            matmulParams.m_OfmapTileWidth      = serWaveOp.gOfmapTileWidth();
            matmulParams.m_PsumBankId           = serWaveOp.gPsumBankId();
            matmulParams.m_PsumBankOffset           = serWaveOp.gPsumBankOffset();
            matmulParams.m_Start                = serWaveOp.qStart();
            matmulParams.m_WaveIdFormat         = serWaveOp.gWaveIdFormat();
            matmulParams.m_WeightsAtomId        = serWaveOp.gWeightsAtomId();
            matmulParams.m_WeightsOffsetInAtom  = serWaveOp.gWeightsOffsetInAtom();
            matmulParams.m_WaveId               = serWaveOp.gWaveId();

            waveOp = new wave::MatMulWaveOp(matmulParams, prevWaveOps);
            assert(waveOp->gName() == matmulParams.m_WaveOpName);

        } else {
            assert(false && "Wrong WaveOp type during deserialization");
        }

        m_WaveOps.push_back(waveOp);
        assert(m_Name2WaveOp.find(waveOp->gName()) == m_Name2WaveOp.end());
        m_Name2WaveOp[waveOp->gName()] = waveOp;

    }
}

//--------------------------------------------------------
layers::Layer*
Network::findLayer(const std::string& layerName)
{
    layers::Layer* layer = m_Name2Layer[layerName];
    assert(layer && "Could not find layer");
    return layer;
}

wave::WaveOp*
Network::findWaveOp(const std::string& waveOpName)
{
    wave::WaveOp* waveOp = m_Name2WaveOp[waveOpName];
    assert(waveOp && "Could not find WaveOp");
    return waveOp;
}

}}


