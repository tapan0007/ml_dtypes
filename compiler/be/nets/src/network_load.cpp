#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <map>

#include <cereal/types/memory.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/map.hpp>


#include "utils/inc/asserter.hpp"
#include "utils/inc/consts.hpp"
#include "arch/inc/arch.hpp"

#include "layers/inc/layer.hpp"
#include "layers/inc/inputlayer.hpp"
#include "layers/inc/constlayer.hpp"
#include "layers/inc/convlayer.hpp"
#include "layers/inc/matmullayer.hpp"
#include "layers/inc/reshapelayer.hpp"
#include "layers/inc/relulayer.hpp"
#include "layers/inc/tanhlayer.hpp"
#include "layers/inc/maxpoollayer.hpp"
#include "layers/inc/avgpoollayer.hpp"
#include "layers/inc/resaddlayer.hpp"
#include "layers/inc/biasaddlayer.hpp"

#include "nets/inc/network_load.hpp"

#include "wave/inc/sbatomloadwaveop.hpp"
#include "wave/inc/sbatomsavewaveop.hpp"

#include "wave/inc/matmulwaveop.hpp"
#include "wave/inc/poolwaveop.hpp"
#include "wave/inc/activationwaveop.hpp"
#include "wave/inc/resaddwaveop.hpp"

#include "serialize/inc/serlayer.hpp"
#include "serialize/inc/serwaveop.hpp"

#define KCC_UNSERIALIZE(X) PARAMS.KCC_CONCAT(m_,X) = serWaveOp.KCC_CONCAT(m_,X);
namespace kcc {


#define ASSERT_NUM_LAYERS(serLayer, N) \
    Assert((serLayer).gNumPrevLayers() == (N), (serLayer).gTypeStr(), " layer '", (serLayer).gLayerName(), \
                   "' should have ", (N), " input", ((N)==1 ? "" : "s"), ", but it has ", (serLayer).gNumPrevLayers())

#define ASSERT_PREV_LAYER(prevLayer, serLayer, prevLayerName) \
    Assert((prevLayer) != nullptr, (serLayer).gTypeStr(), " layer '", (serLayer).gLayerName(), \
                       "': Previous layer '", (prevLayerName), "' not found");

namespace nets {

//--------------------------------------------------------
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
        Assert(false, "Unsupported data type ", dataType);
    }

    std::vector<serialize::SerLayer> serLayers;
    archive(cereal::make_nvp(NetKey_Layers, serLayers));

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
            ASSERT_NUM_LAYERS(serLayer, 0);
            layer = new layers::InputLayer(params, fmap_desc);
        } else if (serLayer.gTypeStr() == LayerTypeStr_Const) {
            ASSERT_NUM_LAYERS(serLayer, 0);
            layer = new layers::ConstLayer(params, fmap_desc);

        } else if (serLayer.gTypeStr() == LayerTypeStr_Conv) {
            ASSERT_NUM_LAYERS(serLayer, 1);

            const std::string& prevLayerName = serLayer.gPrevLayer(0);
            layers::Layer* prevLayer = findLayer(prevLayerName);
            ASSERT_PREV_LAYER(prevLayer, serLayer, prevLayerName);
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
            layers::ConvLayer::Params convParams(params);
            /*
            convParams.m_BatchingInWave    = serLayer.gBatchingInWave();
            */

            layer = new layers::ConvLayer(convParams, prevLayer,
                                          fmap_desc,
                                          stride, kernel, padding,
                                          filterFileName.c_str(),
                                          filterTensorDimSemantics.c_str());

        } else if (serLayer.gTypeStr() == LayerTypeStr_Matmul) {
            ASSERT_NUM_LAYERS(serLayer, 1);
            const std::string& prevLayerName = serLayer.gPrevLayer(0);
            layers::Layer* prevLayer = findLayer(prevLayerName);
            ASSERT_PREV_LAYER(prevLayer, serLayer, prevLayerName);
            std::tuple<kcc_int32,kcc_int32> kernel = std::make_tuple(
                                            serLayer.gConvFilterHeight(),
                                            serLayer.gConvFilterWidth());

            const std::string filterFileName = serLayer.gKernelFile();
            const std::string filterTensorDimSemantics = serLayer.gKernelFormat();
            layers::MatmulLayer::Params matmulParams(params);

            layer = new layers::MatmulLayer(matmulParams, prevLayer,
                                          fmap_desc,
                                          kernel,
                                          filterFileName.c_str(),
                                          filterTensorDimSemantics.c_str());

        } else if (serLayer.gTypeStr() == LayerTypeStr_Reshape) {
            ASSERT_NUM_LAYERS(serLayer, 1);

            const std::string& prevLayerName = serLayer.gPrevLayer(0);
            layers::Layer* prevLayer = findLayer(prevLayerName);
            ASSERT_PREV_LAYER(prevLayer, serLayer, prevLayerName);

            layers::ReshapeLayer::Params reshapeParams(params);

            layer = new layers::ReshapeLayer(reshapeParams, prevLayer, fmap_desc);

        } else if (serLayer.gTypeStr() == LayerTypeStr_MaxPool || serLayer.gTypeStr() == LayerTypeStr_AvgPool) {
            ASSERT_NUM_LAYERS(serLayer, 1);
            const std::string& prevLayerName = serLayer.gPrevLayer(0);
            layers::Layer* prevLayer = findLayer(prevLayerName);
            ASSERT_PREV_LAYER(prevLayer, serLayer, prevLayerName);

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
            ASSERT_NUM_LAYERS(serLayer, 1);
            const std::string& prevLayerName = serLayer.gPrevLayer(0);
            layers::Layer* prevLayer = findLayer(prevLayerName);
            ASSERT_PREV_LAYER(prevLayer, serLayer, prevLayerName);
            layer = new layers::ReluLayer(params, prevLayer);
        } else if (serLayer.gTypeStr() == LayerTypeStr_Tanh) {
            ASSERT_NUM_LAYERS(serLayer, 1);
            const std::string& prevLayerName = serLayer.gPrevLayer(0);
            layers::Layer* prevLayer = findLayer(prevLayerName);
            ASSERT_PREV_LAYER(prevLayer, serLayer, prevLayerName);
            layer = new layers::TanhLayer(params, prevLayer);
        } else if (serLayer.gTypeStr() == LayerTypeStr_ResAdd || serLayer.gTypeStr() == LayerTypeStr_Multiply) {
            // TODO: check dimensions and types of inputs
            ASSERT_NUM_LAYERS(serLayer, 2);
            std::vector<layers::Layer*> prevLayers;
            for (const auto& prevLayerName : serLayer.gPrevLayers()) {
                layers::Layer* prevLayer = findLayer(prevLayerName);
                ASSERT_PREV_LAYER(prevLayer, serLayer, prevLayerName);
                prevLayers.push_back(prevLayer);
            }
            layer = new layers::ResAddLayer(params, fmap_desc,prevLayers);
        } else if (serLayer.gTypeStr() == LayerTypeStr_BiasAdd) {
            // TODO check dimensions and types of inputs
            ASSERT_NUM_LAYERS(serLayer, 2);
            std::vector<layers::Layer*> prevLayers;
            for (const auto& prevLayerName : serLayer.gPrevLayers()) {
                layers::Layer* prevLayer = findLayer(prevLayerName);
                ASSERT_PREV_LAYER(prevLayer, serLayer, prevLayerName);
                prevLayers.push_back(prevLayer);
            }
            layer = new layers::BiasAddLayer(params, fmap_desc, prevLayers);
        } else if (serLayer.gTypeStr() == LayerTypeStr_StridedSlice 
                || serLayer.gTypeStr() == LayerTypeStr_Unstack 
                || serLayer.gTypeStr() == LayerTypeStr_Sigmoid) {   // FIXME: placeholder
            ASSERT_NUM_LAYERS(serLayer, 1);
            const std::string& prevLayerName = serLayer.gPrevLayer(0);
            layers::Layer* prevLayer = findLayer(prevLayerName);
            ASSERT_PREV_LAYER(prevLayer, serLayer, prevLayerName);
            layer = new layers::TanhLayer(params, prevLayer);   // FIXME: placeholder
        } else {
            Assert(false, "Unsuported layer type ", serLayer.gTypeStr());
        }

        Assert(m_Name2Layer.find(params.m_LayerName) == m_Name2Layer.end(),
               "Layer ", params.m_LayerName, " already exists");
        m_Name2Layer[params.m_LayerName] = layer;
    }
    Assert(m_Layers.size() == serLayers.size(),
        "Layer mismatch count after input deserialization: ", m_Layers.size(),
        " != ", serLayers.size());


    //===========================================================================
    if (m_UseWave) {
        std::vector<serialize::SerWaveOp> serWaveOps;
        archive(cereal::make_nvp(NetKey_WaveOps, serWaveOps));
        for (unsigned i = 0; i < serWaveOps.size(); ++i) {
            const serialize::SerWaveOp& serWaveOp(serWaveOps[i]);

            wave::WaveOp* waveOp = nullptr;

            if (serWaveOp.m_WaveOpType == wave::SbAtomLoadWaveOp::gTypeStrStatic()) {
                waveOp = m_Load->loadSbAtomLoad(serWaveOp);
            } else if (serWaveOp.m_WaveOpType == wave::SbAtomSaveWaveOp::gTypeStrStatic()) {
                waveOp = m_Load->loadSbAtomSave(serWaveOp);
            } else if (serWaveOp.m_WaveOpType == wave::PoolWaveOp::gTypeStrStatic()) {
                waveOp = m_Load->loadPool(serWaveOp);
            } else if (serWaveOp.m_WaveOpType == wave::MatMulWaveOp::gTypeStrStatic()) {
                waveOp = m_Load->loadMatMul(serWaveOp);
            } else if (serWaveOp.m_WaveOpType == wave::ActivationWaveOp::gTypeStrStatic()) {
                waveOp = m_Load->loadActivation(serWaveOp);
            } else if (serWaveOp.m_WaveOpType == wave::ResAddWaveOp::gTypeStrStatic()) {
                waveOp = m_Load->loadResAdd(serWaveOp);
            } else {
                Assert(false, "Wrong WaveOp type during deserialization: ", serWaveOp.m_WaveOpType);
            }

            m_WaveOps.push_back(waveOp);
            Assert(m_Name2WaveOp.find(waveOp->gName()) == m_Name2WaveOp.end(),
                   "Waveop ", waveOp->gName(), " already exists");
            m_Name2WaveOp[waveOp->gName()] = waveOp;
        }
    }
}




Network::Load::Load(Network& network)
    : m_Network(network)
{
}



wave::SbAtomLoadWaveOp*
Network::Load::loadSbAtomLoad(const serialize::SerWaveOp& serWaveOp)
{
#define PARAMS sbatomLoadParams
    std::vector<wave::WaveOp*> prevWaveOps;
    wave::SbAtomLoadWaveOp::Params sbatomLoadParams;
    fillWaveOpParams(serWaveOp, prevWaveOps, sbatomLoadParams);

    KCC_UNSERIALIZE(SbAddress);
    KCC_UNSERIALIZE(BatchFoldIdx);
    sbatomLoadParams.m_DataType = DataType::dataTypeStr2Id(serWaveOp.m_DataType);
    KCC_UNSERIALIZE(Length);
    KCC_UNSERIALIZE(OffsetInFile);
    KCC_UNSERIALIZE(PartitionStepBytes);
    sbatomLoadParams.m_RefFileName = serWaveOp.m_RefFile;
    KCC_UNSERIALIZE(RefFileFormat);
    for (unsigned int i = 0; i < sbatomLoadParams.m_RefFileShape.size(); ++i) {
        sbatomLoadParams.m_RefFileShape[i] = serWaveOp.m_RefFileShape[i];
    }

    KCC_UNSERIALIZE(IfmapCount);
    KCC_UNSERIALIZE(IfmapsFoldIdx);
    KCC_UNSERIALIZE(IfmapsReplicate);
    KCC_UNSERIALIZE(ContainWeights);

    auto waveOp = new wave::SbAtomLoadWaveOp(sbatomLoadParams, prevWaveOps);
    Assert(waveOp && waveOp->gName() == sbatomLoadParams.m_WaveOpName,
           "Wrong wave op name: should be ", sbatomLoadParams.m_WaveOpName,
           ", it is ", waveOp->gName());
    return waveOp;
#undef PARAMS
}


wave::SbAtomSaveWaveOp*
Network::Load::loadSbAtomSave(const serialize::SerWaveOp& serWaveOp)
{
#define PARAMS sbatomsaveParams
    std::vector<wave::WaveOp*> prevWaveOps;
    wave::SbAtomSaveWaveOp::Params sbatomsaveParams;
    fillWaveOpParams(serWaveOp, prevWaveOps, sbatomsaveParams);

    KCC_UNSERIALIZE(SbAddress);
    KCC_UNSERIALIZE(BatchFoldIdx);
    sbatomsaveParams.m_DataType = DataType::dataTypeStr2Id(serWaveOp.m_DataType);
    KCC_UNSERIALIZE(Length);
    KCC_UNSERIALIZE(OffsetInFile);
    KCC_UNSERIALIZE(PartitionStepBytes);
    sbatomsaveParams.m_RefFileName = serWaveOp.m_RefFile;
    KCC_UNSERIALIZE(RefFileFormat);
    for (unsigned int i = 0; i < sbatomsaveParams.m_RefFileShape.size(); ++i) {
        sbatomsaveParams.m_RefFileShape[i] = serWaveOp.m_RefFileShape[i];
    }

    KCC_UNSERIALIZE(OfmapCount);
    KCC_UNSERIALIZE(OfmapsFoldIdx);

    auto waveOp = new wave::SbAtomSaveWaveOp(sbatomsaveParams, prevWaveOps);
    Assert(waveOp && waveOp->gName() == sbatomsaveParams.m_WaveOpName,
           "Wrong wave op name: should be ", sbatomsaveParams.m_WaveOpName,
           ", it is ", waveOp->gName());
    return waveOp;
#undef PARAMS
}

wave::PoolWaveOp*
Network::Load::loadPool(const serialize::SerWaveOp& serWaveOp)
{
#define PARAMS poolParams
    std::vector<wave::WaveOp*> prevWaveOps;
    wave::PoolWaveOp::Params poolParams;
    fillWaveOpParams(serWaveOp, prevWaveOps, poolParams);

    KCC_UNSERIALIZE(DstSbAddress);
    KCC_UNSERIALIZE(DstXNum);
    KCC_UNSERIALIZE(DstXStep);
    KCC_UNSERIALIZE(DstYNum);
    KCC_UNSERIALIZE(DstYStep);
    KCC_UNSERIALIZE(DstZNum);
    KCC_UNSERIALIZE(DstZStep);
    poolParams.m_InDtypeId  = DataType::dataTypeStr2Id(serWaveOp.m_InDtype);
    KCC_UNSERIALIZE(NumPartitions);
    poolParams.m_OutDtypeId = DataType::dataTypeStr2Id(serWaveOp.m_OutDtype);
    KCC_UNSERIALIZE(PoolFrequency);
    poolParams.m_PoolFunc  = utils::poolTypeStr2Id(serWaveOp.m_PoolFunc);

    KCC_UNSERIALIZE(SrcIsPsum);
    if (serWaveOp.m_SrcIsPsum) {
        KCC_UNSERIALIZE(SrcPsumBankId);
        KCC_UNSERIALIZE(SrcPsumBankOffset);
    } else {
        KCC_UNSERIALIZE(SrcSbAddress);
    }

    KCC_UNSERIALIZE(SrcWNum);
    KCC_UNSERIALIZE(SrcWStep);
    KCC_UNSERIALIZE(SrcXNum);
    KCC_UNSERIALIZE(SrcXStep);
    KCC_UNSERIALIZE(SrcYNum);
    KCC_UNSERIALIZE(SrcYStep);
    KCC_UNSERIALIZE(SrcZNum);
    KCC_UNSERIALIZE(SrcZStep);

    Assert(poolParams.m_TileId.size() == serWaveOp.m_TileId.size(),
        serWaveOp.m_WaveOpType, " waveop '", serWaveOp.m_WaveOpName,
        "' has wrong tile id size: ", poolParams.m_TileId.size());
    for (unsigned int i = 0; i < serWaveOp.m_TileId.size(); ++i) {
        poolParams.m_TileId[i] = serWaveOp.m_TileId[i];
    }
    poolParams.m_TileIdFormat           = serWaveOp.m_TileIdFormat;

    auto waveOp = new wave::PoolWaveOp(poolParams, prevWaveOps);
    Assert(waveOp->gName() == poolParams.m_WaveOpName, "Wrong waveop name ", waveOp->gName());
    return waveOp;
#undef PARAMS
}

wave::MatMulWaveOp*
Network::Load::loadMatMul(const serialize::SerWaveOp& serWaveOp)
{
#define PARAMS matmulParams
    std::vector<wave::WaveOp*> prevWaveOps;
    wave::MatMulWaveOp::Params matmulParams;
    fillWaveOpParams(serWaveOp, prevWaveOps, matmulParams);

    KCC_UNSERIALIZE(BatchingInWave);
    KCC_UNSERIALIZE(FmapXNum);
    KCC_UNSERIALIZE(FmapXStep);
    KCC_UNSERIALIZE(FmapYNum);
    KCC_UNSERIALIZE(FmapYStep);
    KCC_UNSERIALIZE(FmapZNum);
    KCC_UNSERIALIZE(FmapZStepAtoms);
    KCC_UNSERIALIZE(IfmapCount);
    KCC_UNSERIALIZE(IfmapTileHeight);
    KCC_UNSERIALIZE(IfmapTileWidth);
    KCC_UNSERIALIZE(IfmapsSbAddress);
    matmulParams.m_InDtypeId = DataType::dataTypeStr2Id(serWaveOp.m_InDtype);
    // layer_name
    KCC_UNSERIALIZE(NumColumnPartitions);
    KCC_UNSERIALIZE(NumRowPartitions);
    KCC_UNSERIALIZE(OfmapCount);
    KCC_UNSERIALIZE(OfmapTileHeight);
    KCC_UNSERIALIZE(OfmapTileWidth);
    matmulParams.m_OutDtypeId = DataType::dataTypeStr2Id(serWaveOp.m_OutDtype);
    // previous layers
    KCC_UNSERIALIZE(PsumBankId);
    KCC_UNSERIALIZE(PsumBankOffset);
    KCC_UNSERIALIZE(PsumXNum);
    KCC_UNSERIALIZE(PsumXStep);
    KCC_UNSERIALIZE(PsumYNum);
    KCC_UNSERIALIZE(PsumYStep);

    KCC_UNSERIALIZE(StartTensorCalc);
    KCC_UNSERIALIZE(StopTensorCalc);
    KCC_UNSERIALIZE(StrideX);
    KCC_UNSERIALIZE(StrideY);

    KCC_UNSERIALIZE(WaveId);
    KCC_UNSERIALIZE(WaveIdFormat);
    // waveop name
    // waveop type
    KCC_UNSERIALIZE(WeightsSbAddress);

    auto waveOp = new wave::MatMulWaveOp(matmulParams, prevWaveOps);
    Assert(waveOp->gName() == matmulParams.m_WaveOpName, "Wrong waveop name ", waveOp->gName());
    return waveOp;
#undef PARAMS
}

wave::ActivationWaveOp*
Network::Load::loadActivation(const serialize::SerWaveOp& serWaveOp)
{
#define PARAMS activationParams
    std::vector<wave::WaveOp*> prevWaveOps;
    wave::ActivationWaveOp::Params activationParams;
    fillWaveOpParams(serWaveOp, prevWaveOps, activationParams);

    activationParams.m_ActivationFunc   = serialize::SerWaveOp::str2ActivationFunc(serWaveOp.m_ActivationFunc);

    KCC_UNSERIALIZE(BiasAddEn);
    KCC_UNSERIALIZE(BiasSbAddress);

    KCC_UNSERIALIZE(DstXNum);
    KCC_UNSERIALIZE(DstXStep);
    KCC_UNSERIALIZE(DstYNum);
    KCC_UNSERIALIZE(DstYStep);
    KCC_UNSERIALIZE(DstZNum);
    KCC_UNSERIALIZE(DstZStep);

    KCC_UNSERIALIZE(DstIsPsum);
    if (serWaveOp.m_DstIsPsum) {
        KCC_UNSERIALIZE(DstPsumBankId);
    } else {
        KCC_UNSERIALIZE(DstSbAddress);
    }

    activationParams.m_InDtypeId        = DataType::dataTypeStr2Id(serWaveOp.m_InDtype);
    activationParams.m_BiasDtypeId      = DataType::dataTypeStr2Id(serWaveOp.m_BiasDtype);
    KCC_UNSERIALIZE(NumPartitions);
    activationParams.m_OutDtypeId       = DataType::dataTypeStr2Id(serWaveOp.m_OutDtype);
    KCC_UNSERIALIZE(SrcPsumBankId);
    KCC_UNSERIALIZE(SrcXNum);
    KCC_UNSERIALIZE(SrcXStep);
    KCC_UNSERIALIZE(SrcYNum);
    KCC_UNSERIALIZE(SrcYStep);
    KCC_UNSERIALIZE(SrcZNum);
    KCC_UNSERIALIZE(SrcZStep);

    Assert(activationParams.m_TileId.size() == serWaveOp.m_TileId.size(),
        serWaveOp.m_WaveOpType, " waveop '", serWaveOp.m_WaveOpName,
        "' has wrong tile id size: ", activationParams.m_TileId.size());
    for (unsigned int i = 0; i < serWaveOp.m_TileId.size(); ++i) {
        activationParams.m_TileId[i] = serWaveOp.m_TileId[i];
    }
    KCC_UNSERIALIZE(TileIdFormat);

    auto waveOp = new wave::ActivationWaveOp(activationParams, prevWaveOps);
    Assert(waveOp->gName() == activationParams.m_WaveOpName, "Wrong waveop name ", waveOp->gName());
    return waveOp;
#undef PARAMS
}


wave::ResAddWaveOp*
Network::Load::loadResAdd(const serialize::SerWaveOp& serWaveOp)
{
#define PARAMS resAddParams
    std::vector<wave::WaveOp*> prevWaveOps;
    wave::ResAddWaveOp::Params resAddParams;
    fillWaveOpParams(serWaveOp, prevWaveOps, resAddParams);

    resAddParams.m_InADtypeId        = DataType::dataTypeStr2Id(serWaveOp.m_InADtype);
    resAddParams.m_InBDtypeId        = DataType::dataTypeStr2Id(serWaveOp.m_InBDtype);
    resAddParams.m_OutDtypeId       = DataType::dataTypeStr2Id(serWaveOp.m_OutDtype);
    KCC_UNSERIALIZE(NumPartitions);
    KCC_UNSERIALIZE(Multiply);       /* Hack in ResAdd to get Multiply to work with old ISA */

    // SrcA
    KCC_UNSERIALIZE(SrcAIsPsum);
    if (serWaveOp.m_SrcAIsPsum) {
        KCC_UNSERIALIZE(SrcAPsumBankId);
        KCC_UNSERIALIZE(SrcAPsumBankOffset);
    } else {
        KCC_UNSERIALIZE(SrcASbAddress);
    }
    KCC_UNSERIALIZE(SrcAXNum);
    KCC_UNSERIALIZE(SrcAXStep);
    KCC_UNSERIALIZE(SrcAYNum);
    KCC_UNSERIALIZE(SrcAYStep);
    KCC_UNSERIALIZE(SrcAZNum);
    KCC_UNSERIALIZE(SrcAZStep);

    // SrcB
    KCC_UNSERIALIZE(SrcBIsPsum);
    if (serWaveOp.m_SrcBIsPsum) {
        KCC_UNSERIALIZE(SrcBPsumBankId);
        KCC_UNSERIALIZE(SrcBPsumBankOffset);
    } else {
        KCC_UNSERIALIZE(SrcBSbAddress);
    }
    KCC_UNSERIALIZE(SrcBXNum);
    KCC_UNSERIALIZE(SrcBXStep);
    KCC_UNSERIALIZE(SrcBYNum);
    KCC_UNSERIALIZE(SrcBYStep);
    KCC_UNSERIALIZE(SrcBZNum);
    KCC_UNSERIALIZE(SrcBZStep);

    // Dst
    KCC_UNSERIALIZE(DstIsPsum);
    if (serWaveOp.m_DstIsPsum) {
        KCC_UNSERIALIZE(DstPsumBankId);
        KCC_UNSERIALIZE(DstPsumBankOffset);
    } else {
        KCC_UNSERIALIZE(DstSbAddress);
    }
    KCC_UNSERIALIZE(DstXNum);
    KCC_UNSERIALIZE(DstXStep);
    KCC_UNSERIALIZE(DstYNum);
    KCC_UNSERIALIZE(DstYStep);
    KCC_UNSERIALIZE(DstZNum);
    KCC_UNSERIALIZE(DstZStep);

    auto waveOp = new wave::ResAddWaveOp(resAddParams, prevWaveOps);
    Assert(waveOp->gName() == resAddParams.m_WaveOpName, "Wrong waveop name ", waveOp->gName());
    return waveOp;
}





/* in
 * template<>
 * void
 * Network::Load::load<cereal::JSONInputArchive>(cereal::JSONInputArchive& archive)
 * {
 *      ...
 *      auto fillWaveOpParams = [this, &prevWaveOps](
 *                              const serialize::SerWaveOp& serWaveOp,
 *                              wave::WaveOp::Params& waveOpParams) -> void
 *      ...
 * }
 */

void
Network::Load::fillWaveOpParams(const serialize::SerWaveOp& serWaveOp,
                     std::vector<wave::WaveOp*>& prevWaveOps,
                     wave::WaveOp::Params& waveOpParams)
{
    waveOpParams.m_WaveOpName   = serWaveOp.m_WaveOpName;
    waveOpParams.m_Layer        = m_Network.findLayer(serWaveOp.m_LayerName);
    waveOpParams.m_Order        = m_Network.gWaveOps().size();
    Assert(waveOpParams.m_Layer, "Missing layer for waveop ", serWaveOp.m_WaveOpName);
    for (const auto& prevWaveOpName : serWaveOp.m_PreviousWaveOps) {
        prevWaveOps.push_back(m_Network.findWaveOp(prevWaveOpName));
    }
}

#undef KCC_UNSERIALIZE
#undef ASSERT_NUM_LAYERS
#undef ASSERT_PREV_LAYER

}}


