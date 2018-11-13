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

#include "layers/inc/layerconsts.hpp"
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
#include "wave/inc/clipbyvaluewaveop.hpp"
#include "wave/inc/tensortensorwaveop.hpp"
#include "wave/inc/tensorscalarconstwaveop.hpp"

#include "serialize/inc/serlayer.hpp"
#include "serialize/inc/serwaveop.hpp"



#define ASSERT_NUM_LAYERS(serLayer, N) \
    Assert((serLayer).gNumPrevLayers() == (N), (serLayer).gTypeStr(), " layer '", (serLayer).gLayerName(), \
                   "' should have ", (N), " input", ((N)==1 ? "" : "s"), ", but it has ", (serLayer).gNumPrevLayers())

#define ASSERT_NUM_LAYERS2(serLayer, N1, N2) \
    Assert( ((serLayer).gNumPrevLayers() == (N1) || (serLayer).gNumPrevLayers() == (N2) ), \
                   (serLayer).gTypeStr(), " layer '", (serLayer).gLayerName(), \
                   "' should have ", (N1), " or ", (N2), " inputs, but it has ", \
                   (serLayer).gNumPrevLayers())

#define ASSERT_PREV_LAYER(prevLayer, serLayer, prevLayerName) \
    Assert((prevLayer) != nullptr, (serLayer).gTypeStr(), " layer '", (serLayer).gLayerName(), \
                       "': Previous layer '", (prevLayerName), "' not found");

namespace kcc {
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

    } else if (dataType == DataTypeBFloat16::gNameStatic()) {
        m_DataType = std::make_unique<DataTypeBFloat16>();

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

        if (serLayer.gTypeStr() != layers::LayerTypeStr_Const) {
            Assert(serLayer.gNumOfmaps() > 0,
                   "Non const tensor must have more than on Fmap");
            Assert(serLayer.gOfmapHeight() > 0,
                   "Non const tensor height must be positive");
            Assert(serLayer.gOfmapWidth() > 0,
                   "Non const tensor height must be positive");
        }

        FmapDesc fmap_desc(serLayer.gNumOfmaps(), serLayer.gOfmapHeight(), serLayer.gOfmapWidth());

        layers::Layer* layer = nullptr;
        if (serLayer.gTypeStr() == layers::LayerTypeStr_Input) {
            ASSERT_NUM_LAYERS(serLayer, 0);
            layer = new layers::InputLayer(params, fmap_desc);
        } else if (serLayer.gTypeStr() == layers::LayerTypeStr_Const) {
            ASSERT_NUM_LAYERS(serLayer, 0);
            layer = new layers::ConstLayer(params, fmap_desc);

        } else if (serLayer.gTypeStr() == layers::LayerTypeStr_Conv) {
            ASSERT_NUM_LAYERS(serLayer, 1);

            const std::string& prevLayerName = serLayer.gPrevLayer(0);
            layers::Layer* prevLayer = findLayer(prevLayerName, true);
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

        } else if (serLayer.gTypeStr() == layers::LayerTypeStr_Matmul) {
            ASSERT_NUM_LAYERS(serLayer, 1);
            const std::string& prevLayerName = serLayer.gPrevLayer(0);
            layers::Layer* prevLayer = findLayer(prevLayerName, true);
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

        } else if (serLayer.gTypeStr() == layers::LayerTypeStr_Reshape) {
            ASSERT_NUM_LAYERS(serLayer, 1);

            const std::string& prevLayerName = serLayer.gPrevLayer(0);
            layers::Layer* prevLayer = findLayer(prevLayerName, true);
            ASSERT_PREV_LAYER(prevLayer, serLayer, prevLayerName);

            layers::ReshapeLayer::Params reshapeParams(params);

            layer = new layers::ReshapeLayer(reshapeParams, prevLayer, fmap_desc);

        } else if (serLayer.gTypeStr() == layers::LayerTypeStr_MaxPool || serLayer.gTypeStr() == layers::LayerTypeStr_AvgPool) {
            ASSERT_NUM_LAYERS(serLayer, 1);
            const std::string& prevLayerName = serLayer.gPrevLayer(0);
            layers::Layer* prevLayer = findLayer(prevLayerName, true);
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

            if (serLayer.gTypeStr() == layers::LayerTypeStr_MaxPool) {
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

        } else if (serLayer.gTypeStr() == layers::LayerTypeStr_Relu) {
            ASSERT_NUM_LAYERS(serLayer, 1);
            const std::string& prevLayerName = serLayer.gPrevLayer(0);
            layers::Layer* prevLayer = findLayer(prevLayerName, true);
            ASSERT_PREV_LAYER(prevLayer, serLayer, prevLayerName);
            layer = new layers::ReluLayer(params, prevLayer);
        } else if (serLayer.gTypeStr() == layers::LayerTypeStr_Tanh) {
            ASSERT_NUM_LAYERS(serLayer, 1);
            const std::string& prevLayerName = serLayer.gPrevLayer(0);
            layers::Layer* prevLayer = findLayer(prevLayerName, true);
            ASSERT_PREV_LAYER(prevLayer, serLayer, prevLayerName);
            layer = new layers::TanhLayer(params, prevLayer);
        } else if (serLayer.gTypeStr() == layers::LayerTypeStr_ResAdd
                   || serLayer.gTypeStr() == layers::LayerTypeStr_Multiply
                   || serLayer.gTypeStr() == layers::LayerTypeStr_Sub
                   || serLayer.gTypeStr() == layers::LayerTypeStr_Add) {
            // TODO: check dimensions and types of inputs
            if (serLayer.gTypeStr() == layers::LayerTypeStr_ResAdd) {
                ASSERT_NUM_LAYERS(serLayer, 2);
            } else {
                ASSERT_NUM_LAYERS2(serLayer, 1, 2);
            }
            std::vector<layers::Layer*> prevLayers;
            for (const auto& prevLayerName : serLayer.gPrevLayers()) {
                layers::Layer* prevLayer = findLayer(prevLayerName, true);
                ASSERT_PREV_LAYER(prevLayer, serLayer, prevLayerName);
                prevLayers.push_back(prevLayer);
            }
            layer = new layers::ResAddLayer(params, fmap_desc,prevLayers);
        } else if (serLayer.gTypeStr() == layers::LayerTypeStr_BiasAdd) {
            // TODO check dimensions and types of inputs
            ASSERT_NUM_LAYERS(serLayer, 2);
            std::vector<layers::Layer*> prevLayers;
            for (const auto& prevLayerName : serLayer.gPrevLayers()) {
                layers::Layer* prevLayer = findLayer(prevLayerName, true);
                ASSERT_PREV_LAYER(prevLayer, serLayer, prevLayerName);
                prevLayers.push_back(prevLayer);
            }
            layer = new layers::BiasAddLayer(params, fmap_desc, prevLayers);
        } else if (serLayer.gTypeStr() == layers::LayerTypeStr_StridedSlice
                || serLayer.gTypeStr() == layers::LayerTypeStr_Unstack
                || serLayer.gTypeStr() == layers::LayerTypeStr_Sigmoid
                || serLayer.gTypeStr() == layers::LayerTypeStr_ConvTranspose
                || serLayer.gTypeStr() == layers::LayerTypeStr_ClipByValue
                || serLayer.gTypeStr() == layers::LayerTypeStr_Split
                || serLayer.gTypeStr() == layers::LayerTypeStr_Squeeze
                || serLayer.gTypeStr() == layers::LayerTypeStr_ExpandDims
                || serLayer.gTypeStr() == layers::LayerTypeStr_Slice
                || serLayer.gTypeStr() == layers::LayerTypeStr_Minimum
                || serLayer.gTypeStr() == layers::LayerTypeStr_Maximum
                || serLayer.gTypeStr() == layers::LayerTypeStr_Pad
                || serLayer.gTypeStr() == layers::LayerTypeStr_Softplus
                || serLayer.gTypeStr() == layers::LayerTypeStr_Transpose
                || serLayer.gTypeStr() == layers::LayerTypeStr_SpaceToBatchND
                || serLayer.gTypeStr() == layers::LayerTypeStr_BatchToSpaceND
                || serLayer.gTypeStr() == layers::LayerTypeStr_Concat
                )
        {   // FIXME: placeholder
            const std::string& prevLayerName = serLayer.gPrevLayer(0);
            layers::Layer* prevLayer = findLayer(prevLayerName, true);
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
            } else if (serWaveOp.m_WaveOpType == wave::ClipByValueWaveOp::gTypeStrStatic()) {
                waveOp = m_Load->loadClipByValue(serWaveOp);
            } else if (serWaveOp.m_WaveOpType == wave::TensorWaveOp::gTypeStrScaleAddStatic()) {
                waveOp = m_Load->loadScaleAdd(serWaveOp);

            } else if (serWaveOp.m_WaveOpType == wave::TensorWaveOp::gTypeStrMultiplyStatic()) {
                const TensorAluOpType aluOp = TensorAluOpType::Mult;
                if (serWaveOp.m_IsScalarOp) {
                    waveOp = m_Load->loadTensorScalarConst(serWaveOp, aluOp);
                } else {
                    waveOp = m_Load->loadTensorTensor(serWaveOp, aluOp);
                }
            } else if (serWaveOp.m_WaveOpType == wave::TensorWaveOp::gTypeStrAddStatic()) {
                const TensorAluOpType aluOp = TensorAluOpType::Add;
                if (serWaveOp.m_IsScalarOp) {
                    waveOp = m_Load->loadTensorScalarConst(serWaveOp, aluOp);
                } else {
                    waveOp = m_Load->loadTensorTensor(serWaveOp, aluOp);
                }
            } else if (serWaveOp.m_WaveOpType == wave::TensorWaveOp::gTypeStrSubStatic()) {
                const TensorAluOpType aluOp = TensorAluOpType::Sub;
                if (serWaveOp.m_IsScalarOp) {
                    waveOp = m_Load->loadTensorScalarConst(serWaveOp, aluOp);
                } else {
                    waveOp = m_Load->loadTensorTensor(serWaveOp, aluOp);
                }
            } else if (serWaveOp.m_WaveOpType == wave::TensorWaveOp::gTypeStrResAddStatic()) {
                const TensorAluOpType aluOp = TensorAluOpType::Add;
                waveOp = m_Load->loadTensorTensor(serWaveOp, aluOp);
            } else if (serWaveOp.m_WaveOpType == wave::TensorWaveOp::gTypeStrMinimum()) {
                waveOp = m_Load->loadMinimum(serWaveOp);
            } else if (serWaveOp.m_WaveOpType == wave::TensorWaveOp::gTypeStrMaximum()) {
                waveOp = m_Load->loadMaximum(serWaveOp);

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
#undef PARAMS
#define PARAMS sbatomLoadParams
    std::vector<wave::WaveOp*> prevWaveOps;
    wave::SbAtomLoadWaveOp::Params sbatomLoadParams;
    fillWaveOpParams(serWaveOp, prevWaveOps, sbatomLoadParams);

    KCC_UNSERIALIZE(SbAddress);
    KCC_UNSERIALIZE(StartAtMidPart);
    sbatomLoadParams.m_DataType = DataType::dataTypeStr2Id(serWaveOp.m_DataType);
    KCC_UNSERIALIZE(Length);
    KCC_UNSERIALIZE(OffsetInFile);
    KCC_UNSERIALIZE(PartitionStepBytes);
    sbatomLoadParams.m_RefFileName = serWaveOp.m_RefFile;
    KCC_UNSERIALIZE(RefFileFormat);
    for (unsigned int i = 0; i < sbatomLoadParams.m_RefFileShape.size(); ++i) {
        sbatomLoadParams.m_RefFileShape[i] = serWaveOp.m_RefFileShape[i];
    }

    KCC_UNSERIALIZE(NumPartitions);
    KCC_UNSERIALIZE(ContainWeights);

    KCC_UNSERIALIZE(IfmapReplicationNumRows);
    KCC_UNSERIALIZE(IfmapReplicationResolution);
    KCC_UNSERIALIZE(IfmapReplicationStepBytes);

    KCC_UNSERIALIZE(SrcStepElem);

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
#undef PARAMS
#define PARAMS sbatomsaveParams
    std::vector<wave::WaveOp*> prevWaveOps;
    wave::SbAtomSaveWaveOp::Params sbatomsaveParams;
    fillWaveOpParams(serWaveOp, prevWaveOps, sbatomsaveParams);

    KCC_UNSERIALIZE(SbAddress);
    KCC_UNSERIALIZE(StartAtMidPart);
    sbatomsaveParams.m_DataType = DataType::dataTypeStr2Id(serWaveOp.m_DataType);
    KCC_UNSERIALIZE(Length);
    KCC_UNSERIALIZE(OffsetInFile);
    KCC_UNSERIALIZE(PartitionStepBytes);
    sbatomsaveParams.m_RefFileName = serWaveOp.m_RefFile;
    KCC_UNSERIALIZE(RefFileFormat);
    for (unsigned int i = 0; i < sbatomsaveParams.m_RefFileShape.size(); ++i) {
        sbatomsaveParams.m_RefFileShape[i] = serWaveOp.m_RefFileShape[i];
    }

    KCC_UNSERIALIZE(NumPartitions);
    KCC_UNSERIALIZE(FinalLayerOfmap);

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
#undef PARAMS
#define PARAMS poolParams
    std::vector<wave::WaveOp*> prevWaveOps;
    wave::PoolWaveOp::Params poolParams;
    fillWaveOpParams(serWaveOp, prevWaveOps, PARAMS);

    loadSrc(PARAMS, serWaveOp, Dims::XYZ);
    KCC_UNSERIALIZE(SrcWNum);
    KCC_UNSERIALIZE(SrcWStep);
    loadDst(PARAMS, serWaveOp, Dims::XYZ);

    poolParams.m_InDtypeId  = DataType::dataTypeStr2Id(serWaveOp.m_InDtype);

    KCC_UNSERIALIZE(NumPartitions);
    KCC_UNSERIALIZE(PoolFrequency);
    poolParams.m_PoolFunc  = utils::poolTypeStr2Id(serWaveOp.m_PoolFunc);

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
#undef PARAMS
#define PARAMS matmulParams
    std::vector<wave::WaveOp*> prevWaveOps;
    wave::MatMulWaveOp::Params PARAMS;
    fillWaveOpParams(serWaveOp, prevWaveOps, PARAMS);

    KCC_UNSERIALIZE(FmapXNum);
    KCC_UNSERIALIZE(FmapXStep);
    KCC_UNSERIALIZE(FmapYNum);
    KCC_UNSERIALIZE(FmapYStep);
    KCC_UNSERIALIZE(FmapZNum);
    KCC_UNSERIALIZE(FmapZStep);
    KCC_UNSERIALIZE(IfmapsSbAddress);
    PARAMS.m_InDtypeId = DataType::dataTypeStr2Id(serWaveOp.m_InDtype);
    // layer_name
    KCC_UNSERIALIZE(NumColumnPartitions);
    KCC_UNSERIALIZE(NumRowPartitions);
    PARAMS.m_OutDtypeId = DataType::dataTypeStr2Id(serWaveOp.m_OutDtype);
    // previous layers
    KCC_UNSERIALIZE(PsumBankId);
    KCC_UNSERIALIZE(PsumBankOffset);
    KCC_UNSERIALIZE(PsumXNum);
    KCC_UNSERIALIZE(PsumXStep);
    KCC_UNSERIALIZE(PsumYNum);
    KCC_UNSERIALIZE(PsumYStep);
    KCC_UNSERIALIZE(PsumZNum);
    KCC_UNSERIALIZE(PsumZStep);

    KCC_UNSERIALIZE(StartTensorCalc);
    KCC_UNSERIALIZE(StopTensorCalc);

    // waveop name
    // waveop type
    KCC_UNSERIALIZE(WeightsSbAddress);

    KCC_UNSERIALIZE(IfmapReplicationNumRows);
    KCC_UNSERIALIZE(IfmapReplicationResolution);
    KCC_UNSERIALIZE(IfmapReplicationShiftAmnt);

    auto waveOp = new wave::MatMulWaveOp(PARAMS, prevWaveOps);
    Assert(waveOp->gName() == PARAMS.m_WaveOpName, "Wrong waveop name ", waveOp->gName());
    return waveOp;
#undef PARAMS
} // Network::Load::loadMatMul

wave::ActivationWaveOp*
Network::Load::loadActivation(const serialize::SerWaveOp& serWaveOp)
{
#undef PARAMS
#define PARAMS activationParams
    std::vector<wave::WaveOp*> prevWaveOps;
    wave::ActivationWaveOp::Params PARAMS;
    fillWaveOpParams(serWaveOp, prevWaveOps, PARAMS);

    PARAMS.m_ActivationFunc   = serialize::SerWaveOp::str2ActivationFunc(serWaveOp.m_ActivationFunc);

    PARAMS.m_BiasDtypeId      = DataType::dataTypeStr2Id(serWaveOp.m_BiasDtype);
    KCC_UNSERIALIZE(BiasAddEn);
    KCC_UNSERIALIZE(BiasSbAddress);
    KCC_UNSERIALIZE(BiasStartAtMidPart);

    loadSrc(PARAMS, serWaveOp, Dims::XYZ);
    loadDst(PARAMS, serWaveOp, Dims::XYZ);

    KCC_UNSERIALIZE(NumPartitions);


    Assert(PARAMS.m_TileId.size() == serWaveOp.m_TileId.size(),
        serWaveOp.m_WaveOpType, " waveop '", serWaveOp.m_WaveOpName,
        "' has wrong tile id size: ", PARAMS.m_TileId.size());
    for (unsigned int i = 0; i < serWaveOp.m_TileId.size(); ++i) {
        PARAMS.m_TileId[i] = serWaveOp.m_TileId[i];
    }
    KCC_UNSERIALIZE(TileIdFormat);

    auto waveOp = new wave::ActivationWaveOp(PARAMS, prevWaveOps);
    Assert(waveOp->gName() == PARAMS.m_WaveOpName, "Wrong waveop name ", waveOp->gName());
    return waveOp;
#undef PARAMS
}

wave::ClipByValueWaveOp*
Network::Load::loadClipByValue(const serialize::SerWaveOp& serWaveOp)
{
#undef PARAMS
#define PARAMS clipByValueParams
    std::vector<wave::WaveOp*> prevWaveOps;
    wave::ClipByValueWaveOp::Params PARAMS;
    fillWaveOpParams(serWaveOp, prevWaveOps, PARAMS);

    KCC_UNSERIALIZE(NumPartitions);
    KCC_UNSERIALIZE(MinValue);
    KCC_UNSERIALIZE(MaxValue);

    loadSrc(PARAMS, serWaveOp, Dims::XYZ);
    loadDst(PARAMS, serWaveOp, Dims::XYZ);

    Assert(PARAMS.m_TileId.size() == serWaveOp.m_TileId.size(),
        serWaveOp.m_WaveOpType, " waveop '", serWaveOp.m_WaveOpName,
        "' has wrong tile id size: ", PARAMS.m_TileId.size());
    for (unsigned int i = 0; i < serWaveOp.m_TileId.size(); ++i) {
        PARAMS.m_TileId[i] = serWaveOp.m_TileId[i];
    }
    KCC_UNSERIALIZE(TileIdFormat);

    auto waveOp = new wave::ClipByValueWaveOp(PARAMS, prevWaveOps);
    Assert(waveOp->gName() == PARAMS.m_WaveOpName, "Wrong waveop name ", waveOp->gName());
    return waveOp;
#undef PARAMS
}




wave::TensorWaveOp*
Network::Load::loadMinimum(const serialize::SerWaveOp& serWaveOp)
{
    const TensorAluOpType aluOp = TensorAluOpType::Min;
    wave::TensorWaveOp* waveOp = nullptr;
    if (serWaveOp.m_IsScalarOp) {
        waveOp = loadTensorScalarConst(serWaveOp, aluOp);
    } else {
        waveOp = loadTensorTensor(serWaveOp, aluOp);
    }
    return waveOp;
}

wave::TensorWaveOp*
Network::Load::loadMaximum(const serialize::SerWaveOp& serWaveOp)
{
    const TensorAluOpType aluOp = TensorAluOpType::Max;
    wave::TensorWaveOp* waveOp = nullptr;
    if (serWaveOp.m_IsScalarOp) {
        waveOp = loadTensorScalarConst(serWaveOp, aluOp);
    } else {
        waveOp = loadTensorTensor(serWaveOp, aluOp);
    }
    return waveOp;
}

wave::TensorScalarConstWaveOp*
Network::Load::loadScaleAdd(const serialize::SerWaveOp& serWaveOp)
{
#undef PARAMS
#define PARAMS tensorScalarConstParams
    std::vector<wave::WaveOp*> prevWaveOps;
    wave::TensorScalarConstWaveOp::Params PARAMS;
    fillWaveOpParams(serWaveOp, prevWaveOps, PARAMS);

    KCC_UNSERIALIZE(NumPartitions);

    loadSrc(PARAMS, serWaveOp, Dims::XYZ);
    loadDst(PARAMS, serWaveOp, Dims::XYZ);

    KCC_UNSERIALIZE(WaveOpType);

    if (serWaveOp.m_WaveOpType == wave::TensorScalarConstWaveOp::gTypeStrScaleAddStatic()) {
        // y = aluOp[1] * (x + aluOp[0])
        PARAMS.m_AluOp[0] = TensorAluOpType::Mult; 
        PARAMS.m_AluOp[1] = TensorAluOpType::Add; 
        PARAMS.m_ImmVal[0] = serWaveOp.m_Scale;
        PARAMS.m_ImmVal[1] = serWaveOp.m_Add;
    } else {
        Assert(false, "Supported ALU ops are: Add, Mult");
    }

    auto waveOp = new wave::TensorScalarConstWaveOp(PARAMS, prevWaveOps);
    Assert(waveOp->gName() == PARAMS.m_WaveOpName, "Wrong waveop name ", waveOp->gName());
    return waveOp;
#undef PARAMS
}


wave::TensorTensorWaveOp*
Network::Load::loadTensorTensor(const serialize::SerWaveOp& serWaveOp, TensorAluOpType aluOp)
{
#undef PARAMS
#define PARAMS tensorTensorParams
    std::vector<wave::WaveOp*> prevWaveOps;
    wave::TensorTensorWaveOp::Params PARAMS;
    fillWaveOpParams(serWaveOp, prevWaveOps, PARAMS);

    PARAMS.m_InADtypeId        = DataType::dataTypeStr2Id(serWaveOp.m_InADtype);
    PARAMS.m_InBDtypeId        = DataType::dataTypeStr2Id(serWaveOp.m_InBDtype);
    PARAMS.m_OutDtypeId       = DataType::dataTypeStr2Id(serWaveOp.m_OutDtype);
    KCC_UNSERIALIZE(NumPartitions);

    loadSrcAB(PARAMS, serWaveOp, Dims::XYZ);
    loadDst(PARAMS, serWaveOp, Dims::XYZ);

    KCC_UNSERIALIZE(WaveOpType);

    auto waveOp = new wave::TensorTensorWaveOp(aluOp, PARAMS, prevWaveOps);
    Assert(waveOp->gName() == PARAMS.m_WaveOpName, "Wrong waveop name ", waveOp->gName());
    return waveOp;
#undef PARAMS
}

wave::TensorScalarConstWaveOp*
Network::Load::loadTensorScalarConst(const serialize::SerWaveOp& serWaveOp, TensorAluOpType aluOp)
{
#undef PARAMS
#define PARAMS tensorScalarAddParams
    std::vector<wave::WaveOp*> prevWaveOps;
    wave::TensorScalarConstWaveOp::Params PARAMS;
    fillWaveOpParams(serWaveOp, prevWaveOps, tensorScalarAddParams);

    PARAMS.m_InDtypeId        = DataType::dataTypeStr2Id(serWaveOp.m_InDtype);
    PARAMS.m_OutDtypeId       = DataType::dataTypeStr2Id(serWaveOp.m_OutDtype);
    KCC_UNSERIALIZE(NumPartitions);

    loadSrc(PARAMS, serWaveOp, Dims::XYZ);
    loadDst(PARAMS, serWaveOp, Dims::XYZ);

    KCC_UNSERIALIZE(WaveOpType);

    switch (aluOp) {
    case TensorAluOpType::Add:
    case TensorAluOpType::Sub:
    case TensorAluOpType::Mult:
    case TensorAluOpType::Min:
    case TensorAluOpType::Max:
        PARAMS.m_AluOp[0]  = TensorAluOpType::Bypass;
        PARAMS.m_ImmVal[0] = 0.0;
        PARAMS.m_AluOp[1]  = aluOp;
        PARAMS.m_ImmVal[1] = serWaveOp.m_ScalarVal;
        break;
    default:
        Assert(false, "Supported TensorScalar ops are: Add, Sub, Mult, Minimum, Maximum: ", 
            static_cast<kcc_int32>(aluOp));
        break;
    }

    auto waveOp = new wave::TensorScalarConstWaveOp(PARAMS, prevWaveOps);
    Assert(waveOp->gName() == PARAMS.m_WaveOpName, "Wrong waveop name ", waveOp->gName());
    return waveOp;
#undef PARAMS
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
    waveOpParams.m_LayerName    = serWaveOp.m_LayerName;
    waveOpParams.m_Layer        = m_Network.findLayer(serWaveOp.m_LayerName, false);
    waveOpParams.m_Order        = m_Network.gWaveOps().size();
    Assert(waveOpParams.m_LayerName != "", "Missing layer name for waveop ", serWaveOp.m_WaveOpName);
    for (const auto& prevWaveOpName : serWaveOp.m_PreviousWaveOps) {
        prevWaveOps.push_back(m_Network.findWaveOp(prevWaveOpName));
    }
}




#undef KCC_UNSERIALIZE
#undef ASSERT_NUM_LAYERS
#undef ASSERT_PREV_LAYER

}}


