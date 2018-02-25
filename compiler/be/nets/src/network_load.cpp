#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <map>

#include <cereal/types/memory.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/map.hpp>


#include "utils/inc/consts.hpp"
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

#include "wave/inc/sbatomfilewaveop.hpp"
#include "wave/inc/sbatomsavewaveop.hpp"

#include "wave/inc/matmulwaveop.hpp"
#include "wave/inc/poolwaveop.hpp"
#include "wave/inc/activationwaveop.hpp"

#include "serialize/inc/serlayer.hpp"
#include "serialize/inc/serwaveop.hpp"

#define KCC_UNSERIALIZE(X) PARAMS.KCC_CONCAT(m_,X) = serWaveOp.KCC_CONCAT(m_,X);
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
            layers::ConvLayer::Params convParams(params);
            /*
            convParams.m_BatchingInWave    = serLayer.gBatchingInWave();
            */

            layer = new layers::ConvLayer(convParams, prevLayer,
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
    if (m_UseWave) {
        std::vector<serialize::SerWaveOp> serWaveOps;
        archive(cereal::make_nvp(NetKey_WaveOps, serWaveOps));
        for (unsigned i = 0; i < serWaveOps.size(); ++i) {
            const serialize::SerWaveOp& serWaveOp(serWaveOps[i]);

            wave::WaveOp* waveOp = nullptr;

            if (serWaveOp.m_WaveOpType == wave::SbAtomFileWaveOp::gTypeStr()) {
                waveOp = loadSbAtomFile(serWaveOp);
            } else if (serWaveOp.m_WaveOpType == wave::SbAtomSaveWaveOp::gTypeStr()) {
                waveOp = loadSbAtomSave(serWaveOp);
            } else if (serWaveOp.m_WaveOpType == wave::PoolWaveOp::gTypeStr()) {
                waveOp = loadPool(serWaveOp);
            } else if (serWaveOp.m_WaveOpType == wave::MatMulWaveOp::gTypeStr()) {
                waveOp = loadMatMul(serWaveOp);
            } else if (serWaveOp.m_WaveOpType == wave::ActivationWaveOp::gTypeStr()) {
                waveOp = loadActivation(serWaveOp);
            } else {
                assert(false && "Wrong WaveOp type during deserialization");
            }

            m_WaveOps.push_back(waveOp);
            assert(m_Name2WaveOp.find(waveOp->gName()) == m_Name2WaveOp.end());
            m_Name2WaveOp[waveOp->gName()] = waveOp;
        }
    }
}


wave::SbAtomFileWaveOp*
Network::loadSbAtomFile(const serialize::SerWaveOp& serWaveOp)
{
#define PARAMS sbatomfileParams
    std::vector<wave::WaveOp*> prevWaveOps;
    wave::SbAtomFileWaveOp::Params sbatomfileParams;
    fillWaveOpParams(serWaveOp, prevWaveOps, sbatomfileParams);

    KCC_UNSERIALIZE(AtomId);
    KCC_UNSERIALIZE(AtomSize);
    KCC_UNSERIALIZE(BatchFoldIdx);
    sbatomfileParams.m_DataType = DataType::dataTypeStr2Id(serWaveOp.m_DataType);
    KCC_UNSERIALIZE(Length);
    KCC_UNSERIALIZE(OffsetInFile);
    KCC_UNSERIALIZE(PartitionStepBytes);
    sbatomfileParams.m_RefFileName = serWaveOp.m_RefFile;
    KCC_UNSERIALIZE(RefFileFormat);
    for (unsigned int i = 0; i < sbatomfileParams.m_RefFileShape.size(); ++i) {
        sbatomfileParams.m_RefFileShape[i] = serWaveOp.m_RefFileShape[i];
    }

    KCC_UNSERIALIZE(IfmapCount);
    KCC_UNSERIALIZE(IfmapsFoldIdx);
    KCC_UNSERIALIZE(IfmapsReplicate);

    auto waveOp = new wave::SbAtomFileWaveOp(sbatomfileParams, prevWaveOps);
    assert(waveOp && waveOp->gName() == sbatomfileParams.m_WaveOpName);
    return waveOp;
#undef PARAMS
}


wave::SbAtomSaveWaveOp*
Network::loadSbAtomSave(const serialize::SerWaveOp& serWaveOp)
{
#define PARAMS sbatomsaveParams
    std::vector<wave::WaveOp*> prevWaveOps;
    wave::SbAtomSaveWaveOp::Params sbatomsaveParams;
    fillWaveOpParams(serWaveOp, prevWaveOps, sbatomsaveParams);

    KCC_UNSERIALIZE(AtomId);
    KCC_UNSERIALIZE(AtomSize);
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
    assert(waveOp->gName() == sbatomsaveParams.m_WaveOpName);
    return waveOp;
#undef PARAMS
}

wave::PoolWaveOp*
Network::loadPool(const serialize::SerWaveOp& serWaveOp)
{
#define PARAMS poolParams
    std::vector<wave::WaveOp*> prevWaveOps;
    wave::PoolWaveOp::Params poolParams;
    fillWaveOpParams(serWaveOp, prevWaveOps, poolParams);

    KCC_UNSERIALIZE(DstSbAtomId);
    KCC_UNSERIALIZE(DstSbOffsetInAtom);
    KCC_UNSERIALIZE(DstXNum);
    KCC_UNSERIALIZE(DstXStep);
    KCC_UNSERIALIZE(DstYNum);
    KCC_UNSERIALIZE(DstYStep);
    KCC_UNSERIALIZE(DstZNum);
    KCC_UNSERIALIZE(DstZStep);
    poolParams.m_InDtype  = DataType::dataTypeStr2Id(serWaveOp.m_InDtype);
    KCC_UNSERIALIZE(NumPartitions);
    poolParams.m_OutDtype = DataType::dataTypeStr2Id(serWaveOp.m_OutDtype);
    KCC_UNSERIALIZE(PoolFrequency);
    poolParams.m_PoolFunc  = utils::poolTypeStr2Id(serWaveOp.m_PoolFunc);

    KCC_UNSERIALIZE(SrcIsPsum);
    if (serWaveOp.m_SrcIsPsum) {
        KCC_UNSERIALIZE(SrcPsumBankId);
        KCC_UNSERIALIZE(SrcPsumBankOffset);
    } else {
        KCC_UNSERIALIZE(SrcSbAtomId);
        KCC_UNSERIALIZE(SrcSbOffsetInAtom);
    }

    KCC_UNSERIALIZE(SrcWNum);
    KCC_UNSERIALIZE(SrcWStep);
    KCC_UNSERIALIZE(SrcXNum);
    KCC_UNSERIALIZE(SrcXStep);
    KCC_UNSERIALIZE(SrcYNum);
    KCC_UNSERIALIZE(SrcYStep);
    KCC_UNSERIALIZE(SrcZNum);
    KCC_UNSERIALIZE(SrcZStep);

    assert(poolParams.m_TileId.size() == serWaveOp.m_TileId.size());
    for (unsigned int i = 0; i < serWaveOp.m_TileId.size(); ++i) {
        poolParams.m_TileId[i] = serWaveOp.m_TileId[i];
    }
    poolParams.m_TileIdFormat           = serWaveOp.m_TileIdFormat;

    auto waveOp = new wave::PoolWaveOp(poolParams, prevWaveOps);
    assert(waveOp->gName() == poolParams.m_WaveOpName);
    return waveOp;
#undef PARAMS
}

wave::MatMulWaveOp*
Network::loadMatMul(const serialize::SerWaveOp& serWaveOp)
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
    KCC_UNSERIALIZE(IfmapsAtomId);
    KCC_UNSERIALIZE(IfmapsAtomSize);
    KCC_UNSERIALIZE(IfmapsOffsetInAtom);
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
    KCC_UNSERIALIZE(WeightsAtomId);
    KCC_UNSERIALIZE(WeightsOffsetInAtom);

    auto waveOp = new wave::MatMulWaveOp(matmulParams, prevWaveOps);
    assert(waveOp->gName() == matmulParams.m_WaveOpName);
    return waveOp;
#undef PARAMS
}

wave::ActivationWaveOp*
Network::loadActivation(const serialize::SerWaveOp& serWaveOp)
{
#define PARAMS activationParams
    std::vector<wave::WaveOp*> prevWaveOps;
    wave::ActivationWaveOp::Params activationParams;
    fillWaveOpParams(serWaveOp, prevWaveOps, activationParams);

    activationParams.m_ActivationFunc   = serialize::SerWaveOp::str2ActivationFunc(serWaveOp.m_ActivationFunc);

    KCC_UNSERIALIZE(BiasAddEn);
    KCC_UNSERIALIZE(BiasAtomId);
    KCC_UNSERIALIZE(BiasOffsetInAtom);

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
        KCC_UNSERIALIZE(DstSbAtomId);
        KCC_UNSERIALIZE(DstSbOffsetInAtom);
    }

    activationParams.m_InDtypeId        = DataType::dataTypeStr2Id(serWaveOp.m_InDtype);
    KCC_UNSERIALIZE(NumPartitions);
    activationParams.m_OutDtypeId       = DataType::dataTypeStr2Id(serWaveOp.m_OutDtype);
    KCC_UNSERIALIZE(SrcPsumBankId);
    KCC_UNSERIALIZE(SrcXNum);
    KCC_UNSERIALIZE(SrcXStep);
    KCC_UNSERIALIZE(SrcYNum);
    KCC_UNSERIALIZE(SrcYStep);
    KCC_UNSERIALIZE(SrcZNum);
    KCC_UNSERIALIZE(SrcZStep);

    assert(activationParams.m_TileId.size() == serWaveOp.m_TileId.size());
    for (unsigned int i = 0; i < serWaveOp.m_TileId.size(); ++i) {
        activationParams.m_TileId[i] = serWaveOp.m_TileId[i];
    }
    KCC_UNSERIALIZE(TileIdFormat);

    auto waveOp = new wave::ActivationWaveOp(activationParams, prevWaveOps);
    assert(waveOp->gName() == activationParams.m_WaveOpName);
    return waveOp;
#undef PARAMS
}


/* in
 * template<>
 * void
 * Network::load<cereal::JSONInputArchive>(cereal::JSONInputArchive& archive)
 * {
 *      ...
 *      auto fillWaveOpParams = [this, &prevWaveOps](
 *                              const serialize::SerWaveOp& serWaveOp,
 *                              wave::WaveOp::Params& waveOpParams) -> void
 *      ...
 * }
 */

void
Network::fillWaveOpParams(const serialize::SerWaveOp& serWaveOp,
                     std::vector<wave::WaveOp*>& prevWaveOps,
                     wave::WaveOp::Params& waveOpParams)
{
    waveOpParams.m_WaveOpName   = serWaveOp.m_WaveOpName;
    waveOpParams.m_Layer        = this->findLayer(serWaveOp.m_LayerName);
    assert(waveOpParams.m_Layer);
    for (const auto& prevWaveOpName : serWaveOp.m_PreviousWaveOps) {
        prevWaveOps.push_back(findWaveOp(prevWaveOpName));
    }
}

#undef KCC_UNSERIALIZE

}}


