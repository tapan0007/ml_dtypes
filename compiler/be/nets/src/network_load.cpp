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

#include "wave/inc/sbatomfilewaveop.hpp"
#include "wave/inc/sbatomsavewaveop.hpp"

#include "wave/inc/matmulwaveop.hpp"
#include "wave/inc/poolwaveop.hpp"
#include "wave/inc/activationwaveop.hpp"

#include "serialize/inc/serlayer.hpp"
#include "serialize/inc/serwaveop.hpp"

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

            std::vector<wave::WaveOp*> prevWaveOps;
            wave::WaveOp* waveOp = nullptr;

#if 0
            auto fillWaveOpParams = [this, &prevWaveOps](
                                        const serialize::SerWaveOp& serWaveOp,
                                        wave::WaveOp::Params& waveOpParams) -> void
            {
                waveOpParams.m_WaveOpName   = serWaveOp.m_WaveOpName;
                waveOpParams.m_Layer        = this->findLayer(serWaveOp.m_LayerName);
                assert(waveOpParams.m_Layer);
                for (const auto& prevWaveOpName : serWaveOp.m_PreviousWaveOps) {
                    prevWaveOps.push_back(findWaveOp(prevWaveOpName));
                }
            };
#endif

            if (serWaveOp.m_WaveOpType == wave::SbAtomFileWaveOp::gTypeStr()) {
                waveOp = loadSbAtomFile(serWaveOp, prevWaveOps);
            } else if (serWaveOp.m_WaveOpType == wave::SbAtomSaveWaveOp::gTypeStr()) {
                waveOp = loadSbAtomSave(serWaveOp, prevWaveOps);
            } else if (serWaveOp.m_WaveOpType == wave::PoolWaveOp::gTypeStr()) {
                waveOp = loadPool(serWaveOp, prevWaveOps);
            } else if (serWaveOp.m_WaveOpType == wave::MatMulWaveOp::gTypeStr()) {
                waveOp = loadMatMul(serWaveOp, prevWaveOps);
            } else if (serWaveOp.m_WaveOpType == wave::ActivationWaveOp::gTypeStr()) {
                waveOp = loadActivation(serWaveOp, prevWaveOps);
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
Network::loadSbAtomFile(const serialize::SerWaveOp& serWaveOp,
                        std::vector<wave::WaveOp*> prevWaveOps)
{
    wave::SbAtomFileWaveOp::Params sbatomfileParams;
    fillWaveOpParams(serWaveOp, prevWaveOps, sbatomfileParams);
    sbatomfileParams.m_AtomId            = serWaveOp.m_AtomId;
    sbatomfileParams.m_AtomSize          = serWaveOp.m_AtomSize;
    sbatomfileParams.m_BatchFoldIdx      = serWaveOp.m_BatchFoldIdx;
    sbatomfileParams.m_DataType = DataType::dataTypeStr2Id(
                                    serWaveOp.m_DataType);
    sbatomfileParams.m_Length            = serWaveOp.m_Length;
    sbatomfileParams.m_OffsetInFile      = serWaveOp.m_OffsetInFile;
    sbatomfileParams.m_RefFileName       = serWaveOp.m_RefFile;
    sbatomfileParams.m_RefFileFormat     = serWaveOp.m_RefFileFormat;
    for (unsigned int i = 0; i < sbatomfileParams.m_RefFileShape.size(); ++i) {
        sbatomfileParams.m_RefFileShape[i] = serWaveOp.m_RefFileShape[i];
    }

    sbatomfileParams.m_IfmapCount        = serWaveOp.m_IfmapCount;
    sbatomfileParams.m_IfmapsFoldIdx     = serWaveOp.m_IfmapsFoldIdx;
    sbatomfileParams.m_IfmapsReplicate   = serWaveOp.m_IfmapsReplicate;

    auto waveOp = new wave::SbAtomFileWaveOp(sbatomfileParams, prevWaveOps);
    assert(waveOp && waveOp->gName() == sbatomfileParams.m_WaveOpName);
    return waveOp;
}


wave::SbAtomSaveWaveOp*
Network::loadSbAtomSave(const serialize::SerWaveOp& serWaveOp,
               std::vector<wave::WaveOp*> prevWaveOps)
{
    wave::SbAtomSaveWaveOp::Params sbatomsaveParams;
    fillWaveOpParams(serWaveOp, prevWaveOps, sbatomsaveParams);

    sbatomsaveParams.m_AtomId           = serWaveOp.m_AtomId;
    sbatomsaveParams.m_AtomSize         = serWaveOp.m_AtomSize;
    sbatomsaveParams.m_BatchFoldIdx     = serWaveOp.m_BatchFoldIdx;
    sbatomsaveParams.m_DataType = DataType::dataTypeStr2Id(
                                    serWaveOp.m_DataType);
    sbatomsaveParams.m_Length           = serWaveOp.m_Length;
    sbatomsaveParams.m_OffsetInFile     = serWaveOp.m_OffsetInFile;
    sbatomsaveParams.m_RefFileName      = serWaveOp.m_RefFile;
    sbatomsaveParams.m_RefFileFormat    = serWaveOp.m_RefFileFormat;
    for (unsigned int i = 0; i < sbatomsaveParams.m_RefFileShape.size(); ++i) {
        sbatomsaveParams.m_RefFileShape[i] = serWaveOp.m_RefFileShape[i];
    }

    sbatomsaveParams.m_OfmapCount       = serWaveOp.m_OfmapCount;
    sbatomsaveParams.m_OfmapsFoldIdx    = serWaveOp.m_OfmapsFoldIdx;

    auto waveOp = new wave::SbAtomSaveWaveOp(sbatomsaveParams, prevWaveOps);
    assert(waveOp->gName() == sbatomsaveParams.m_WaveOpName);
    return waveOp;
}

wave::PoolWaveOp*
Network::loadPool(const serialize::SerWaveOp& serWaveOp,
                  std::vector<wave::WaveOp*> prevWaveOps)
{
    wave::PoolWaveOp::Params poolParams;
    fillWaveOpParams(serWaveOp, prevWaveOps, poolParams);

    poolParams.m_DstSbAtomId            = serWaveOp.m_DstSbAtomId;
    poolParams.m_DstSbOffsetInAtom      = serWaveOp.m_DstSbOffsetInAtom;
    poolParams.m_DstXNum                = serWaveOp.m_DstXNum;
    poolParams.m_DstXStep               = serWaveOp.m_DstXStep;
    poolParams.m_DstYNum                = serWaveOp.m_DstYNum;
    poolParams.m_DstYStep               = serWaveOp.m_DstYStep;
    poolParams.m_DstZNum                = serWaveOp.m_DstZNum;
    poolParams.m_DstZStep               = serWaveOp.m_DstZStep;
    poolParams.m_InDtype                = DataType::dataTypeStr2Id(serWaveOp.m_InDtype);
    poolParams.m_NumPartitions          = serWaveOp.m_NumPartitions;
    poolParams.m_OutDtype               = DataType::dataTypeStr2Id(serWaveOp.m_OutDtype);
    poolParams.m_PoolFrequency          = serWaveOp.m_PoolFrequency;
    poolParams.m_PoolFunc               = utils::poolTypeStr2Id(serWaveOp.m_PoolFunc);
    poolParams.m_SrcIsPsum              = serWaveOp.m_SrcIsPsum;
    poolParams.m_SrcPsumBankId          = serWaveOp.m_SrcPsumBankId;
    poolParams.m_SrcPsumBankOffset      = serWaveOp.m_SrcPsumBankOffset;
    poolParams.m_SrcSbAtomId            = serWaveOp.m_SrcSbAtomId;
    poolParams.m_SrcSbOffsetInAtom      = serWaveOp.m_SrcSbOffsetInAtom;
    poolParams.m_SrcWNum                = serWaveOp.m_SrcWNum;
    poolParams.m_SrcWStep               = serWaveOp.m_SrcWStep;
    poolParams.m_SrcXNum                = serWaveOp.m_SrcXNum;
    poolParams.m_SrcXStep               = serWaveOp.m_SrcXStep;
    poolParams.m_SrcYNum                = serWaveOp.m_SrcYNum;
    poolParams.m_SrcYStep               = serWaveOp.m_SrcYStep;
    poolParams.m_SrcZNum                = serWaveOp.m_SrcZNum;
    poolParams.m_SrcZStep               = serWaveOp.m_SrcZStep;

    assert(poolParams.m_TileId.size() == serWaveOp.m_TileId.size());
    for (unsigned int i = 0; i < serWaveOp.m_TileId.size(); ++i) {
        poolParams.m_TileId[i] = serWaveOp.m_TileId[i];
    }
    poolParams.m_TileIdFormat           = serWaveOp.m_TileIdFormat;

    auto waveOp = new wave::PoolWaveOp(poolParams, prevWaveOps);
    assert(waveOp->gName() == poolParams.m_WaveOpName);
    return waveOp;
}

wave::MatMulWaveOp*
Network::loadMatMul(const serialize::SerWaveOp& serWaveOp,
                    std::vector<wave::WaveOp*> prevWaveOps)
{
    wave::MatMulWaveOp::Params matmulParams;
    fillWaveOpParams(serWaveOp, prevWaveOps, matmulParams);

    matmulParams.m_BatchingInWave       = serWaveOp.m_BatchingInWave;
    matmulParams.m_FmapXNum             = serWaveOp.m_FmapXNum;
    matmulParams.m_FmapXStep            = serWaveOp.m_FmapXStep;
    matmulParams.m_FmapYNum             = serWaveOp.m_FmapYNum;
    matmulParams.m_FmapYStep            = serWaveOp.m_FmapYStep;
    matmulParams.m_FmapZNum             = serWaveOp.m_FmapZNum;
    matmulParams.m_FmapZStepAtoms       = serWaveOp.m_FmapZStepAtoms;
    matmulParams.m_IfmapCount           = serWaveOp.m_IfmapCount;
    matmulParams.m_IfmapTileHeight      = serWaveOp.m_IfmapTileHeight;
    matmulParams.m_IfmapTileWidth       = serWaveOp.m_IfmapTileWidth;
    matmulParams.m_IfmapsAtomId         = serWaveOp.m_IfmapsAtomId;
    matmulParams.m_IfmapsAtomSize       = serWaveOp.m_IfmapsAtomSize;
    matmulParams.m_IfmapsOffsetInAtom   = serWaveOp.m_IfmapsOffsetInAtom;
    // layer_name
    matmulParams.m_NumColumnPartitions  = serWaveOp.m_NumColumnPartitions;
    matmulParams.m_NumRowPartitions     = serWaveOp.m_NumRowPartitions;
    matmulParams.m_OfmapCount           = serWaveOp.m_OfmapCount;
    matmulParams.m_OfmapTileHeight      = serWaveOp.m_OfmapTileHeight;
    matmulParams.m_OfmapTileWidth       = serWaveOp.m_OfmapTileWidth;
    // previous layers
    matmulParams.m_PsumBankId           = serWaveOp.m_PsumBankId;
    matmulParams.m_PsumBankOffset       = serWaveOp.m_PsumBankOffset;
    matmulParams.m_PsumXNum             = serWaveOp.m_PsumXNum;
    matmulParams.m_PsumXStep            = serWaveOp.m_PsumXStep;
    matmulParams.m_PsumYNum             = serWaveOp.m_PsumYNum;
    matmulParams.m_PsumYStep            = serWaveOp.m_PsumYStep;

    matmulParams.m_StartTensorCalc      = serWaveOp.m_StartTensorCalc;
    matmulParams.m_StopTensorCalc       = serWaveOp.m_StopTensorCalc;
    matmulParams.m_StrideX              = serWaveOp.m_StrideX;
    matmulParams.m_StrideY              = serWaveOp.m_StrideY;

    matmulParams.m_WaveId               = serWaveOp.m_WaveId;
    matmulParams.m_WaveIdFormat         = serWaveOp.m_WaveIdFormat;
    // waveop name
    // waveop type
    matmulParams.m_WeightsAtomId        = serWaveOp.m_WeightsAtomId;
    matmulParams.m_WeightsOffsetInAtom  = serWaveOp.m_WeightsOffsetInAtom;

    auto waveOp = new wave::MatMulWaveOp(matmulParams, prevWaveOps);
    assert(waveOp->gName() == matmulParams.m_WaveOpName);
    return waveOp;
}

wave::ActivationWaveOp*
Network::loadActivation(const serialize::SerWaveOp& serWaveOp,
                        std::vector<wave::WaveOp*> prevWaveOps)
{
    wave::ActivationWaveOp::Params activationParams;
    fillWaveOpParams(serWaveOp, prevWaveOps, activationParams);

    activationParams.m_ActivationFunc   = serialize::SerWaveOp::str2ActivationFunc(serWaveOp.m_ActivationFunc);
    activationParams.m_BiasAddEn        = serWaveOp.m_BiasAddEn;
    activationParams.m_BiasAtomId       = serWaveOp.m_BiasAtomId;
    activationParams.m_BiasOffsetInAtom = serWaveOp.m_BiasOffsetInAtom;
    activationParams.m_DstPsumBankId    = serWaveOp.m_DstPsumBankId;
    activationParams.m_DstXNum          = serWaveOp.m_DstXNum;
    activationParams.m_DstXStep         = serWaveOp.m_DstXStep;
    activationParams.m_DstYNum          = serWaveOp.m_DstYNum;
    activationParams.m_DstYStep         = serWaveOp.m_DstXStep;
    activationParams.m_DstZNum          = serWaveOp.m_DstZNum;
    activationParams.m_DstZStep         = serWaveOp.m_DstZStep;
    activationParams.m_InDtypeId        = DataType::dataTypeStr2Id(serWaveOp.m_InDtype);
    activationParams.m_NumPartitions    = serWaveOp.m_NumPartitions;
    activationParams.m_OutDtypeId       = DataType::dataTypeStr2Id(serWaveOp.m_OutDtype);
    activationParams.m_SrcPsumBankId    = serWaveOp.m_SrcPsumBankId;
    activationParams.m_SrcXNum          = serWaveOp.m_SrcXNum;
    activationParams.m_SrcXStep         = serWaveOp.m_SrcXStep;
    activationParams.m_SrcYNum          = serWaveOp.m_SrcYNum;
    activationParams.m_SrcYStep         = serWaveOp.m_SrcYStep;
    activationParams.m_SrcZNum          = serWaveOp.m_SrcZNum;
    activationParams.m_SrcZStep         = serWaveOp.m_SrcZStep;

    assert(activationParams.m_TileId.size() == serWaveOp.m_TileId.size());
    for (unsigned int i = 0; i < serWaveOp.m_TileId.size(); ++i) {
        activationParams.m_TileId[i] = serWaveOp.m_TileId[i];
    }
    activationParams.m_TileIdFormat           = serWaveOp.m_TileIdFormat;

    auto waveOp = new wave::ActivationWaveOp(activationParams, prevWaveOps);
    assert(waveOp->gName() == activationParams.m_WaveOpName);
    return waveOp;
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


}}


