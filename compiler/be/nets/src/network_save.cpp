#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <map>

#include <cereal/types/memory.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/map.hpp>


#include "utils/inc/types.hpp"
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

#include "wave/inc/matmulwaveop.hpp"
#include "wave/inc/sbatomfilewaveop.hpp"
#include "wave/inc/sbatomsavewaveop.hpp"
#include "wave/inc/poolwaveop.hpp"
#include "wave/inc/activationwaveop.hpp"

#include "serialize/inc/serlayer.hpp"
#include "serialize/inc/serwaveop.hpp"

namespace kcc {
namespace nets {

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
            /*
            {
                serLayer.rBatchingInWave(convLayer->gBatchingInWave());
            }
            */

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
    for (unsigned waveOpIdx = 0; waveOpIdx < m_WaveOps.size(); ++waveOpIdx) {
        serialize::SerWaveOp& serWaveOp(serWaveOps[waveOpIdx]);
        wave::WaveOp* waveOp = m_WaveOps[waveOpIdx];
        serWaveOp.m_WaveOpName = waveOp->gName();
        serWaveOp.m_LayerName = waveOp->gLayer()->gName();
        for (auto prevWaveOp : waveOp->gPrevWaveOps()) {
            serWaveOp.addPreviousWaveOp(prevWaveOp->gName());
        }

        if (auto sbatomWaveOp = dynamic_cast<const wave::SbAtomWaveOp*>(waveOp)) {
            saveSbAtom(sbatomWaveOp, serWaveOp);
            continue;
        }

        if (const auto matmulWaveOp = dynamic_cast<wave::MatMulWaveOp*>(waveOp)) {
            serWaveOp.m_WaveOpType = wave::MatMulWaveOp::gTypeStr();

            serWaveOp.m_BatchingInWave = matmulWaveOp->gBatchingInWave();
            serWaveOp.m_FmapXNum = matmulWaveOp->gFmapXNum();
            serWaveOp.m_FmapXStep = matmulWaveOp->gFmapXStep();
            serWaveOp.m_FmapYNum = matmulWaveOp->gFmapYNum();
            serWaveOp.m_FmapYStep = matmulWaveOp->gFmapYStep();
            serWaveOp.m_FmapZNum = matmulWaveOp->gFmapZNum();
            serWaveOp.m_FmapZStepAtoms = matmulWaveOp->gFmapZStepAtoms();
            serWaveOp.m_IfmapCount = matmulWaveOp->gIfmapCount();
            serWaveOp.m_IfmapTileHeight = matmulWaveOp->gIfmapTileHeight();
            serWaveOp.m_IfmapTileWidth = matmulWaveOp->gIfmapTileWidth();
            serWaveOp.m_IfmapsAtomId = matmulWaveOp->gIfmapsAtomId();
            serWaveOp.m_IfmapsAtomSize = matmulWaveOp->gIfmapsAtomSize();
            serWaveOp.m_IfmapsOffsetInAtom = matmulWaveOp->gIfmapsOffsetInAtom();
            // layer_name
            serWaveOp.m_NumColumnPartitions = matmulWaveOp->gNumColumnPartitions();
            serWaveOp.m_NumRowPartitions = matmulWaveOp->gNumRowPartitions();
            serWaveOp.m_OfmapCount = matmulWaveOp->gOfmapCount();
            serWaveOp.m_OfmapTileHeight = matmulWaveOp->gOfmapTileHeight();
            serWaveOp.m_OfmapTileWidth = matmulWaveOp->gOfmapTileWidth();
            // previous layers
            serWaveOp.m_PsumBankId = matmulWaveOp->gPsumBankId();
            serWaveOp.m_PsumBankOffset = matmulWaveOp->gPsumBankOffset();
            serWaveOp.m_PsumXNum = matmulWaveOp->gPsumXNum();
            serWaveOp.m_PsumXStep = matmulWaveOp->gPsumXStep();
            serWaveOp.m_PsumYNum = matmulWaveOp->gPsumYNum();
            serWaveOp.m_PsumYStep = matmulWaveOp->gPsumYStep();
            serWaveOp.m_StartTensorCalc = matmulWaveOp->qStartTensorCalc();
            serWaveOp.m_StopTensorCalc = matmulWaveOp->qStopTensorCalc();
            serWaveOp.m_StrideX = matmulWaveOp->gStrideX();
            serWaveOp.m_StrideY = matmulWaveOp->gStrideY();
            serWaveOp.m_WaveId = matmulWaveOp->gWaveId();
            serWaveOp.m_WaveIdFormat = matmulWaveOp->gWaveIdFormat();
            // waveop name
            // waveop type
            serWaveOp.m_WeightsAtomId = matmulWaveOp->gWeightsAtomId();
            serWaveOp.m_WeightsOffsetInAtom = matmulWaveOp->gWeightsOffsetInAtom();

            continue;
        }
        if (const auto poolWaveOp = dynamic_cast<wave::PoolWaveOp*>(waveOp)) {
            serWaveOp.m_WaveOpType = wave::PoolWaveOp::gTypeStr();

            serWaveOp.m_DstSbAtomId             = poolWaveOp->gDstSbAtomId();
            serWaveOp.m_DstSbOffsetInAtom       = poolWaveOp->gDstSbOffsetInAtom();
            serWaveOp.m_DstXNum                 = poolWaveOp->gDstXNum();
            serWaveOp.m_DstXStep                = poolWaveOp->gDstXStep();
            serWaveOp.m_DstYNum                 = poolWaveOp->gDstYNum();
            serWaveOp.m_DstYStep                = poolWaveOp->gDstYStep();
            serWaveOp.m_DstZNum                 = poolWaveOp->gDstZNum();
            serWaveOp.m_DstZStep                = poolWaveOp->gDstZStep();
            serWaveOp.m_InDtype                 = poolWaveOp->gInDtype().gName();
            serWaveOp.m_NumPartitions           = poolWaveOp->gNumPartitions();
            serWaveOp.m_OutDtype                = poolWaveOp->gOutDtype().gName();
            serWaveOp.m_PoolFrequency           = poolWaveOp->gPoolFrequency();
            serWaveOp.m_PoolFunc                = utils::poolType2Str(poolWaveOp->gPoolFunc());
            serWaveOp.m_SrcIsPsum               = poolWaveOp->qSrcIsPsum();
            serWaveOp.m_SrcPsumBankId           = poolWaveOp->gSrcPsumBankId();
            serWaveOp.m_SrcPsumBankOffset       = poolWaveOp->gSrcPsumBankOffset();
            serWaveOp.m_SrcSbAtomId             = poolWaveOp->gSrcSbAtomId();
            serWaveOp.m_SrcSbOffsetInAtom       = poolWaveOp->gSrcSbOffsetInAtom();
            serWaveOp.m_SrcWNum                 = poolWaveOp->gSrcWNum();
            serWaveOp.m_SrcWStep                = poolWaveOp->gSrcWStep();
            serWaveOp.m_SrcXNum                 = poolWaveOp->gSrcXNum();
            serWaveOp.m_SrcXStep                = poolWaveOp->gSrcXStep();
            serWaveOp.m_SrcYNum                 = poolWaveOp->gSrcYNum();
            serWaveOp.m_SrcYStep                = poolWaveOp->gSrcYStep();
            serWaveOp.m_SrcZNum                 = poolWaveOp->gSrcZNum();
            serWaveOp.m_SrcZStep                = poolWaveOp->gSrcZStep();

            for (unsigned int i = 0; i < poolWaveOp->gTileId().size(); ++i) {
                serWaveOp.m_TileId[i] = poolWaveOp->gTileId()[i];
            }
            serWaveOp.m_TileIdFormat            = poolWaveOp->gTileIdFormat();


            continue;
        }
        if (const auto activationWaveOp = dynamic_cast<const wave::ActivationWaveOp*>(waveOp)) {
            saveActivaton(activationWaveOp, serWaveOp);
            continue;
        }
        assert(false && "Unsupported WaveOp");
    }
    archive(cereal::make_nvp(NetKey_WaveOps, serWaveOps));
}


void
Network::saveSbAtom(const wave::SbAtomWaveOp* sbatomWaveOp,
                    serialize::SerWaveOp& serWaveOp) const
{
    serWaveOp.m_AtomId = sbatomWaveOp->gAtomId();
    serWaveOp.m_AtomSize = sbatomWaveOp->gAtomSize();
    serWaveOp.m_BatchFoldIdx = sbatomWaveOp->gBatchFoldIdx();
    serWaveOp.m_DataType = DataType::dataTypeId2Str(
                              sbatomWaveOp->gDataType().gDataTypeId());
    serWaveOp.m_Length = sbatomWaveOp->gLength();
    serWaveOp.m_OffsetInFile = sbatomWaveOp->gOffsetInFile();
    serWaveOp.m_RefFile = sbatomWaveOp->gRefFileName();
    serWaveOp.m_RefFileFormat = sbatomWaveOp->gRefFileFormat();
    const std::array<kcc_int32,4>& refFileShape(sbatomWaveOp->gRefFileShape());
    for (unsigned int shapeIdx = 0; shapeIdx < refFileShape.size(); ++shapeIdx) {
        serWaveOp.m_RefFileShape[shapeIdx] = refFileShape[shapeIdx];
    }

    if (auto sbatomfileWaveOp = dynamic_cast<const wave::SbAtomFileWaveOp*>(sbatomWaveOp)) {
        serWaveOp.m_WaveOpType = wave::SbAtomFileWaveOp::gTypeStr();
        serWaveOp.m_IfmapCount = sbatomfileWaveOp->gIfmapCount();
        serWaveOp.m_IfmapsFoldIdx = sbatomfileWaveOp->gIfmapsFoldIdx();
        serWaveOp.m_IfmapsReplicate = sbatomfileWaveOp->qIfmapsReplicate();
    } else {
        auto sbatomsaveWaveOp = dynamic_cast<const wave::SbAtomSaveWaveOp*>(sbatomWaveOp);
        assert(sbatomsaveWaveOp && "Wrong SbAtaom WaveOp");
        serWaveOp.m_WaveOpType = wave::SbAtomSaveWaveOp::gTypeStr();
        serWaveOp.m_OfmapCount = sbatomsaveWaveOp->gOfmapCount();
        serWaveOp.m_OfmapsFoldIdx = sbatomsaveWaveOp->gOfmapsFoldIdx();
    }
}


void
Network::saveActivaton(const wave::ActivationWaveOp* activationWaveOp,
                       serialize::SerWaveOp& serWaveOp) const
{
    serWaveOp.m_WaveOpType = wave::ActivationWaveOp::gTypeStr();

    serWaveOp.m_ActType             = serialize::SerWaveOp::activationType2Str(activationWaveOp->gActType());
    serWaveOp.m_BiasAddEn           = activationWaveOp->qBiasAddEn();
    serWaveOp.m_BiasAtomId          = activationWaveOp->gBiasAtomId();
    serWaveOp.m_BiasOffsetInAtom    = activationWaveOp->gBiasOffsetInAtom();
    serWaveOp.m_PsumBankIdDst       = activationWaveOp->gPsumBankIdDst();
    serWaveOp.m_PsumBankIdSrc       = activationWaveOp->gPsumBankIdSrc();
    const std::array<kcc_int32, 4>& tileId(activationWaveOp->gTileId());
    for (unsigned int i = 0; i < tileId.size(); ++i) {
        serWaveOp.m_TileId[i]       = tileId[i];
    }
    serWaveOp.m_TileIdFormat        = activationWaveOp->gTileIdFormat();
}

}}



