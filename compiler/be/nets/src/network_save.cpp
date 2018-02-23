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

#define KCC_SERIALIZE(X) serWaveOp.KCC_CONCAT(m_,X) = WAVE_OP->KCC_CONCAT(g,X)()

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

            saveMatmul(matmulWaveOp, serWaveOp);
            continue;
        }
        if (const auto poolWaveOp = dynamic_cast<wave::PoolWaveOp*>(waveOp)) {
            savePool(poolWaveOp, serWaveOp);
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
Network::saveMatmul(const wave::MatMulWaveOp* matmulWaveOp,
                    serialize::SerWaveOp& serWaveOp) const
{
#define WAVE_OP matmulWaveOp
    serWaveOp.m_WaveOpType = wave::MatMulWaveOp::gTypeStr();

    KCC_SERIALIZE(BatchingInWave);
    KCC_SERIALIZE(FmapXNum);
    KCC_SERIALIZE(FmapXStep);
    KCC_SERIALIZE(FmapYNum);
    KCC_SERIALIZE(FmapYStep);
    KCC_SERIALIZE(FmapZNum);
    KCC_SERIALIZE(FmapZStepAtoms);
    KCC_SERIALIZE(IfmapCount);
    KCC_SERIALIZE(IfmapTileHeight);
    KCC_SERIALIZE(IfmapTileWidth);
    KCC_SERIALIZE(IfmapsAtomId);
    KCC_SERIALIZE(IfmapsAtomSize);
    KCC_SERIALIZE(IfmapsOffsetInAtom);
    // layer_name
    KCC_SERIALIZE(NumColumnPartitions);
    KCC_SERIALIZE(NumRowPartitions);
    KCC_SERIALIZE(OfmapCount);
    KCC_SERIALIZE(OfmapTileHeight);
    KCC_SERIALIZE(OfmapTileWidth);
    // previous layers
    KCC_SERIALIZE(PsumBankId);
    KCC_SERIALIZE(PsumBankOffset);
    KCC_SERIALIZE(PsumXNum);
    KCC_SERIALIZE(PsumXStep);
    KCC_SERIALIZE(PsumYNum);
    KCC_SERIALIZE(PsumYStep);
    serWaveOp.m_StartTensorCalc = matmulWaveOp->qStartTensorCalc();
    serWaveOp.m_StopTensorCalc = matmulWaveOp->qStopTensorCalc();
    KCC_SERIALIZE(StrideX);
    KCC_SERIALIZE(StrideY);
    KCC_SERIALIZE(WaveId);
    KCC_SERIALIZE(WaveIdFormat);
    // waveop name
    // waveop type
    KCC_SERIALIZE(WeightsAtomId);
    KCC_SERIALIZE(WeightsOffsetInAtom);
#undef WAVE_OP
}




void
Network::savePool(const wave::PoolWaveOp* poolWaveOp,
                    serialize::SerWaveOp& serWaveOp) const
{
#define WAVE_OP poolWaveOp
    serWaveOp.m_WaveOpType = wave::PoolWaveOp::gTypeStr();

    KCC_SERIALIZE(DstSbAtomId);
    KCC_SERIALIZE(DstSbOffsetInAtom);
    KCC_SERIALIZE(DstXNum);
    KCC_SERIALIZE(DstXStep);
    KCC_SERIALIZE(DstYNum);
    KCC_SERIALIZE(DstYStep);
    KCC_SERIALIZE(DstZNum);
    KCC_SERIALIZE(DstZStep);
    serWaveOp.m_InDtype  = poolWaveOp->gInDtype().gName();
    KCC_SERIALIZE(NumPartitions);
    serWaveOp.m_OutDtype = poolWaveOp->gOutDtype().gName();
    KCC_SERIALIZE(PoolFrequency);
    serWaveOp.m_PoolFunc                = utils::poolType2Str(poolWaveOp->gPoolFunc());
    serWaveOp.m_SrcIsPsum               = poolWaveOp->qSrcIsPsum();
    KCC_SERIALIZE(SrcPsumBankId);
    KCC_SERIALIZE(SrcPsumBankOffset);
    KCC_SERIALIZE(SrcSbAtomId);
    KCC_SERIALIZE(SrcSbOffsetInAtom);
    KCC_SERIALIZE(SrcWNum);
    KCC_SERIALIZE(SrcWStep);
    KCC_SERIALIZE(SrcXNum);
    KCC_SERIALIZE(SrcXStep);
    KCC_SERIALIZE(SrcYNum);
    KCC_SERIALIZE(SrcYStep);
    KCC_SERIALIZE(SrcZNum);
    KCC_SERIALIZE(SrcZStep);

    for (unsigned int i = 0; i < poolWaveOp->gTileId().size(); ++i) {
        serWaveOp.m_TileId[i] = poolWaveOp->gTileId()[i];
    }
    KCC_SERIALIZE(TileIdFormat);
#undef WAVE_OP
}

void
Network::saveSbAtom(const wave::SbAtomWaveOp* sbatomWaveOp,
                    serialize::SerWaveOp& serWaveOp) const
{
#define WAVE_OP sbatomWaveOp
    KCC_SERIALIZE(AtomId);
    KCC_SERIALIZE(AtomSize);
    KCC_SERIALIZE(BatchFoldIdx);
    serWaveOp.m_DataType = DataType::dataTypeId2Str(
                              sbatomWaveOp->gDataType().gDataTypeId());
    KCC_SERIALIZE(Length);
    KCC_SERIALIZE(OffsetInFile);
    KCC_SERIALIZE(PartitionStepBytes);
    serWaveOp.m_RefFile = sbatomWaveOp->gRefFileName();
    KCC_SERIALIZE(RefFileFormat);
    const std::array<kcc_int32,4>& refFileShape(sbatomWaveOp->gRefFileShape());
    for (unsigned int shapeIdx = 0; shapeIdx < refFileShape.size(); ++shapeIdx) {
        serWaveOp.m_RefFileShape[shapeIdx] = refFileShape[shapeIdx];
    }
#undef WAVE_OP

    if (auto sbatomfileWaveOp = dynamic_cast<const wave::SbAtomFileWaveOp*>(sbatomWaveOp)) {
#define WAVE_OP sbatomfileWaveOp
        serWaveOp.m_WaveOpType = wave::SbAtomFileWaveOp::gTypeStr();
        KCC_SERIALIZE(IfmapCount);
        KCC_SERIALIZE(IfmapsFoldIdx);
        serWaveOp.m_IfmapsReplicate = sbatomfileWaveOp->qIfmapsReplicate();
#undef WAVE_OP
    } else {
#define WAVE_OP sbatomsaveWaveOp
        auto sbatomsaveWaveOp = dynamic_cast<const wave::SbAtomSaveWaveOp*>(sbatomWaveOp);
        assert(sbatomsaveWaveOp && "Wrong SbAtaom WaveOp");
        serWaveOp.m_WaveOpType = wave::SbAtomSaveWaveOp::gTypeStr();
        KCC_SERIALIZE(OfmapCount);
        KCC_SERIALIZE(OfmapsFoldIdx);
#undef WAVE_OP
    }
}


void
Network::saveActivaton(const wave::ActivationWaveOp* activationWaveOp,
                       serialize::SerWaveOp& serWaveOp) const
{
#define WAVE_OP activationWaveOp
    serWaveOp.m_WaveOpType = wave::ActivationWaveOp::gTypeStr();

    serWaveOp.m_ActivationFunc      = serialize::SerWaveOp::activationType2Str(activationWaveOp->gActivationFunc());
    serWaveOp.m_BiasAddEn           = activationWaveOp->qBiasAddEn();

    //serWaveOp.m_BiasAtomId          = activationWaveOp->gBiasAtomId();
    KCC_SERIALIZE(BiasAtomId);

    KCC_SERIALIZE(BiasOffsetInAtom);

    KCC_SERIALIZE(DstPsumBankId);
    KCC_SERIALIZE(DstXNum);
    KCC_SERIALIZE(DstXStep);
    KCC_SERIALIZE(DstYNum);
    KCC_SERIALIZE(DstYStep);
    KCC_SERIALIZE(DstZNum);
    KCC_SERIALIZE(DstZStep);

    serWaveOp.m_InDtype             = activationWaveOp->gInDtype().gName();
    KCC_SERIALIZE(NumPartitions);
    serWaveOp.m_OutDtype            = activationWaveOp->gOutDtype().gName();

    KCC_SERIALIZE(SrcPsumBankId);
    KCC_SERIALIZE(SrcXNum);
    KCC_SERIALIZE(SrcXStep);
    KCC_SERIALIZE(SrcYNum);
    KCC_SERIALIZE(SrcYStep);
    KCC_SERIALIZE(SrcZNum);
    KCC_SERIALIZE(SrcZStep);

    const std::array<kcc_int32, 4>& tileId(activationWaveOp->gTileId());
    for (unsigned int i = 0; i < tileId.size(); ++i) {
        serWaveOp.m_TileId[i]       = tileId[i];
    }
    KCC_SERIALIZE(TileIdFormat);
#undef WAVE_OP
}
#undef KCC_SERIALIZE

}}



