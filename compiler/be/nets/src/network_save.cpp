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
    for (unsigned i = 0; i < m_WaveOps.size(); ++i) {
        serialize::SerWaveOp& serWaveOp(serWaveOps[i]);
        wave::WaveOp* waveOp = m_WaveOps[i];
        serWaveOp.m_WaveOpName = waveOp->gName();
        serWaveOp.m_LayerName = waveOp->gLayer()->gName();
        for (auto prevWaveOp : waveOp->gPrevWaveOps()) {
            serWaveOp.addPreviousWaveOp(prevWaveOp->gName());
        }

        if (const auto sbatomfileWaveOp = dynamic_cast<wave::SbAtomFileWaveOp*>(waveOp)) {
            serWaveOp.m_WaveOpType = wave::SbAtomFileWaveOp::gTypeStr();

            serWaveOp.m_AtomId = sbatomfileWaveOp->gAtomId();
            serWaveOp.m_BatchFoldIdx = sbatomfileWaveOp->gBatchFoldIdx();
            serWaveOp.m_Length = sbatomfileWaveOp->gLength();
            serWaveOp.m_OffsetInFile = sbatomfileWaveOp->gOffsetInFile();
            serWaveOp.m_RefFile = sbatomfileWaveOp->gRefFileName();

            serWaveOp.m_IfmapsFoldIdx = sbatomfileWaveOp->gIfmapsFoldIdx();
            serWaveOp.m_IfmapsReplicate = sbatomfileWaveOp->qIfmapsReplicate();
            continue;
        }

        if (const auto sbatomsaveWaveOp = dynamic_cast<wave::SbAtomSaveWaveOp*>(waveOp)) {
            serWaveOp.m_WaveOpType = wave::SbAtomSaveWaveOp::gTypeStr();

            serWaveOp.m_AtomId = sbatomsaveWaveOp->gAtomId();
            serWaveOp.m_BatchFoldIdx = sbatomsaveWaveOp->gBatchFoldIdx();
            serWaveOp.m_Length = sbatomsaveWaveOp->gLength();
            serWaveOp.m_OffsetInFile = sbatomsaveWaveOp->gOffsetInFile();
            serWaveOp.m_RefFile = sbatomsaveWaveOp->gRefFileName();

            serWaveOp.m_OfmapsFoldIdx = sbatomsaveWaveOp->gOfmapsFoldIdx();
            continue;
        }

        if (const auto matmulWaveOp = dynamic_cast<wave::MatMulWaveOp*>(waveOp)) {
            serWaveOp.m_WaveOpType = wave::MatMulWaveOp::gTypeStr();

            serWaveOp.m_BatchingInWave = matmulWaveOp->gBatchingInWave();
            serWaveOp.m_IfmapCount = matmulWaveOp->gIfmapCount();
            serWaveOp.m_IfmapTileHeight = matmulWaveOp->gIfmapTileHeight();
            serWaveOp.m_IfmapTileWidth = matmulWaveOp->gIfmapTileWidth();
            serWaveOp.m_IfmapsAtomId = matmulWaveOp->gIfmapsAtomId();
            serWaveOp.m_IfmapsOffsetInAtom = matmulWaveOp->gIfmapsOffsetInAtom();
            // layer_name
            serWaveOp.m_OfmapCount = matmulWaveOp->gOfmapCount();
            serWaveOp.m_OfmapTileHeight = matmulWaveOp->gOfmapTileHeight();
            serWaveOp.m_OfmapTileWidth = matmulWaveOp->gOfmapTileWidth();
            // previous layers
            serWaveOp.m_PsumBankId = matmulWaveOp->gPsumBankId();
            serWaveOp.m_PsumBankOffset = matmulWaveOp->gPsumBankOffset();
            serWaveOp.m_StartTensorCalc = matmulWaveOp->qStartTensorCalc();
            serWaveOp.m_StopTensorCalc = matmulWaveOp->qStopTensorCalc();
            serWaveOp.m_WaveId = matmulWaveOp->gWaveId();
            serWaveOp.m_WaveIdFormat = matmulWaveOp->gWaveIdFormat();
            // waveop name
            // waveop type
            serWaveOp.m_WeightsAtomId = matmulWaveOp->gWeightsAtomId();
            serWaveOp.m_WeightsOffsetInAtom = matmulWaveOp->gWeightsOffsetInAtom();
            continue;
        }
        assert(false && "Unsupported WaveOp");
    }
    archive(cereal::make_nvp(NetKey_WaveOps, serWaveOps));
}

}}



