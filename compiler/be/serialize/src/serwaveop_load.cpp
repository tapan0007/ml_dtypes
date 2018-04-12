#include "serialize/inc/serwaveop.hpp"

#include "wave/inc/matmulwaveop.hpp"
#include "wave/inc/sbatomloadwaveop.hpp"
#include "wave/inc/sbatomsavewaveop.hpp"
#include "wave/inc/poolwaveop.hpp"

namespace kcc {
namespace serialize {
#define KCC_ARCHIVE(X) archive(cereal::make_nvp(KCC_CONCAT(WaveOpKey_,X), KCC_CONCAT(m_,X)))


template<>
void
SerWaveOp::load<cereal::JSONInputArchive>(cereal::JSONInputArchive& archive)
{
    KCC_ARCHIVE(WaveOpType);
    KCC_ARCHIVE(WaveOpName);
    KCC_ARCHIVE(LayerName);
    KCC_ARCHIVE(PreviousWaveOps);

    if (m_WaveOpType == WaveOpTypeStr_SBAtomLoad ||
        m_WaveOpType == WaveOpTypeStr_SBAtomSave)
    {
        loadSbAtom(archive);
    } else if (m_WaveOpType == WaveOpTypeStr_Pool) {
        loadPool(archive);
    } else if (m_WaveOpType == WaveOpTypeStr_MatMul) {
        loadMatMul(archive);
    } else if (m_WaveOpType == WaveOpTypeStr_Activation) {
        loadActivation(archive);
    } else if (m_WaveOpType == WaveOpTypeStr_ResAdd) {
        loadResAdd(archive);

    } else {
        assert(false && "Serialization: unsupported WaveOp");
    }
    assert(verify());
} // SerWaveOp::load


void
SerWaveOp::loadSbAtom(cereal::JSONInputArchive& archive)
{
    KCC_ARCHIVE(SbAddress);
    KCC_ARCHIVE(BatchFoldIdx);
    KCC_ARCHIVE(DataType);
    KCC_ARCHIVE(Length);
    KCC_ARCHIVE(OffsetInFile);
    KCC_ARCHIVE(PartitionStepBytes);
    KCC_ARCHIVE(RefFile);
    KCC_ARCHIVE(RefFileFormat);
    KCC_ARCHIVE(RefFileShape);

    if (m_WaveOpType == WaveOpTypeStr_SBAtomLoad) {
        KCC_ARCHIVE(IfmapCount);
        KCC_ARCHIVE(IfmapsFoldIdx);
        KCC_ARCHIVE(IfmapsReplicate);
        KCC_ARCHIVE(ContainWeights);
    } else {
        KCC_ARCHIVE(OfmapCount);
        KCC_ARCHIVE(OfmapsFoldIdx);
    }
}

void
SerWaveOp::loadPool(cereal::JSONInputArchive& archive)
{
    KCC_ARCHIVE(DstSbAddress);
    KCC_ARCHIVE(DstXNum);
    KCC_ARCHIVE(DstXStep);
    KCC_ARCHIVE(DstYNum);
    KCC_ARCHIVE(DstYStep);
    KCC_ARCHIVE(DstZNum);
    KCC_ARCHIVE(DstZStep);
    KCC_ARCHIVE(InDtype);
    // layername
    KCC_ARCHIVE(NumPartitions);
    KCC_ARCHIVE(OutDtype);
    KCC_ARCHIVE(PoolFrequency);
    KCC_ARCHIVE(PoolFunc);
    // previouswaveops

    KCC_ARCHIVE(SrcIsPsum);
    if (m_SrcIsPsum) {
        KCC_ARCHIVE(SrcPsumBankOffset);
        KCC_ARCHIVE(SrcPsumBankId);
    } else {
        KCC_ARCHIVE(SrcSbAddress);
    }
    KCC_ARCHIVE(SrcWNum);
    KCC_ARCHIVE(SrcWStep);
    KCC_ARCHIVE(SrcXNum);
    KCC_ARCHIVE(SrcXStep);
    KCC_ARCHIVE(SrcYNum);
    KCC_ARCHIVE(SrcYStep);
    KCC_ARCHIVE(SrcZNum);
    KCC_ARCHIVE(SrcZStep);

    KCC_ARCHIVE(TileId);
    KCC_ARCHIVE(TileIdFormat);
}

void
SerWaveOp::loadMatMul(cereal::JSONInputArchive& archive)
{
    KCC_ARCHIVE(BatchingInWave);
    KCC_ARCHIVE(FmapXNum);
    KCC_ARCHIVE(FmapXStep);
    KCC_ARCHIVE(FmapYNum);
    KCC_ARCHIVE(FmapYStep);
    KCC_ARCHIVE(FmapZNum);
    KCC_ARCHIVE(FmapZStepAtoms);
    KCC_ARCHIVE(IfmapCount);
    KCC_ARCHIVE(IfmapTileHeight);
    KCC_ARCHIVE(IfmapTileWidth);
    KCC_ARCHIVE(IfmapsSbAddress);
    KCC_ARCHIVE(InDtype);
    // layer name
    KCC_ARCHIVE(NumColumnPartitions);
    KCC_ARCHIVE(NumRowPartitions);
    KCC_ARCHIVE(OfmapCount);
    KCC_ARCHIVE(OfmapTileHeight);
    KCC_ARCHIVE(OfmapTileWidth);
    KCC_ARCHIVE(OutDtype);
    // previous waveops
    KCC_ARCHIVE(PsumBankId);
    KCC_ARCHIVE(PsumBankOffset);
    KCC_ARCHIVE(PsumXNum);
    KCC_ARCHIVE(PsumXStep);
    KCC_ARCHIVE(PsumYNum);
    KCC_ARCHIVE(PsumYStep);
    KCC_ARCHIVE(StartTensorCalc);
    KCC_ARCHIVE(StopTensorCalc);
    KCC_ARCHIVE(StrideX);
    KCC_ARCHIVE(StrideY);
    KCC_ARCHIVE(WaveIdFormat);
    // waveop name
    // waveop type
    KCC_ARCHIVE(WeightsSbAddress);

    assert(m_WaveIdFormat.size() == WaveIdFormatSize);
    std::vector<int> waveId(WaveIdFormatSize, -1); // undefined value before deserial
    archive(cereal::make_nvp(WaveOpKey_WaveId, waveId));
    assert(WaveIdFormatSize == waveId.size() && "Number of element in WaveId wrong");
    m_WaveId.convertFrom(m_WaveIdFormat.c_str(), waveId);
}

void
SerWaveOp::loadActivation(cereal::JSONInputArchive& archive)
{

    //archive(cereal::make_nvp(WaveOpKey_ActivationFunc, m_ActivationFunc);
    KCC_ARCHIVE(ActivationFunc);
    KCC_ARCHIVE(BiasAddEn);
    KCC_ARCHIVE(BiasSbAddress);

    KCC_ARCHIVE(DstXNum);
    KCC_ARCHIVE(DstXStep);
    KCC_ARCHIVE(DstYNum);
    KCC_ARCHIVE(DstYStep);
    KCC_ARCHIVE(DstZNum);
    KCC_ARCHIVE(DstZStep);

    KCC_ARCHIVE(DstIsPsum);
    if (m_DstIsPsum) {
        KCC_ARCHIVE(DstPsumBankId);
    } else {
        KCC_ARCHIVE(DstSbAddress);
    }

    KCC_ARCHIVE(InDtype);
    KCC_ARCHIVE(BiasDtype);
    KCC_ARCHIVE(NumPartitions);
    KCC_ARCHIVE(OutDtype);

    KCC_ARCHIVE(SrcPsumBankId);
    KCC_ARCHIVE(SrcXNum);
    KCC_ARCHIVE(SrcXStep);
    KCC_ARCHIVE(SrcYNum);
    KCC_ARCHIVE(SrcYStep);
    //KCC_ARCHIVE(SrcZNum);
    //KCC_ARCHIVE(SrcZStep);
    KCC_ARCHIVE(TileId);
    KCC_ARCHIVE(TileIdFormat);
}


void
SerWaveOp::loadResAdd(cereal::JSONInputArchive& archive)
{

    //archive(cereal::make_nvp(WaveOpKey_ActivationFunc, m_ActivationFunc);

    KCC_ARCHIVE(InADtype);
    KCC_ARCHIVE(InBDtype);
    KCC_ARCHIVE(OutDtype);
    KCC_ARCHIVE(NumPartitions);
    KCC_ARCHIVE(Multiply);

    // Src A
    KCC_ARCHIVE(SrcAIsPsum);
    if (m_SrcAIsPsum) {
        KCC_ARCHIVE(SrcAPsumBankId);
        KCC_ARCHIVE(SrcAPsumBankOffset);
    } else {
        KCC_ARCHIVE(SrcASbAddress);
    }
    KCC_ARCHIVE(SrcAXNum);
    KCC_ARCHIVE(SrcAXStep);
    KCC_ARCHIVE(SrcAYNum);
    KCC_ARCHIVE(SrcAYStep);
    KCC_ARCHIVE(SrcAZNum);
    KCC_ARCHIVE(SrcAZStep);

    // Src B
    KCC_ARCHIVE(SrcBIsPsum);
    if (m_SrcBIsPsum) {
        KCC_ARCHIVE(SrcBPsumBankId);
        KCC_ARCHIVE(SrcBPsumBankOffset);
    } else {
        KCC_ARCHIVE(SrcBSbAddress);
    }
    KCC_ARCHIVE(SrcBXNum);
    KCC_ARCHIVE(SrcBXStep);
    KCC_ARCHIVE(SrcBYNum);
    KCC_ARCHIVE(SrcBYStep);
    KCC_ARCHIVE(SrcBZNum);
    KCC_ARCHIVE(SrcBZStep);

    // Dst
    KCC_ARCHIVE(DstIsPsum);
    if (m_DstIsPsum) {
        KCC_ARCHIVE(DstPsumBankId);
        KCC_ARCHIVE(DstPsumBankOffset);
    } else {
        KCC_ARCHIVE(DstSbAddress);
    }
    KCC_ARCHIVE(DstXNum);
    KCC_ARCHIVE(DstXStep);
    KCC_ARCHIVE(DstYNum);
    KCC_ARCHIVE(DstYStep);
    KCC_ARCHIVE(DstZNum);
    KCC_ARCHIVE(DstZStep);
}

#undef KCC_ARCHIVE


} // namespace serialize
} // namespace kcc


