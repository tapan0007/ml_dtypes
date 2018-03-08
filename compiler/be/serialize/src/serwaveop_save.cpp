#include "serialize/inc/serwaveop.hpp"


namespace kcc {
namespace serialize {
#define KCC_ARCHIVE(X) archive(cereal::make_nvp(KCC_CONCAT(WaveOpKey_,X), KCC_CONCAT(m_,X)))


template<>
void
SerWaveOp::save<cereal::JSONOutputArchive>(cereal::JSONOutputArchive& archive) const
{
    assert(verify());
    KCC_ARCHIVE(WaveOpType);
    KCC_ARCHIVE(WaveOpName);
    KCC_ARCHIVE(LayerName);
    KCC_ARCHIVE(PreviousWaveOps);

    if (m_WaveOpType == WaveOpTypeStr_SBAtomFile ||
        m_WaveOpType == WaveOpTypeStr_SBAtomSave)
    {
       saveSbAtom(archive);
    } else if (m_WaveOpType == WaveOpTypeStr_Pool) {
        savePool(archive);
    } else if (m_WaveOpType == WaveOpTypeStr_MatMul) {
        saveMatMul(archive);
    } else if (m_WaveOpType == WaveOpTypeStr_Activation) {
        saveActivation(archive);
    } else if (m_WaveOpType == WaveOpTypeStr_ResAdd) {
        saveResAdd(archive);
    } else {
        assert(false && "Serialization: unsupported WaveOp");
    }
} // SerWaveOp::save


void
SerWaveOp::saveSbAtom(cereal::JSONOutputArchive& archive) const
{
    KCC_ARCHIVE(AtomId);
    KCC_ARCHIVE(AtomSize);
    KCC_ARCHIVE(BatchFoldIdx);
    KCC_ARCHIVE(DataType);
    KCC_ARCHIVE(Length);
    KCC_ARCHIVE(OffsetInFile);
    KCC_ARCHIVE(PartitionStepBytes);
    KCC_ARCHIVE(RefFile);
    KCC_ARCHIVE(RefFileFormat);
    KCC_ARCHIVE(RefFileShape);
    if (m_WaveOpType == WaveOpTypeStr_SBAtomFile) {
        KCC_ARCHIVE(IfmapCount);
        KCC_ARCHIVE(IfmapsFoldIdx);
        KCC_ARCHIVE(IfmapsReplicate);
    } else {
        KCC_ARCHIVE(OfmapCount);
        KCC_ARCHIVE(OfmapsFoldIdx);
    }
}

void
SerWaveOp::savePool(cereal::JSONOutputArchive& archive) const
{
    KCC_ARCHIVE(DstSbAtomId);
    KCC_ARCHIVE(DstSbOffsetInAtom);
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
        KCC_ARCHIVE(SrcPsumBankId);
        KCC_ARCHIVE(SrcPsumBankOffset);
    } else {
        KCC_ARCHIVE(SrcSbAtomId);
        KCC_ARCHIVE(SrcSbOffsetInAtom);
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
SerWaveOp::saveMatMul(cereal::JSONOutputArchive& archive) const
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
    KCC_ARCHIVE(IfmapsAtomId);
    KCC_ARCHIVE(IfmapsAtomSize);
    KCC_ARCHIVE(IfmapsOffsetInAtom);
    KCC_ARCHIVE(InDtype);
    // layer name
    KCC_ARCHIVE(NumColumnPartitions);
    KCC_ARCHIVE(NumRowPartitions);
    KCC_ARCHIVE(OutDtype);
    KCC_ARCHIVE(OfmapCount);
    KCC_ARCHIVE(OfmapTileHeight);
    KCC_ARCHIVE(OfmapTileWidth);
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
    KCC_ARCHIVE(WeightsAtomId);
    KCC_ARCHIVE(WeightsOffsetInAtom);

    assert(m_WaveIdFormat.size() == WaveIdFormatSize);
    std::vector<int> waveId(WaveIdFormatSize, -1); // undefined value before converting
    m_WaveId.convertTo(m_WaveIdFormat.c_str(), waveId);
    archive(cereal::make_nvp(WaveOpKey_WaveId, waveId));
}


void
SerWaveOp::saveActivation(cereal::JSONOutputArchive& archive) const
{
    KCC_ARCHIVE(ActivationFunc);
    KCC_ARCHIVE(BiasAddEn);
    KCC_ARCHIVE(BiasAtomId);
    KCC_ARCHIVE(BiasOffsetInAtom);

    KCC_ARCHIVE(DstIsPsum);
    if (m_DstIsPsum) {
        KCC_ARCHIVE(DstPsumBankId);
    } else {
        KCC_ARCHIVE(DstSbAtomId);
        KCC_ARCHIVE(DstSbOffsetInAtom);
    }

    KCC_ARCHIVE(DstXNum);
    KCC_ARCHIVE(DstXStep);
    KCC_ARCHIVE(DstYNum);
    KCC_ARCHIVE(DstYStep);
    KCC_ARCHIVE(DstZNum);
    KCC_ARCHIVE(DstZStep);

    KCC_ARCHIVE(InDtype);
    KCC_ARCHIVE(BiasDtype);
    KCC_ARCHIVE(NumPartitions);
    KCC_ARCHIVE(OutDtype);
    KCC_ARCHIVE(SrcPsumBankId);
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
SerWaveOp::saveResAdd(cereal::JSONOutputArchive& archive) const
{
    KCC_ARCHIVE(InADtype);
    KCC_ARCHIVE(InBDtype);
    KCC_ARCHIVE(OutDtype);
    KCC_ARCHIVE(NumPartitions);

    // SrcA
    KCC_ARCHIVE(SrcAIsPsum);
    if (m_SrcAIsPsum) {
        KCC_ARCHIVE(SrcAPsumBankId);
        KCC_ARCHIVE(SrcAPsumBankOffset);
    } else {
        KCC_ARCHIVE(SrcASbAtomId);
        KCC_ARCHIVE(SrcASbOffsetInAtom);
    }
    KCC_ARCHIVE(SrcAXNum);
    KCC_ARCHIVE(SrcAXStep);
    KCC_ARCHIVE(SrcAYNum);
    KCC_ARCHIVE(SrcAYStep);
    KCC_ARCHIVE(SrcAZNum);
    KCC_ARCHIVE(SrcAZStep);

    // SrcB
    KCC_ARCHIVE(SrcBIsPsum);
    if (m_SrcBIsPsum) {
        KCC_ARCHIVE(SrcBPsumBankId);
        KCC_ARCHIVE(SrcBPsumBankOffset);
    } else {
        KCC_ARCHIVE(SrcBSbAtomId);
        KCC_ARCHIVE(SrcBSbOffsetInAtom);
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
        KCC_ARCHIVE(DstSbAtomId);
        KCC_ARCHIVE(DstSbOffsetInAtom);
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


