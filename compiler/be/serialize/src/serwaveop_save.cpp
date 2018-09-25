#include "wave/inc/waveconsts.hpp"
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
    KCC_ARCHIVE(PreviousEventIds);
    KCC_ARCHIVE(PreviousEventWaitModes);
    KCC_ARCHIVE(PreviousEventSetModes);
    KCC_ARCHIVE(Order);


    if (m_WaveOpType == wave::WaveOpTypeStr_SBAtomLoad ||
        m_WaveOpType == wave::WaveOpTypeStr_SBAtomSave)
    {
       saveSbAtom(archive);
    } else if (m_WaveOpType == wave::WaveOpTypeStr_Pool) {
        savePool(archive);
    } else if (m_WaveOpType == wave::WaveOpTypeStr_MatMul) {
        saveMatMul(archive);
    } else if (m_WaveOpType == wave::WaveOpTypeStr_Activation) {
        saveActivation(archive);
    } else if (m_WaveOpType == wave::WaveOpTypeStr_ClipByValue) {
        saveClipByValue(archive);
    } else if (m_WaveOpType == wave::WaveOpTypeStr_ResAdd) {
        saveResAdd(archive);
    } else if (m_WaveOpType == wave::WaveOpTypeStr_Barrier) {
        saveBarrier(archive);
    } else if (m_WaveOpType == wave::WaveOpTypeStr_Nop) {
        saveNop(archive);
    } else {
        assert(false && "Serialization: unsupported WaveOp");
    }
} // SerWaveOp::save


void
SerWaveOp::saveSbAtom(cereal::JSONOutputArchive& archive) const
{
    KCC_ARCHIVE(Engine);
    KCC_ARCHIVE(SbAddress);
    KCC_ARCHIVE(DataType);
    KCC_ARCHIVE(Length);
    KCC_ARCHIVE(OffsetInFile);
    KCC_ARCHIVE(PartitionStepBytes);
    KCC_ARCHIVE(RefFile);
    KCC_ARCHIVE(RefFileFormat);
    KCC_ARCHIVE(RefFileShape);
    if (m_WaveOpType == wave::WaveOpTypeStr_SBAtomLoad) {
        KCC_ARCHIVE(NumPartitions);
        KCC_ARCHIVE(ContainWeights);

        KCC_ARCHIVE(IfmapReplicationNumRows);
        KCC_ARCHIVE(IfmapReplicationResolution);
        KCC_ARCHIVE(IfmapReplicationStepBytes);

        KCC_ARCHIVE(SrcStepElem);
    } else {
        KCC_ARCHIVE(NumPartitions);
        KCC_ARCHIVE(FinalLayerOfmap);
    }
} // SerWaveOp::saveSbAtom

void
SerWaveOp::savePool(cereal::JSONOutputArchive& archive) const
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
        KCC_ARCHIVE(SrcPsumBankId);
        KCC_ARCHIVE(SrcPsumBankOffset);
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
SerWaveOp::saveMatMul(cereal::JSONOutputArchive& archive) const
{
    KCC_ARCHIVE(FmapXNum);
    KCC_ARCHIVE(FmapXStep);
    KCC_ARCHIVE(FmapYNum);
    KCC_ARCHIVE(FmapYStep);
    KCC_ARCHIVE(FmapZNum);
    KCC_ARCHIVE(FmapZStep);
    KCC_ARCHIVE(IfmapsSbAddress);
    KCC_ARCHIVE(InDtype);
    // layer name
    KCC_ARCHIVE(NumColumnPartitions);
    KCC_ARCHIVE(NumRowPartitions);
    KCC_ARCHIVE(OutDtype);
    // previous waveops
    KCC_ARCHIVE(PsumBankId);
    KCC_ARCHIVE(PsumBankOffset);
    KCC_ARCHIVE(PsumXNum);
    KCC_ARCHIVE(PsumXStep);
    KCC_ARCHIVE(PsumYNum);
    KCC_ARCHIVE(PsumYStep);
    KCC_ARCHIVE(PsumZNum);
    KCC_ARCHIVE(PsumZStep);
    KCC_ARCHIVE(StartTensorCalc);
    KCC_ARCHIVE(StopTensorCalc);
    // waveop name
    // waveop type
    KCC_ARCHIVE(WeightsSbAddress);

    KCC_ARCHIVE(IfmapReplicationNumRows);
    KCC_ARCHIVE(IfmapReplicationResolution);
    KCC_ARCHIVE(IfmapReplicationShiftAmnt);

} // SerWaveOp::saveMatMul


void
SerWaveOp::saveActivation(cereal::JSONOutputArchive& archive) const
{
    KCC_ARCHIVE(ActivationFunc);
    KCC_ARCHIVE(BiasAddEn);
    KCC_ARCHIVE(BiasSbAddress);

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

    KCC_ARCHIVE(InDtype);
    KCC_ARCHIVE(BiasDtype);
    KCC_ARCHIVE(NumPartitions);
    KCC_ARCHIVE(OutDtype);
    KCC_ARCHIVE(SrcIsPsum);
    if (m_SrcIsPsum) {
        KCC_ARCHIVE(SrcPsumBankId);
        KCC_ARCHIVE(SrcPsumBankOffset);
    } else {
        KCC_ARCHIVE(SrcSbAddress);
    }

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
SerWaveOp::saveClipByValue(cereal::JSONOutputArchive& archive) const
{
    KCC_ARCHIVE(InDtype);
    KCC_ARCHIVE(OutDtype);
    KCC_ARCHIVE(SrcIsPsum);
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

    if (m_SrcIsPsum) {
        KCC_ARCHIVE(SrcPsumBankId);
        KCC_ARCHIVE(SrcPsumBankOffset);
    } else {
        KCC_ARCHIVE(SrcSbAddress);
        KCC_ARCHIVE(SrcStartAtMidPart);
    }

    KCC_ARCHIVE(SrcXNum);
    KCC_ARCHIVE(SrcXStep);
    KCC_ARCHIVE(SrcYNum);
    KCC_ARCHIVE(SrcYStep);
    KCC_ARCHIVE(SrcZNum);
    KCC_ARCHIVE(SrcZStep);

    KCC_ARCHIVE(NumPartitions);
    KCC_ARCHIVE(TileId);
    KCC_ARCHIVE(TileIdFormat);

    KCC_ARCHIVE(MinValue);
    KCC_ARCHIVE(MaxValue);
}

void
SerWaveOp::saveResAdd(cereal::JSONOutputArchive& archive) const
{
    KCC_ARCHIVE(InADtype);
    KCC_ARCHIVE(InBDtype);
    KCC_ARCHIVE(OutDtype);
    KCC_ARCHIVE(NumPartitions);
    KCC_ARCHIVE(Multiply);

    // SrcA
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

    // SrcB
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

void
SerWaveOp::saveBarrier(cereal::JSONOutputArchive& /*archive*/) const
{
}

void
SerWaveOp::saveNop(cereal::JSONOutputArchive& archive) const
{
    KCC_ARCHIVE(Engine);
}


#undef KCC_ARCHIVE
} // namespace serialize
} // namespace kcc


