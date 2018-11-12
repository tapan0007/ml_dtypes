#include "utils/inc/asserter.hpp"

#include "wave/inc/waveconsts.hpp"
#include "serialize/inc/serwaveop.hpp"


namespace kcc {
namespace serialize {
#define KCC_ARCHIVE(X) archive(cereal::make_nvp(KCC_CONCAT(WaveOpKey_,X), KCC_CONCAT(m_,X)))


//===========================================================================

//===========================================================================
template<>
void
SerWaveOp::save<cereal::JSONOutputArchive>(cereal::JSONOutputArchive& archive) const
{
    assert(verify());
    KCC_ARCHIVE(WaveOpType);
    KCC_ARCHIVE(WaveOpName);
    KCC_ARCHIVE(LayerName);
    KCC_ARCHIVE(PreviousWaveOps);

    if (m_PreviousSyncs.size() > 0) {
        KCC_ARCHIVE(PreviousSyncs);
    }

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
    } else if (m_WaveOpType == wave::WaveOpTypeStr_ScaleAdd) {
        saveScaleAdd(archive);
    } else if (m_WaveOpType == wave::WaveOpTypeStr_Barrier) {
        saveBarrier(archive);
    } else if (m_WaveOpType == wave::WaveOpTypeStr_Nop) {
        saveNop(archive);
    } else if (m_WaveOpType == wave::WaveOpTypeStr_Minimum) {
        saveMinimum(archive);
    } else if (m_WaveOpType == wave::WaveOpTypeStr_Maximum) {
        saveMaximum(archive);
    } else if (m_WaveOpType == wave::WaveOpTypeStr_Add) {
        saveAdd(archive);
    } else if (m_WaveOpType == wave::WaveOpTypeStr_Sub) {
        saveSub(archive);
    } else if (m_WaveOpType == wave::WaveOpTypeStr_Multiply) {
        saveMult(archive);
    } else {
        Assert(false, "Serialization: unsupported WaveOp ", m_WaveOpType);
    }
} // SerWaveOp::save


//===========================================================================
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

//===========================================================================
void
SerWaveOp::savePool(cereal::JSONOutputArchive& archive) const
{
    saveSrc(archive, Dims::XYZW);
    saveDst(archive, Dims::XYZ);
    KCC_ARCHIVE(DstStartAtMidPart);

    KCC_ARCHIVE(NumPartitions);
    KCC_ARCHIVE(PoolFrequency);
    KCC_ARCHIVE(PoolFunc);

    KCC_ARCHIVE(TileId);
    KCC_ARCHIVE(TileIdFormat);
}


//===========================================================================
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


//===========================================================================
void
SerWaveOp::saveActivation(cereal::JSONOutputArchive& archive) const
{
    saveSrc(archive, Dims::XYZ);
    saveDst(archive, Dims::XYZ);

    KCC_ARCHIVE(ActivationFunc);
    KCC_ARCHIVE(BiasAddEn);
    KCC_ARCHIVE(BiasSbAddress);
    KCC_ARCHIVE(BiasStartAtMidPart);

    KCC_ARCHIVE(BiasDtype);
    KCC_ARCHIVE(NumPartitions);

    KCC_ARCHIVE(TileId);
    KCC_ARCHIVE(TileIdFormat);
}


//===========================================================================
void
SerWaveOp::saveClipByValue(cereal::JSONOutputArchive& archive) const
{
    saveSrc(archive, Dims::XYZ);
    saveDst(archive, Dims::XYZ);

    KCC_ARCHIVE(NumPartitions);
    KCC_ARCHIVE(TileId);
    KCC_ARCHIVE(TileIdFormat);

    KCC_ARCHIVE(MinValue);
    KCC_ARCHIVE(MaxValue);
}

//===========================================================================
void
SerWaveOp::saveResAdd(cereal::JSONOutputArchive& archive) const
{
    saveSrcAB(archive, Dims::XYZ);
    saveDst(archive, Dims::XYZ);

    KCC_ARCHIVE(NumPartitions);
}

//===========================================================================
void
SerWaveOp::saveMaximum(cereal::JSONOutputArchive& archive) const
{
    KCC_ARCHIVE(IsScalarOp);
    if (m_IsScalarOp) {
        saveSrc(archive, Dims::XYZ);
    } else {
        saveSrcAB(archive, Dims::XYZ);
    }
    saveDst(archive, Dims::XYZ);

    KCC_ARCHIVE(NumPartitions);
}

//===========================================================================
void
SerWaveOp::saveMinimum(cereal::JSONOutputArchive& archive) const
{
    KCC_ARCHIVE(IsScalarOp);
    if (m_IsScalarOp) {
        saveSrc(archive, Dims::XYZ);
    } else {
        saveSrcAB(archive, Dims::XYZ);
    }
    saveDst(archive, Dims::XYZ);

    KCC_ARCHIVE(NumPartitions);
}

//===========================================================================
void
SerWaveOp::saveAdd(cereal::JSONOutputArchive& archive) const
{
    KCC_ARCHIVE(IsScalarOp);
    if (m_IsScalarOp) {
        saveSrc(archive, Dims::XYZ);
    } else {
        saveSrcAB(archive, Dims::XYZ);
    }
    saveDst(archive, Dims::XYZ);

    KCC_ARCHIVE(NumPartitions);
}

//===========================================================================
void
SerWaveOp::saveSub(cereal::JSONOutputArchive& archive) const
{
    KCC_ARCHIVE(IsScalarOp);
    if (m_IsScalarOp) {
        saveSrc(archive, Dims::XYZ);
    } else {
        saveSrcAB(archive, Dims::XYZ);
    }
    saveDst(archive, Dims::XYZ);

    KCC_ARCHIVE(NumPartitions);
}

//===========================================================================
void
SerWaveOp::saveMult(cereal::JSONOutputArchive& archive) const
{
    KCC_ARCHIVE(IsScalarOp);
    if (m_IsScalarOp) {
        saveSrc(archive, Dims::XYZ);
    } else {
        saveSrcAB(archive, Dims::XYZ);
    }
    saveDst(archive, Dims::XYZ);

    KCC_ARCHIVE(NumPartitions);
}

//===========================================================================
void
SerWaveOp::saveScaleAdd(cereal::JSONOutputArchive& archive) const
{
    saveSrc(archive, Dims::XYZ);
    saveDst(archive, Dims::XYZ);

    KCC_ARCHIVE(NumPartitions);
    KCC_ARCHIVE(Add);
    KCC_ARCHIVE(Scale);
}




//===========================================================================
void
SerWaveOp::saveBarrier(cereal::JSONOutputArchive& /*archive*/) const
{
}

//===========================================================================
void
SerWaveOp::saveNop(cereal::JSONOutputArchive& archive) const
{
    KCC_ARCHIVE(Engine);
}




//===========================================================================
void
SerWaveOp::saveSrc(cereal::JSONOutputArchive& archive, Dims dims) const
{
    KCC_ARCHIVE(InDtype);
    KCC_ARCHIVE(SrcIsPsum);
    if (m_SrcIsPsum) {
        KCC_ARCHIVE(SrcPsumBankId);
        KCC_ARCHIVE(SrcPsumBankOffset);
    } else {
        KCC_ARCHIVE(SrcSbAddress);
        KCC_ARCHIVE(SrcStartAtMidPart);
    }
    switch (dims) {
    case Dims::XYZW:
        KCC_ARCHIVE(SrcWNum);
        KCC_ARCHIVE(SrcWStep);
        // Fall through!
    case Dims::XYZ:
        KCC_ARCHIVE(SrcZNum);
        KCC_ARCHIVE(SrcZStep);
        // Fall through!
    case Dims::XY:
        KCC_ARCHIVE(SrcYNum);
        KCC_ARCHIVE(SrcYStep);
        // Fall through!
    case Dims::X:
        KCC_ARCHIVE(SrcXNum);
        KCC_ARCHIVE(SrcXStep);
        break;
    default:
        Assert(false, "Dims to save Src are wrong");
    }
}

//===========================================================================
void
SerWaveOp::saveSrcA(cereal::JSONOutputArchive& archive, Dims dims) const
{
    KCC_ARCHIVE(InADtype);
    KCC_ARCHIVE(SrcAIsPsum);
    if (m_SrcAIsPsum) {
        KCC_ARCHIVE(SrcAPsumBankId);
        KCC_ARCHIVE(SrcAPsumBankOffset);
    } else {
        KCC_ARCHIVE(SrcASbAddress);
        KCC_ARCHIVE(SrcAStartAtMidPart);
    }
    switch (dims) {
    case Dims::XYZW:
        KCC_ARCHIVE(SrcAWNum);
        KCC_ARCHIVE(SrcAWStep);
        // Fall through!
    case Dims::XYZ:
        KCC_ARCHIVE(SrcAZNum);
        KCC_ARCHIVE(SrcAZStep);
        // Fall through!
    case Dims::XY:
        KCC_ARCHIVE(SrcAYNum);
        KCC_ARCHIVE(SrcAYStep);
        // Fall through!
    case Dims::X:
        KCC_ARCHIVE(SrcAXNum);
        KCC_ARCHIVE(SrcAXStep);
        break;
    default:
        Assert(false, "Dims to save SrcB are wrong");
    }
}

//===========================================================================
void
SerWaveOp::saveSrcB(cereal::JSONOutputArchive& archive, Dims dims) const
{
    KCC_ARCHIVE(InBDtype);
    KCC_ARCHIVE(SrcBIsPsum);
    if (m_SrcBIsPsum) {
        KCC_ARCHIVE(SrcBPsumBankId);
        KCC_ARCHIVE(SrcBPsumBankOffset);
    } else {
        KCC_ARCHIVE(SrcBSbAddress);
        KCC_ARCHIVE(SrcBStartAtMidPart);
    }
    switch (dims) {
    case Dims::XYZW:
        KCC_ARCHIVE(SrcBWNum);
        KCC_ARCHIVE(SrcBWStep);
        // Fall through!
    case Dims::XYZ:
        KCC_ARCHIVE(SrcBZNum);
        KCC_ARCHIVE(SrcBZStep);
        // Fall through!
    case Dims::XY:
        KCC_ARCHIVE(SrcBYNum);
        KCC_ARCHIVE(SrcBYStep);
        // Fall through!
    case Dims::X:
        KCC_ARCHIVE(SrcBXNum);
        KCC_ARCHIVE(SrcBXStep);
        break;
    default:
        Assert(false, "Dims to save SrcB are wrong");
    }
}

//===========================================================================
void
SerWaveOp::saveSrcAB(cereal::JSONOutputArchive& archive, Dims dims) const
{
    saveSrcA(archive, dims);
    saveSrcB(archive, dims);
}

//===========================================================================
void
SerWaveOp::saveDst(cereal::JSONOutputArchive& archive, Dims dims) const
{
    KCC_ARCHIVE(OutDtype);
    KCC_ARCHIVE(DstIsPsum);
    if (m_DstIsPsum) {
        KCC_ARCHIVE(DstPsumBankId);
        KCC_ARCHIVE(DstPsumBankOffset);
    } else {
        KCC_ARCHIVE(DstSbAddress);
        KCC_ARCHIVE(DstStartAtMidPart);
    }
    switch (dims) {
    //case Dims::XYZW:
    //    KCC_ARCHIVE(DstWNum);
    //    KCC_ARCHIVE(DstWStep);
        // Fall through!
    case Dims::XYZ:
        KCC_ARCHIVE(DstZNum);
        KCC_ARCHIVE(DstZStep);
        // Fall through!
    case Dims::XY:
        KCC_ARCHIVE(DstYNum);
        KCC_ARCHIVE(DstYStep);
        // Fall through!
    case Dims::X:
        KCC_ARCHIVE(DstXNum);
        KCC_ARCHIVE(DstXStep);
        break;
    default:
        Assert(false, "Dims to save Dst are wrong");
    }
}


#undef KCC_ARCHIVE
} // namespace serialize
} // namespace kcc


