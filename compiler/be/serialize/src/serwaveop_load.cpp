
#include "utils/inc/asserter.hpp"

#include "arch/inc/statebuffer.hpp"
#include "serialize/inc/serwaveop.hpp"

#include "wave/inc/waveconsts.hpp"
#include "wave/inc/matmulwaveop.hpp"
#include "wave/inc/sbatomloadwaveop.hpp"
#include "wave/inc/sbatomsavewaveop.hpp"
#include "wave/inc/poolwaveop.hpp"

namespace kcc {
namespace serialize {
#define KCC_ARCHIVE(X) archive(cereal::make_nvp(WaveOpKey::X, KCC_CONCAT(m_,X)))


template<>
void
SerWaveOp::load<cereal::JSONInputArchive>(cereal::JSONInputArchive& archive)
{
    KCC_ARCHIVE(WaveOpType);
    KCC_ARCHIVE(WaveOpName);
    KCC_ARCHIVE(LayerName);
    KCC_ARCHIVE(PreviousWaveOps);

    if (m_WaveOpType == wave::WaveOpTypeStr::SBAtomLoad ||
        m_WaveOpType == wave::WaveOpTypeStr::SBAtomSave)
    {
        loadSbAtom(archive);
    } else if (m_WaveOpType == wave::WaveOpTypeStr::Pool) {
        loadPool(archive);
    } else if (m_WaveOpType == wave::WaveOpTypeStr::Reciprocal) {
        loadReciprocal(archive);
    } else if (m_WaveOpType == wave::WaveOpTypeStr::RegLoad) {
        loadRegLoad(archive);
    } else if (m_WaveOpType == wave::WaveOpTypeStr::RegStore) {
        loadRegStore(archive);
    } else if (m_WaveOpType == wave::WaveOpTypeStr::MatMul) {
        loadMatMul(archive);
    } else if (m_WaveOpType == wave::WaveOpTypeStr::Activation) {
        loadActivation(archive);
    } else if (m_WaveOpType == wave::WaveOpTypeStr::ResAdd) {
        loadResAdd(archive);
    } else if (m_WaveOpType == wave::WaveOpTypeStr::Multiply) {
        loadMultiply(archive);
    } else if (m_WaveOpType == wave::WaveOpTypeStr::Sub) {
        loadSub(archive);
    } else if (m_WaveOpType == wave::WaveOpTypeStr::Add) {
        loadAdd(archive);
    } else if (m_WaveOpType == wave::WaveOpTypeStr::ClipByValue) {
        loadClipByValue(archive);
    } else if (m_WaveOpType == wave::WaveOpTypeStr::ScaleAdd) {
        loadScaleAdd(archive);
    } else if (m_WaveOpType == wave::WaveOpTypeStr::Maximum) {
        loadMaximum(archive);
    } else if (m_WaveOpType == wave::WaveOpTypeStr::Minimum) {
        loadMinimum(archive);
    } else if (m_WaveOpType == wave::WaveOpTypeStr::TensorTensor) {
        loadTensorTensor(archive);
    } else if (m_WaveOpType == wave::WaveOpTypeStr::TensorScalar) {
        loadTensorScalar(archive);
    } else if (m_WaveOpType == wave::WaveOpTypeStr::TensorScalarPtr) {
        loadTensorScalarPtr(archive);
    } else {
        Assert(false, "Unknown waveop type: ", m_WaveOpType);
    }
    Assert(verify(), "Failed verification on waveop of type ", m_WaveOpType);
} // SerWaveOp::load


void
SerWaveOp::loadSbAtom(cereal::JSONInputArchive& archive)
{
    KCC_ARCHIVE(SbAddress);
    KCC_ARCHIVE(StartAtMidPart);
    KCC_ARCHIVE(DataType);
    KCC_ARCHIVE(Length);
    KCC_ARCHIVE(OffsetInFile);
    KCC_ARCHIVE(PartitionStepBytes);
    KCC_ARCHIVE(RefFile);
    KCC_ARCHIVE(RefFileFormat);
    KCC_ARCHIVE(RefFileShape);

    const arch::StateBuffer stateBuf(arch::Arch::gArch().gStateBuffer());
    if (m_WaveOpType == wave::WaveOpTypeStr::SBAtomLoad) {
        Assert(stateBuf.qTpbWriteAccessCheck(m_SbAddress, m_Length),
            "State buffer write address 0x",
            std::hex, m_SbAddress, std::dec,
            ", size=", m_Length,
            " in SbAtomLoad Waveop '", m_WaveOpName,
            "' not aligned correctly");
        KCC_ARCHIVE(NumPartitions);
        KCC_ARCHIVE(ContainWeights);

        KCC_ARCHIVE(IfmapReplicationNumRows);
        KCC_ARCHIVE(IfmapReplicationResolution);
        KCC_ARCHIVE(IfmapReplicationStepBytes);

        KCC_ARCHIVE(SrcStepElem);
    } else {
        Assert(stateBuf.qTpbReadAccessCheck(m_SbAddress, m_Length),
            "State buffer read address 0x",
            std::hex, m_SbAddress, std::dec,
            ", size=", m_Length,
            " in SbAtomSave Waveop '", m_WaveOpName,
            "' not aligned correctly");
        KCC_ARCHIVE(NumPartitions);
        KCC_ARCHIVE(FinalLayerOfmap);
    }
} // SerWaveOp::loadSbAtom

void
SerWaveOp::loadPool(cereal::JSONInputArchive& archive)
{
    loadSrc(archive, Dims::XYZW);
    loadDst(archive, Dims::XYZ);

    KCC_ARCHIVE(NumPartitions);
    KCC_ARCHIVE(PoolFrequency);
    KCC_ARCHIVE(PoolFunc);
    // previouswaveops

    KCC_ARCHIVE(TileId);
    KCC_ARCHIVE(TileIdFormat);
}

void
SerWaveOp::loadReciprocal(cereal::JSONInputArchive& archive)
{
    loadSrc(archive, Dims::XYZ);
    loadDst(archive, Dims::XYZ);

    KCC_ARCHIVE(NumPartitions);
    // previouswaveops

    KCC_ARCHIVE(TileId);
    KCC_ARCHIVE(TileIdFormat);
}

void
SerWaveOp::loadRegLoad(cereal::JSONInputArchive& archive)
{
    KCC_ARCHIVE(NumPartitions);
    KCC_ARCHIVE(ParallelMode);
    loadSrc(archive, Dims::XYZ);
    KCC_ARCHIVE(InDtype);
    KCC_ARCHIVE(SrcIsPsum);

}

void
SerWaveOp::loadRegStore(cereal::JSONInputArchive& archive)
{
    KCC_ARCHIVE(NumPartitions);
    KCC_ARCHIVE(ParallelMode);
    loadDst(archive, Dims::XYZ);
    KCC_ARCHIVE(OutDtype);
    KCC_ARCHIVE(DstIsPsum);
}

void
SerWaveOp::loadMatMul(cereal::JSONInputArchive& archive)
{
    KCC_ARCHIVE(FmapXNum);
    KCC_ARCHIVE(FmapXStep);
    KCC_ARCHIVE(FmapYNum);
    KCC_ARCHIVE(FmapYStep);
    KCC_ARCHIVE(FmapZNum);
    KCC_ARCHIVE(FmapZStep);
    KCC_ARCHIVE(IfmapsSbAddress);
    KCC_ARCHIVE(InDtype);
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

    KCC_ARCHIVE(IsDynamicWeights);

    if (DataType::qNeedsQuantization(m_InDtype.c_str())) {
        KCC_ARCHIVE(QuantOffsetIfmaps);
        KCC_ARCHIVE(QuantOffsetWeights);
        if (DataTypeId::Uint8 == DataType::dataTypeStr2Id(m_InDtype.c_str())) {
            KCC_ARCHIVE(PEPerfOptMode);
        }
    }
}

void
SerWaveOp::loadActivation(cereal::JSONInputArchive& archive)
{

    //archive(cereal::make_nvp(WaveOpKey::ActivationFunc, m_ActivationFunc);
    KCC_ARCHIVE(ActivationFunc);
    KCC_ARCHIVE(BiasAddEn);
    KCC_ARCHIVE(BiasSbAddress);
    KCC_ARCHIVE(BiasStartAtMidPart);
    KCC_ARCHIVE(Scale);

    loadDst(archive, Dims::XYZ);
    loadSrc(archive, Dims::XYZ);

    KCC_ARCHIVE(BiasDtype);
    KCC_ARCHIVE(NumPartitions);
    KCC_ARCHIVE(OutDtype);

    KCC_ARCHIVE(TileId);
    KCC_ARCHIVE(TileIdFormat);
}

//===========================================================================
void
SerWaveOp::loadTensorTensor(cereal::JSONInputArchive& archive)
{
    loadSrcAB(archive, Dims::XYZ);
    loadDst(archive, Dims::XYZ);

    KCC_ARCHIVE(NumPartitions);
    KCC_ARCHIVE(TileId);
    KCC_ARCHIVE(TileIdFormat);

    KCC_ARCHIVE(Op);
}

//===========================================================================
void
SerWaveOp::loadTensorScalar(cereal::JSONInputArchive& archive)
{
    loadSrc(archive, Dims::XYZ);
    loadDst(archive, Dims::XYZ);

    KCC_ARCHIVE(NumPartitions);
    KCC_ARCHIVE(TileId);
    KCC_ARCHIVE(TileIdFormat);

    KCC_ARCHIVE(Op0);
    KCC_ARCHIVE(Op1);
    KCC_ARCHIVE(Reverse0);
    KCC_ARCHIVE(Reverse1);
    KCC_ARCHIVE(ImmVal0);
    KCC_ARCHIVE(ImmVal1);
}

void
SerWaveOp::loadTensorScalarPtr(cereal::JSONInputArchive& archive)
{
    loadSrc(archive, Dims::XYZ);
    loadDst(archive, Dims::XYZ);

    KCC_ARCHIVE(NumPartitions);
    KCC_ARCHIVE(TileId);
    KCC_ARCHIVE(TileIdFormat);

    KCC_ARCHIVE(Op0);
    KCC_ARCHIVE(Op1);
    KCC_ARCHIVE(Reverse0);
    KCC_ARCHIVE(Reverse1);
    KCC_ARCHIVE(ImmPtr0);
    KCC_ARCHIVE(ImmPtr1);
}

void
SerWaveOp::loadClipByValue(cereal::JSONInputArchive& archive)
{
    loadSrc(archive, Dims::XYZ);
    loadDst(archive, Dims::XYZ);


    KCC_ARCHIVE(NumPartitions);
    KCC_ARCHIVE(TileId);
    KCC_ARCHIVE(TileIdFormat);

    KCC_ARCHIVE(MinValue);
    KCC_ARCHIVE(MaxValue);
    Assert(m_MinValue <= m_MaxValue, "ClipByValue: MinValue(", m_MinValue, ") > MaxValue(", m_MaxValue, ")");
}

void
SerWaveOp::loadScaleAdd(cereal::JSONInputArchive& archive)
{
    loadSrc(archive, Dims::XYZ);
    loadDst(archive, Dims::XYZ);

    KCC_ARCHIVE(NumPartitions);
    KCC_ARCHIVE(Add);
    KCC_ARCHIVE(Scale);
}

void
SerWaveOp::loadMinimum(cereal::JSONInputArchive& archive)
{
    KCC_ARCHIVE(NumPartitions);
    KCC_ARCHIVE(IsScalarOp);
    if (m_IsScalarOp) {
        KCC_ARCHIVE(ScalarVal);
        loadSrc(archive, Dims::XYZ);
    } else {
        loadSrcAB(archive, Dims::XYZ);
    }
    loadDst(archive, Dims::XYZ);
}

void
SerWaveOp::loadMaximum(cereal::JSONInputArchive& archive)
{
    KCC_ARCHIVE(NumPartitions);
    KCC_ARCHIVE(IsScalarOp);
    if (m_IsScalarOp) {
        KCC_ARCHIVE(ScalarVal);
        loadSrc(archive, Dims::XYZ);
    } else {
        loadSrcAB(archive, Dims::XYZ);
    }
    loadDst(archive, Dims::XYZ);
}


void
SerWaveOp::loadResAdd(cereal::JSONInputArchive& archive)
{
    loadSrcAB(archive, Dims::XYZ);
    loadDst(archive, Dims::XYZ);

    KCC_ARCHIVE(NumPartitions);
}


void
SerWaveOp::loadAdd(cereal::JSONInputArchive& archive)
{
    KCC_ARCHIVE(NumPartitions);
    KCC_ARCHIVE(IsScalarOp);
    if (m_IsScalarOp) {
        KCC_ARCHIVE(ScalarVal);
        loadSrc(archive, Dims::XYZ);
    } else {
        loadSrcAB(archive, Dims::XYZ);
    }
    loadDst(archive, Dims::XYZ);
}

void
SerWaveOp::loadSub(cereal::JSONInputArchive& archive)
{
    KCC_ARCHIVE(NumPartitions);
    KCC_ARCHIVE(IsScalarOp);
    if (m_IsScalarOp) {
        KCC_ARCHIVE(ScalarVal);
        loadSrc(archive, Dims::XYZ);
    } else {
        loadSrcAB(archive, Dims::XYZ);
    }
    loadDst(archive, Dims::XYZ);
}

void
SerWaveOp::loadMultiply(cereal::JSONInputArchive& archive)
{
    KCC_ARCHIVE(NumPartitions);
    KCC_ARCHIVE(IsScalarOp);
    if (m_IsScalarOp) {
        KCC_ARCHIVE(ScalarVal);
        loadSrc(archive, Dims::XYZ);
    } else {
        loadSrcAB(archive, Dims::XYZ);
    }
    loadDst(archive, Dims::XYZ);
}



void
SerWaveOp::loadSrc(cereal::JSONInputArchive& archive, Dims dims)
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
        Assert(false, "Dims to load Src are wrong");
    }
}

void
SerWaveOp::loadSrcA(cereal::JSONInputArchive& archive, Dims dims)
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
        Assert(false, "Dims to load SrcB are wrong");
    }
}

void
SerWaveOp::loadSrcB(cereal::JSONInputArchive& archive, Dims dims)
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
        Assert(false, "Dims to load SrcB are wrong");
    }
}

void
SerWaveOp::loadSrcAB(cereal::JSONInputArchive& archive, Dims dims)
{
    loadSrcA(archive, dims);
    loadSrcB(archive, dims);
}

void
SerWaveOp::loadDst(cereal::JSONInputArchive& archive, Dims dims)
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
    //    // Fall through!
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
        Assert(false, "Dims to load Dst are wrong");
    }
}

#undef KCC_ARCHIVE


} // namespace serialize
} // namespace kcc


