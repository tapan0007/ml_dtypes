#include "serialize/inc/serwaveop.hpp"

#include "wave/inc/matmulwaveop.hpp"
#include "wave/inc/sbatomfilewaveop.hpp"
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

    if (m_WaveOpType == WaveOpTypeStr_SBAtomFile ||
        m_WaveOpType == WaveOpTypeStr_SBAtomSave)
    {
        loadSbAtom(archive);
    } else if (m_WaveOpType == WaveOpTypeStr_Pool) {
        loadPool(archive);
    } else if (m_WaveOpType == WaveOpTypeStr_MatMul) {
        loadMatMul(archive);

    } else if (m_WaveOpType == WaveOpTypeStr_Activation) {
        loadActivation(archive);

    } else {
        assert(false && "Serialization: unsupported WaveOp");
    }
    assert(verify());
} // SerWaveOp::load


void
SerWaveOp::loadSbAtom(cereal::JSONInputArchive& archive)
{
    KCC_ARCHIVE(AtomId);
    KCC_ARCHIVE(AtomSize);
    KCC_ARCHIVE(BatchFoldIdx);
    KCC_ARCHIVE(DataType);
    KCC_ARCHIVE(Length);
    KCC_ARCHIVE(OffsetInFile);
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
SerWaveOp::loadPool(cereal::JSONInputArchive& archive)
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
    KCC_ARCHIVE(SrcPsumBankId);
    KCC_ARCHIVE(SrcPsumBankOffset);
    KCC_ARCHIVE(SrcSbAtomId);
    KCC_ARCHIVE(SrcSbOffsetInAtom);
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
    KCC_ARCHIVE(IfmapsAtomId);
    KCC_ARCHIVE(IfmapsAtomSize);
    KCC_ARCHIVE(IfmapsOffsetInAtom);
    // layer name
    KCC_ARCHIVE(NumColumnPartitions);
    KCC_ARCHIVE(NumRowPartitions);
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
    KCC_ARCHIVE(BiasAtomId);
    KCC_ARCHIVE(BiasOffsetInAtom);
    KCC_ARCHIVE(DstPsumBankId);
    KCC_ARCHIVE(DstXNum);
    KCC_ARCHIVE(DstXStep);
    KCC_ARCHIVE(DstYNum);
    KCC_ARCHIVE(DstYStep);
    KCC_ARCHIVE(DstZNum);
    KCC_ARCHIVE(DstZStep);
    KCC_ARCHIVE(InDtype);
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
#undef KCC_ARCHIVE


} // namespace serialize
} // namespace kcc


