#include "serialize/inc/serwaveop.hpp"

#include "wave/inc/matmulwaveop.hpp"
#include "wave/inc/sbatomfilewaveop.hpp"
#include "wave/inc/sbatomsavewaveop.hpp"
#include "wave/inc/poolwaveop.hpp"

namespace kcc {
namespace serialize {


template<>
void
SerWaveOp::load<cereal::JSONInputArchive>(cereal::JSONInputArchive& archive)
{
    archive(cereal::make_nvp(WaveOpKey_WaveOpType, m_WaveOpType));
    archive(cereal::make_nvp(WaveOpKey_WaveOpName, m_WaveOpName));
    archive(cereal::make_nvp(WaveOpKey_LayerName, m_LayerName));
    archive(cereal::make_nvp(WaveOpKey_PreviousWaveOps, m_PreviousWaveOps));

    if (m_WaveOpType == WaveOpTypeStr_SBAtomFile ||
        m_WaveOpType == WaveOpTypeStr_SBAtomSave)
    {
        archive(cereal::make_nvp(WaveOpKey_AtomId, m_AtomId));
        archive(cereal::make_nvp(WaveOpKey_AtomSize, m_AtomSize));
        archive(cereal::make_nvp(WaveOpKey_BatchFoldIdx, m_BatchFoldIdx));
        archive(cereal::make_nvp(WaveOpKey_DataType, m_DataType));
        archive(cereal::make_nvp(WaveOpKey_Length, m_Length));
        archive(cereal::make_nvp(WaveOpKey_OffsetInFile, m_OffsetInFile));
        archive(cereal::make_nvp(WaveOpKey_RefFile, m_RefFile));
        archive(cereal::make_nvp(WaveOpKey_RefFileFormat, m_RefFileFormat));
        archive(cereal::make_nvp(WaveOpKey_RefFileShape, m_RefFileShape));

        if (m_WaveOpType == WaveOpTypeStr_SBAtomFile) {
            archive(cereal::make_nvp(WaveOpKey_IfmapCount, m_IfmapCount));
            archive(cereal::make_nvp(WaveOpKey_IfmapsFoldIdx, m_IfmapsFoldIdx));
            archive(cereal::make_nvp(WaveOpKey_IfmapsReplicate, m_IfmapsReplicate));
        } else {
            archive(cereal::make_nvp(WaveOpKey_OfmapCount, m_OfmapCount));
            archive(cereal::make_nvp(WaveOpKey_OfmapsFoldIdx, m_OfmapsFoldIdx));
        }
    } else if (m_WaveOpType == WaveOpTypeStr_Pool) {
        archive(cereal::make_nvp(WaveOpKey_DstSbAtomId, m_DstSbAtomId));
        archive(cereal::make_nvp(WaveOpKey_DstSbOffsetInAtom, m_DstSbOffsetInAtom));
        archive(cereal::make_nvp(WaveOpKey_DstXNum, m_DstXNum));
        archive(cereal::make_nvp(WaveOpKey_DstXStep, m_DstXStep));
        archive(cereal::make_nvp(WaveOpKey_DstYNum, m_DstYNum));
        archive(cereal::make_nvp(WaveOpKey_DstYStep, m_DstYStep));
        archive(cereal::make_nvp(WaveOpKey_DstZNum, m_DstZNum));
        archive(cereal::make_nvp(WaveOpKey_DstZStep, m_DstZStep));
        archive(cereal::make_nvp(WaveOpKey_InDtype, m_InDtype));
        // layername
        archive(cereal::make_nvp(WaveOpKey_NumPartitions, m_NumPartitions));
        archive(cereal::make_nvp(WaveOpKey_OutDtype, m_OutDtype));
        archive(cereal::make_nvp(WaveOpKey_PoolFrequency, m_PoolFrequency));
        archive(cereal::make_nvp(WaveOpKey_PoolFunc, m_PoolFunc));
        // previouswaveops
        archive(cereal::make_nvp(WaveOpKey_SrcIsPsum, m_SrcIsPsum));
        archive(cereal::make_nvp(WaveOpKey_SrcPsumBankId, m_SrcPsumBankId));
        archive(cereal::make_nvp(WaveOpKey_SrcPsumBankOffset, m_SrcPsumBankOffset));
        archive(cereal::make_nvp(WaveOpKey_SrcSbAtomId, m_SrcSbAtomId));
        archive(cereal::make_nvp(WaveOpKey_SrcSbOffsetInAtom, m_SrcSbOffsetInAtom));
        archive(cereal::make_nvp(WaveOpKey_SrcWNum, m_SrcWNum));
        archive(cereal::make_nvp(WaveOpKey_SrcWStep, m_SrcWStep));
        archive(cereal::make_nvp(WaveOpKey_SrcXNum, m_SrcXNum));
        archive(cereal::make_nvp(WaveOpKey_SrcXStep, m_SrcXStep));
        archive(cereal::make_nvp(WaveOpKey_SrcYNum, m_SrcYNum));
        archive(cereal::make_nvp(WaveOpKey_SrcYStep, m_SrcYStep));
        archive(cereal::make_nvp(WaveOpKey_SrcZNum, m_SrcZNum));
        archive(cereal::make_nvp(WaveOpKey_SrcZStep, m_SrcZStep));
        archive(cereal::make_nvp(WaveOpKey_TileId, m_TileId));
        archive(cereal::make_nvp(WaveOpKey_TileIdFormat, m_TileIdFormat));

    } else if (m_WaveOpType == WaveOpTypeStr_MatMul) {
        archive(cereal::make_nvp(WaveOpKey_BatchingInWave, m_BatchingInWave));
        archive(cereal::make_nvp(WaveOpKey_FmapXNum, m_FmapXNum));
        archive(cereal::make_nvp(WaveOpKey_FmapXStep, m_FmapXStep));
        archive(cereal::make_nvp(WaveOpKey_FmapYNum, m_FmapYNum));
        archive(cereal::make_nvp(WaveOpKey_FmapYStep, m_FmapYStep));
        archive(cereal::make_nvp(WaveOpKey_FmapZNum, m_FmapZNum));
        archive(cereal::make_nvp(WaveOpKey_FmapZStepAtoms, m_FmapZStepAtoms));
        archive(cereal::make_nvp(WaveOpKey_IfmapCount, m_IfmapCount));
        archive(cereal::make_nvp(WaveOpKey_IfmapTileHeight, m_IfmapTileHeight));
        archive(cereal::make_nvp(WaveOpKey_IfmapTileWidth, m_IfmapTileWidth));
        archive(cereal::make_nvp(WaveOpKey_IfmapsAtomId, m_IfmapsAtomId));
        archive(cereal::make_nvp(WaveOpKey_IfmapsAtomSize, m_IfmapsAtomSize));
        archive(cereal::make_nvp(WaveOpKey_IfmapsOffsetInAtom, m_IfmapsOffsetInAtom));
        // layer name
        archive(cereal::make_nvp(WaveOpKey_NumColumnPartitions, m_NumColumnPartitions));
        archive(cereal::make_nvp(WaveOpKey_NumRowPartitions, m_NumRowPartitions));
        archive(cereal::make_nvp(WaveOpKey_OfmapCount, m_OfmapCount));
        archive(cereal::make_nvp(WaveOpKey_OfmapTileHeight, m_OfmapTileHeight));
        archive(cereal::make_nvp(WaveOpKey_OfmapTileWidth, m_OfmapTileWidth));
        // previous waveops
        archive(cereal::make_nvp(WaveOpKey_PsumBankId, m_PsumBankId));
        archive(cereal::make_nvp(WaveOpKey_PsumBankOffset, m_PsumBankOffset));
        archive(cereal::make_nvp(WaveOpKey_PsumXNum, m_PsumXNum));
        archive(cereal::make_nvp(WaveOpKey_PsumXStep, m_PsumXStep));
        archive(cereal::make_nvp(WaveOpKey_PsumYNum, m_PsumYNum));
        archive(cereal::make_nvp(WaveOpKey_PsumYStep, m_PsumYStep));
        archive(cereal::make_nvp(WaveOpKey_StartTensorCalc, m_StartTensorCalc));
        archive(cereal::make_nvp(WaveOpKey_StopTensorCalc, m_StopTensorCalc));
        archive(cereal::make_nvp(WaveOpKey_StrideX, m_StrideX));
        archive(cereal::make_nvp(WaveOpKey_StrideY, m_StrideY));
        archive(cereal::make_nvp(WaveOpKey_WaveIdFormat, m_WaveIdFormat));
        // waveop name
        // waveop type
        archive(cereal::make_nvp(WaveOpKey_WeightsAtomId, m_WeightsAtomId));
        archive(cereal::make_nvp(WaveOpKey_WeightsOffsetInAtom, m_WeightsOffsetInAtom));

        assert(m_WaveIdFormat.size() == WaveIdFormatSize);
        std::vector<int> waveId(WaveIdFormatSize, -1); // undefined value before deserial
        archive(cereal::make_nvp(WaveOpKey_WaveId, waveId));
        assert(WaveIdFormatSize == waveId.size() && "Number of element in WaveId wrong");
        m_WaveId.convertFrom(m_WaveIdFormat.c_str(), waveId);

    } else {
        assert(false && "Serialization: unsupported WaveOp");
    }
    assert(verify());
} // SerWaveOp::load




} // namespace serialize
} // namespace kcc


