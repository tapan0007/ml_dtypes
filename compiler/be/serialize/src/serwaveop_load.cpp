#include "serialize/inc/serwaveop.hpp"


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
        archive(cereal::make_nvp(WaveOpKey_BatchFoldIdx, m_BatchFoldIdx));
        archive(cereal::make_nvp(WaveOpKey_Length, m_Length));
        archive(cereal::make_nvp(WaveOpKey_OffsetInFile, m_OffsetInFile));
        archive(cereal::make_nvp(WaveOpKey_RefFile, m_RefFile));

        if (m_WaveOpType == WaveOpTypeStr_SBAtomFile) {
            archive(cereal::make_nvp(WaveOpKey_IfmapsFoldIdx, m_IfmapsFoldIdx));
            archive(cereal::make_nvp(WaveOpKey_IfmapsReplicate, m_IfmapsReplicate));
        } else {
            archive(cereal::make_nvp(WaveOpKey_OfmapsFoldIdx, m_OfmapsFoldIdx));
        }
    } else if (m_WaveOpType == WaveOpTypeStr_MatMul) {
        archive(cereal::make_nvp(WaveOpKey_BatchingInWave, m_BatchingInWave));
        archive(cereal::make_nvp(WaveOpKey_IfmapCount, m_IfmapCount));
        archive(cereal::make_nvp(WaveOpKey_IfmapTileHeight, m_IfmapTileHeight));
        archive(cereal::make_nvp(WaveOpKey_IfmapTileWidth, m_IfmapTileWidth));
        archive(cereal::make_nvp(WaveOpKey_IfmapsAtomId, m_IfmapsAtomId));
        archive(cereal::make_nvp(WaveOpKey_IfmapsOffsetInAtom, m_IfmapsOffsetInAtom));
        // layer name
        archive(cereal::make_nvp(WaveOpKey_OfmapCount, m_OfmapCount));
        archive(cereal::make_nvp(WaveOpKey_OfmapTileHeight, m_OfmapTileHeight));
        archive(cereal::make_nvp(WaveOpKey_OfmapTileWidth, m_OfmapTileWidth));
        // previous waveops
        archive(cereal::make_nvp(WaveOpKey_PsumBankId, m_PsumBankId));
        archive(cereal::make_nvp(WaveOpKey_PsumBankOffset, m_PsumBankOffset));
        archive(cereal::make_nvp(WaveOpKey_Start, m_Start));
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


