#include "serialize/inc/serwaveop.hpp"


namespace kcc {
namespace serialize {

SerWaveOp::SerWaveOp()
{
}


template<>
void
SerWaveOp::save<cereal::JSONOutputArchive>(cereal::JSONOutputArchive& archive) const
{
    archive(cereal::make_nvp(WaveOpKey_WaveOpType, m_WaveOpType));
    archive(cereal::make_nvp(WaveOpKey_WaveOpName, m_WaveOpName));
    archive(cereal::make_nvp(WaveOpKey_LayerName, m_LayerName));
    archive(cereal::make_nvp(WaveOpKey_PreviousWaveOps, m_PreviousWaveOps));

    if (m_WaveOpType == WaveOpTypeStr_SBAtomFile) {
        archive(cereal::make_nvp(WaveOpKey_AtomId, m_AtomId));
        archive(cereal::make_nvp(WaveOpKey_IfmapsFoldIdx, m_IfmapsFoldIdx));
        archive(cereal::make_nvp(WaveOpKey_IfmapsReplicate, m_IfmapsReplicate));
        archive(cereal::make_nvp(WaveOpKey_Length, m_Length));
        archive(cereal::make_nvp(WaveOpKey_OffsetInFile, m_OffsetInFile));
        archive(cereal::make_nvp(WaveOpKey_RefFile, m_RefFile));
    } else if (m_WaveOpType == WaveOpTypeStr_MatMul) {
        archive(cereal::make_nvp(WaveOpKey_IfmapsAtomId, m_IfmapsAtomId));
        archive(cereal::make_nvp(WaveOpKey_IfmapsOffsetInAtom, m_IfmapsOffsetInAtom));
        archive(cereal::make_nvp(WaveOpKey_PsumBankId, m_PsumBankId));
        archive(cereal::make_nvp(WaveOpKey_Start, m_Start));
        archive(cereal::make_nvp(WaveOpKey_WaveIdFormat, m_WaveIdFormat));
        archive(cereal::make_nvp(WaveOpKey_WeightsAtomId, m_WeightsAtomId));
        archive(cereal::make_nvp(WaveOpKey_WeightsOffsetInAtom, m_WeightsOffsetInAtom));

        std::vector<int> waveId(WaveIdFormatSize);
        for (int i = 0; i < WaveIdFormatSize; ++i) {
            // "wave_id_format": "nmhwcrs",
            switch (m_WaveIdFormat[i]) {
            case 'n':
                waveId[i] = m_WaveId.gBatchIdx();
                break;
            case 'm':
                waveId[i] = m_WaveId.gOfmapFoldIdx();
                break;
            case 'h':
                waveId[i] = m_WaveId.gTileY();
                break;
            case 'w':
                waveId[i] = m_WaveId.gTileX();
                break;
            case 'c':
                waveId[i] = m_WaveId.gIfmapFoldIdx();
                break;
            case 'r':
                waveId[i] = m_WaveId.gFilterPixelX();
                break;
            case 's':
                waveId[i] = m_WaveId.gFilterPixelY();
                break;
            default:
                assert(false && "Wrong MatMulWaveOp format character");
            }
        }
        archive(cereal::make_nvp(WaveOpKey_WaveId, waveId));

    } else {
        assert(false && "Serialization: unsupported WaveOp");
    }
}

template<>
void
SerWaveOp::load<cereal::JSONInputArchive>(cereal::JSONInputArchive& archive)
{
    archive(cereal::make_nvp(WaveOpKey_WaveOpType, m_WaveOpType));
    archive(cereal::make_nvp(WaveOpKey_WaveOpName, m_WaveOpName));
    archive(cereal::make_nvp(WaveOpKey_LayerName, m_LayerName));
    archive(cereal::make_nvp(WaveOpKey_PreviousWaveOps, m_PreviousWaveOps));

    if (m_WaveOpType == WaveOpTypeStr_SBAtomFile) {
        archive(cereal::make_nvp(WaveOpKey_AtomId, m_AtomId));
        archive(cereal::make_nvp(WaveOpKey_IfmapsFoldIdx, m_IfmapsFoldIdx));
        archive(cereal::make_nvp(WaveOpKey_IfmapsReplicate, m_IfmapsReplicate));
        archive(cereal::make_nvp(WaveOpKey_Length, m_Length));
        archive(cereal::make_nvp(WaveOpKey_OffsetInFile, m_OffsetInFile));
        archive(cereal::make_nvp(WaveOpKey_RefFile, m_RefFile));
    } else if (m_WaveOpType == WaveOpTypeStr_MatMul) {
        archive(cereal::make_nvp(WaveOpKey_IfmapsAtomId, m_IfmapsAtomId));
        archive(cereal::make_nvp(WaveOpKey_IfmapsOffsetInAtom, m_IfmapsOffsetInAtom));
        archive(cereal::make_nvp(WaveOpKey_PsumBankId, m_PsumBankId));
        archive(cereal::make_nvp(WaveOpKey_Start, m_Start));
        archive(cereal::make_nvp(WaveOpKey_WaveIdFormat, m_WaveIdFormat));
        archive(cereal::make_nvp(WaveOpKey_WeightsAtomId, m_WeightsAtomId));
        archive(cereal::make_nvp(WaveOpKey_WeightsOffsetInAtom, m_WeightsOffsetInAtom));

        std::vector<int> waveId(WaveIdFormatSize);
        archive(cereal::make_nvp(WaveOpKey_WaveId, waveId));
        assert(WaveIdFormatSize == waveId.size() && "Number of element in WaveId wrong");
        for (int i = 0; i < WaveIdFormatSize; ++i) {
            // "wave_id_format": "nmhwcrs",
            switch (m_WaveIdFormat[i]) {
            case 'n':
                m_WaveId.rBatchIdx(waveId[i]);
                break;
            case 'm':
                m_WaveId.rOfmapFoldIdx(waveId[i]);
                break;
            case 'h':
                m_WaveId.rTileY(waveId[i]);
                break;
            case 'w':
                m_WaveId.rTileX(waveId[i]);
                break;
            case 'c':
                m_WaveId.rIfmapFoldIdx(waveId[i]);
                break;
            case 'r':
                m_WaveId.rFilterPixelX(waveId[i]);
                break;
            case 's':
                m_WaveId.rFilterPixelY(waveId[i]);
                break;
            default:
                assert(false && "Wrong MatMulWaveOp format character");
            }
        }

    } else {
        assert(false && "Serialization: unsupported WaveOp");
    }

}

} // namespace serialize
} // namespace kcc

