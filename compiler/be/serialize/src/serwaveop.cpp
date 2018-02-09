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
    assert(verify());
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
            archive(cereal::make_nvp(WaveOpKey_OfmapsFoldIdx, m_IfmapsFoldIdx));
        }
    } else if (m_WaveOpType == WaveOpTypeStr_MatMul) {
        archive(cereal::make_nvp(WaveOpKey_IfmapTileHeight, m_IfmapTileHeight));
        archive(cereal::make_nvp(WaveOpKey_IfmapTileWidth, m_IfmapTileWidth));
        archive(cereal::make_nvp(WaveOpKey_IfmapsAtomId, m_IfmapsAtomId));
        archive(cereal::make_nvp(WaveOpKey_IfmapsOffsetInAtom, m_IfmapsOffsetInAtom));
        archive(cereal::make_nvp(WaveOpKey_OfmapTileHeight, m_OfmapTileHeight));
        archive(cereal::make_nvp(WaveOpKey_OfmapTileWidth, m_OfmapTileWidth));
        archive(cereal::make_nvp(WaveOpKey_PsumBankId, m_PsumBankId));
        archive(cereal::make_nvp(WaveOpKey_PsumBankOffset, m_PsumBankOffset));
        archive(cereal::make_nvp(WaveOpKey_Start, m_Start));
        archive(cereal::make_nvp(WaveOpKey_WaveIdFormat, m_WaveIdFormat));
        archive(cereal::make_nvp(WaveOpKey_WeightsAtomId, m_WeightsAtomId));
        archive(cereal::make_nvp(WaveOpKey_WeightsOffsetInAtom, m_WeightsOffsetInAtom));

        assert(m_WaveIdFormat.size() == WaveIdFormatSize);
        std::vector<int> waveId(WaveIdFormatSize, -1); // undefined value before converting
        m_WaveId.convertTo(m_WaveIdFormat.c_str(), waveId);
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
        archive(cereal::make_nvp(WaveOpKey_IfmapTileHeight, m_IfmapTileHeight));
        archive(cereal::make_nvp(WaveOpKey_IfmapTileWidth, m_IfmapTileWidth));
        archive(cereal::make_nvp(WaveOpKey_IfmapsAtomId, m_IfmapsAtomId));
        archive(cereal::make_nvp(WaveOpKey_IfmapsOffsetInAtom, m_IfmapsOffsetInAtom));
        archive(cereal::make_nvp(WaveOpKey_OfmapTileHeight, m_OfmapTileHeight));
        archive(cereal::make_nvp(WaveOpKey_OfmapTileWidth, m_OfmapTileWidth));
        archive(cereal::make_nvp(WaveOpKey_PsumBankId, m_PsumBankId));
        archive(cereal::make_nvp(WaveOpKey_PsumBankOffset, m_PsumBankOffset));
        archive(cereal::make_nvp(WaveOpKey_Start, m_Start));
        archive(cereal::make_nvp(WaveOpKey_WaveIdFormat, m_WaveIdFormat));
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
}




bool
SerWaveOp::verifySbAtomFile () const
{
    /*
    {
    C: "layer_name": "1conv/i1",
    C: "previous_wave-
    void generate(wops": [],
    C: "waveop_name": "1conv/i1/SBAtomFile_0",
    C: "waveop_type": "SBAtomFile"

    "atom_id": 8,
    "batch_fold_idx": 0,
    "length": 2,
    "offset_in_file": 0,
    "ref_file": "trivnet_1conv__weight1__read:0_CRSM.npy",

    "ifmaps_fold_idx": 0,
    "ifmaps_replicate": false,
    },
    */
    if (m_AtomId < 0) {
        return false;
    }
    if (m_BatchFoldIdx < 0) {
        return false;
    }
    if (m_IfmapsFoldIdx  < 0) {
        return false;
    }
    // "ifmaps_replicate": false,
    if (m_Length  < 0) {
        return false;
    }
    if (m_OffsetInFile  < 0) {
        return false;
    }
    if (m_RefFile == "") {
        return false;
    }
    return true;
}

bool
SerWaveOp::verifySbAtomSave () const
{
    /*
    {
    C: "layer_name": "1conv/i1",
    C: "previous_waveops": [
        "1conv/i1/MatMul_n0_m0_h0_w0_c0_r0_s0"
    ],
    C: "waveop_name": "1conv/i1/SBAtomSave_0",
    C: "waveop_type": "SBAtomSave"

    "atom_id": 88,
    "batch_fold_idx": 0,
    "length": 2,
    "offset_in_file": 0,
    "ref_file": "save_1conv__i1.npy",

    "ofmaps_fold_idx": 0,
    }
    */
    if (m_AtomId < 0) {
        return false;
    }
    if (m_BatchFoldIdx < 0) {
        return false;
    }
    if (m_OfmapsFoldIdx  < 0) {
        return false;
    }
    // "ifmaps_replicate": false,
    if (m_Length  < 0) {
        return false;
    }
    if (m_OffsetInFile  < 0) {
        return false;
    }
    if (m_RefFile == "") {
        return false;
    }
    return true;
}

bool
SerWaveOp::verifyMatMul () const
{
    /*
    {
    C: "layer_name": "1conv/i1",
    C: "previous_waveops": [
        "1conv/i1/SBAtomFile_0",
        "input/SBAtomFile_0"
    ],
    C: "waveop_name": "1conv/i1/MatMul_n0_m0_h0_w0_c0_r0_s0",
    C: "waveop_type": "MatMul",

    "ifmap_tile_height": 0,
    "ifmap_tile_width": 0,
    "ifmaps_atom_id": 0,
    "ifmaps_offset_in_atom": 0,
    "ofmap_tile_height": 0,
    "ofmap_tile_width": 0,
    "psum_bank_id": 0,
    "psum_bank_offset": 0,
    "start": true,
    "wave_id": [ 0, 0, 0, 0, 0, 0, 0 ],
    "wave_id_format": "nmhwcrs",
    "weights_atom_id": 8,
    "weights_offset_in_atom": 0
    },
    */

    if (m_IfmapTileHeight < 0) {  // TODO: should be <=
        return false;
    }
    if (m_IfmapTileWidth < 0) { // TODO: should be <=
        return false;
    }
    if (m_IfmapsAtomId < 0) {
        return false;
    }
    if (m_IfmapsOffsetInAtom < 0) {
        return false;
    }
    if (m_OfmapTileHeight < 0) {
        return false;
    }
    if (m_OfmapTileWidth < 0) { // TODO: should be <=
        return false;
    }
    if (m_PsumBankId < 0) {
        return false;
    }
    if (m_PsumBankOffset < 0) {
        return false;
    }
    // "start": true,
    if (! m_WaveId.verify()) {
        return false;
    }
    if (m_WaveIdFormat == "") {
        return false;
    }
    if (m_WeightsAtomId < 0) {
        return false;
    }
    if (m_WeightsOffsetInAtom < 0) {
        return false;
    }
    return true;
}

bool
SerWaveOp::verify() const
{
    // Common
    if (m_WaveOpType == "") {
        return false;
    }
    if (m_WaveOpName == "") {
        return false;
    }
    if (m_LayerName == "") {
        return false;
    }

    if (m_WaveOpType == WaveOpTypeStr_SBAtomFile) {
        return verifySbAtomFile();
    } else if (m_WaveOpType == WaveOpTypeStr_SBAtomSave) {
        return verifySbAtomSave();
    } else if (m_WaveOpType == WaveOpTypeStr_MatMul) {
        return verifyMatMul();
    } else {
        return false;
    }
    return true;
}

} // namespace serialize
} // namespace kcc

