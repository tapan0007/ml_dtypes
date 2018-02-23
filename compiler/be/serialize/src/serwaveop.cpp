#include "serialize/inc/serwaveop.hpp"


namespace kcc {
namespace serialize {

SerWaveOp::SerWaveOp()
{
    m_RefFileShape.resize(4, -1);
    m_TileId.resize(4, -1);
}

// #define DEBUG_ASSERT(x) assert(true)
#define DEBUG_ASSERT(x) assert(x)

bool
SerWaveOp::verifySbAtom () const
{
    if (m_AtomId < 0) {
        DEBUG_ASSERT(false);
        return false;
    }
    if (m_AtomSize <= 0) {
        DEBUG_ASSERT(false);
        return false;
    }
    if (m_BatchFoldIdx < 0) {
        DEBUG_ASSERT(false);
        return false;
    }
    if (m_DataType == "") {
        DEBUG_ASSERT(false);
        return false;
    }
    if (m_Length  <= 0.0) {
        DEBUG_ASSERT(false);
        return false;
    }
    if (m_OffsetInFile  < 0) {
        DEBUG_ASSERT(false);
        return false;
    }
    if (m_RefFile == "") {
        DEBUG_ASSERT(false);
        return false;
    }
    if (m_RefFileFormat != "NCHW" && m_RefFileFormat != "CRSM") {
        DEBUG_ASSERT(false);
        return false;
    }
    if (m_RefFileShape.size() != 4) {
        DEBUG_ASSERT(false);
        return false;
    }
    for (const auto n : m_RefFileShape) {
        if (n <= 0) {
            DEBUG_ASSERT(false);
            return false;
        }
    }
    return true;
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
    if (! this->verifySbAtom()) {
        DEBUG_ASSERT(false);
        return false;
    }
    if (m_IfmapCount <= 0) {
        DEBUG_ASSERT(false);
        return false;
    }
    if (m_IfmapsFoldIdx  < 0) {
        DEBUG_ASSERT(false);
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
    if (! this->verifySbAtom()) {
        DEBUG_ASSERT(false);
        return false;
    }
    if (m_OfmapCount  <= 0) {
        DEBUG_ASSERT(false);
        return false;
    }
    if (m_OfmapsFoldIdx  < 0) {
        DEBUG_ASSERT(false);
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
        DEBUG_ASSERT(false);
        return false;
    }
    if (m_IfmapTileWidth < 0) { // TODO: should be <=
        DEBUG_ASSERT(false);
        return false;
    }
    if (m_IfmapsAtomId < 0) {
        DEBUG_ASSERT(false);
        return false;
    }
    if (m_IfmapsOffsetInAtom < 0) {
        DEBUG_ASSERT(false);
        return false;
    }
    if (m_OfmapTileHeight < 0) {
        DEBUG_ASSERT(false);
        return false;
    }
    if (m_OfmapTileWidth < 0) { // TODO: should be <=
        DEBUG_ASSERT(false);
        return false;
    }
    if (m_PsumBankId < 0) {
        DEBUG_ASSERT(false);
        return false;
    }
    if (m_PsumBankOffset < 0) {
        DEBUG_ASSERT(false);
        return false;
    }
    if (m_PsumXNum < 1) {
        DEBUG_ASSERT(false);
        return false;
    }
    if (m_PsumXStep < 1) {
        DEBUG_ASSERT(false);
        return false;
    }
    if (m_PsumYNum < 1) {
        DEBUG_ASSERT(false);
        return false;
    }
    if (m_PsumYStep < 1) {
        DEBUG_ASSERT(false);
        return false;
    }
    // "start": true,
    if (! m_WaveId.verify()) {
        DEBUG_ASSERT(false);
        return false;
    }
    if (m_WaveIdFormat == "") {
        DEBUG_ASSERT(false);
        return false;
    }
    if (m_WeightsAtomId < 0) {
        DEBUG_ASSERT(false);
        return false;
    }
    if (m_WeightsOffsetInAtom < 0) {
        DEBUG_ASSERT(false);
        return false;
    }
    return true;
}


bool
SerWaveOp::verifyPool() const
{
    if (m_DstSbAtomId < 0) {
        DEBUG_ASSERT(false);
        return false;
    }
    if (m_DstSbOffsetInAtom < 0) {
        DEBUG_ASSERT(false);
        return false;
    }
    if (m_DstXNum < 1) {
        DEBUG_ASSERT(false);
        return false;
    }
    if (m_DstXStep < 1) {
        DEBUG_ASSERT(false);
        return false;
    }
    if (m_DstYNum < 1) {
        DEBUG_ASSERT(false);
        return false;
    }
    if (m_DstYStep < 1) {
        DEBUG_ASSERT(false);
        return false;
    }
    if (m_DstZNum < 1) {
        DEBUG_ASSERT(false);
        return false;
    }
    if (m_DstZStep < 1) {
        DEBUG_ASSERT(false);
        return false;
    }
    if (m_InDtype == "") {
        DEBUG_ASSERT(false);
        return false;
    }
    // "layername": "1conv/i1",
    if (m_NumPartitions < 1) {
        DEBUG_ASSERT(false);
        return false;
    }
    if (m_OutDtype == "") {
        DEBUG_ASSERT(false);
        return false;
    }
    if (m_PoolFrequency	< 1) {
        DEBUG_ASSERT(false);
        return false;
    }
    if (m_PoolFunc == "") {
        DEBUG_ASSERT(false);
        return false;
    }
    // previouswaveops": [ 1conv/i1/MatMuln0m0h0w0c0r0s0" ]
    // m_SrcIsPsum
    if (m_SrcPsumBankId < 0) {
        DEBUG_ASSERT(false);
        return false;
    }
    if (m_SrcPsumBankOffset < 0) {
        DEBUG_ASSERT(false);
        return false;
    }
    if (m_SrcSbAtomId < 0) {
        DEBUG_ASSERT(false);
        return false;
    }
    if (m_SrcSbOffsetInAtom < 0) {
        DEBUG_ASSERT(false);
        return false;
    }
    if (m_SrcWNum < 1) {
        DEBUG_ASSERT(false);
        return false;
    }
    if (m_SrcWStep < 1) {
        DEBUG_ASSERT(false);
        return false;
    }
    if (m_SrcXNum < 1) {
        DEBUG_ASSERT(false);
        return false;
    }
    if (m_SrcXStep < 1) {
        DEBUG_ASSERT(false);
        return false;
    }
    if (m_SrcYNum < 1) {
        DEBUG_ASSERT(false);
        return false;
    }
    if (m_SrcYStep < 1) {
        DEBUG_ASSERT(false);
        return false;
    }
    if (m_SrcZNum < 1) {
        DEBUG_ASSERT(false);
        return false;
    }
    if (m_SrcZStep < 1) {
        DEBUG_ASSERT(false);
        return false;
    }
    if (m_TileId.size() != 4) {
        DEBUG_ASSERT(false);
        return false;
    }
    for (auto n : m_TileId) {
        if (n < 0) {
            DEBUG_ASSERT(false);
            return false;
        }
    }
    if (m_TileIdFormat == "") {
        DEBUG_ASSERT(false);
        return false;
    }
    //waveopname": "1conv/i1/Pooln0m0h0w0",
    //waveoptype": "Pool"

    return true;
}


bool
SerWaveOp::verifyActivation() const
{
    if (m_ActivationFunc == "") {
        DEBUG_ASSERT(false);
        return false;
    }
    // m_BiasAddEn
    if (m_BiasAtomId < 0) {
        return false;
    }
    if (m_BiasOffsetInAtom < 0) {
        return false;
    }
    if (m_DstPsumBankId < 0) {
        return false;
    }
    if (m_DstXNum < 1) {
        return false;
    }
    if (m_DstXStep < 1) {
        return false;
    }
    if (m_DstYNum < 1) {
        return false;
    }
    if (m_DstYStep < 1) {
        return false;
    }
    if (m_DstZNum < 1) {
        return false;
    }
    if (m_DstZStep < 1) {
        return false;
    }

    if (m_InDtype == "") {
        return false;
    }
    if (m_NumPartitions < 1) {
        return false;
    }
    if (m_OutDtype == "") {
        return false;
    }
    if (m_SrcPsumBankId < 0) {
        return false;
    }
    if (m_SrcXNum < 1) {
        return false;
    }
    if (m_SrcXStep < 1) {
        return false;
    }
    if (m_SrcYNum < 1) {
        return false;
    }
    if (m_SrcYStep < 1) {
        return false;
    }
    if (m_SrcZNum < 1) {
        return false;
    }
    if (m_SrcZStep < 1) {
        return false;
    }

    if (m_TileId.size() != 4) {
        DEBUG_ASSERT(false);
        return false;
    }
    for (auto n : m_TileId) {
        if (n < 0) {
            DEBUG_ASSERT(false);
            return false;
        }
    }

    if (m_TileIdFormat == "") {
        DEBUG_ASSERT(false);
        return false;
    }
    return true;
}


bool
SerWaveOp::verify() const
{
    // Common
    if (m_WaveOpType == "") {
        DEBUG_ASSERT(false);
        return false;
    }
    if (m_WaveOpName == "") {
        DEBUG_ASSERT(false);
        return false;
    }
    if (m_LayerName == "") {
        DEBUG_ASSERT(false);
        return false;
    }

    if (m_WaveOpType == WaveOpTypeStr_SBAtomFile) {
        return verifySbAtomFile();
    } else if (m_WaveOpType == WaveOpTypeStr_SBAtomSave) {
        return verifySbAtomSave();
    } else if (m_WaveOpType == WaveOpTypeStr_MatMul) {
        return verifyMatMul();
    } else if (m_WaveOpType == WaveOpTypeStr_Pool) {
        return verifyPool();
    } else if (m_WaveOpType == WaveOpTypeStr_Activation) {
        return verifyActivation();
    } else {
        DEBUG_ASSERT(false);
        return false;
    }
    return true;
}

std::string
SerWaveOp::activationType2Str(ActivationFunc actType)
{
    switch (actType) {
    case ActivationFunc_Identity:
        return WaveOpKey_ActivationFunc_Identity;
        break;
    case ActivationFunc_Relu:
        return WaveOpKey_ActivationFunc_Relu;
        break;
    case ActivationFunc_LRelu:
        return WaveOpKey_ActivationFunc_Lrelu;
        break;
    case ActivationFunc_PRelu:
        return WaveOpKey_ActivationFunc_Prelu;
        break;
    case ActivationFunc_Sigmoid:
        return WaveOpKey_ActivationFunc_Sigmoid;
        break;
    case ActivationFunc_Tanh:
        return WaveOpKey_ActivationFunc_Tanh;
        break;
    case ActivationFunc_Exp:
        return WaveOpKey_ActivationFunc_Exp;
        break;
    default:
        assert(false && "Wrong activation type");
        break;
    }
    return "";
}

ActivationFunc
SerWaveOp::str2ActivationFunc(const std::string& actType)
{
    if (actType == WaveOpKey_ActivationFunc_Identity
        || actType == WaveOpKey_ActivationFunc_None /* until Jeff fixes none */) {
        return ActivationFunc_Identity;
    } else if (actType  == WaveOpKey_ActivationFunc_Relu) {
        return ActivationFunc_Relu;
    } else if (actType  == WaveOpKey_ActivationFunc_Lrelu) {
        return ActivationFunc_LRelu;
    } else if (actType  == WaveOpKey_ActivationFunc_Prelu) {
        return ActivationFunc_PRelu;
    } else if (actType  == WaveOpKey_ActivationFunc_Sigmoid) {
        return ActivationFunc_Sigmoid;
    } else if (actType  == WaveOpKey_ActivationFunc_Tanh) {
        return ActivationFunc_Tanh;
    } else if (actType  == WaveOpKey_ActivationFunc_Exp) {
        return ActivationFunc_Exp;
    } else {
        assert(false && "Wrong activation type");
    }
    return ActivationFunc_Exp;
}


} // namespace serialize
} // namespace kcc

