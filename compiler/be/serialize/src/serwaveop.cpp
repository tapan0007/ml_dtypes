#include "serialize/inc/serwaveop.hpp"


namespace kcc {
namespace serialize {

SerWaveOp::SerWaveOp()
{
    m_RefFileShape.resize(4, -1);
    m_TileId.resize(4, -1);
}

// #define RETURN_ASSERT(x) return (x)
#define RETURN_ASSERT(x) assert(x); return (x)

bool
SerWaveOp::verifySbAtom () const
{
    if (m_AtomId < 0) {
        RETURN_ASSERT(false);
    }
    if (m_AtomSize <= 0) {
        RETURN_ASSERT(false);
    }
    if (m_BatchFoldIdx < 0) {
        RETURN_ASSERT(false);
    }
    if (m_DataType == "") {
        RETURN_ASSERT(false);
    }
    if (m_Length  <= 0.0) {
        RETURN_ASSERT(false);
    }
    if (m_OffsetInFile  < 0) {
        RETURN_ASSERT(false);
    }
    if (m_PartitionStepBytes < 1) {
        RETURN_ASSERT(false);
    }
    if (m_RefFile == "") {
        RETURN_ASSERT(false);
    }
    if (m_RefFileFormat != "NCHW" && m_RefFileFormat != "CRSM") {
        RETURN_ASSERT(false);
    }
    if (m_RefFileShape.size() != 4) {
        RETURN_ASSERT(false);
    }
    for (const auto n : m_RefFileShape) {
        if (n <= 0) {
            RETURN_ASSERT(false);
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
        RETURN_ASSERT(false);
    }
    if (m_IfmapCount <= 0) {
        RETURN_ASSERT(false);
    }
    if (m_IfmapsFoldIdx  < 0) {
        RETURN_ASSERT(false);
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
        RETURN_ASSERT(false);
    }
    if (m_OfmapCount  <= 0) {
        RETURN_ASSERT(false);
    }
    if (m_OfmapsFoldIdx  < 0) {
        RETURN_ASSERT(false);
    }
    return true;
}

bool
SerWaveOp::verifyMatMul () const
{
    if (m_IfmapTileHeight < 0) {  // TODO: should be <=
        RETURN_ASSERT(false);
    }
    if (m_IfmapTileWidth < 0) { // TODO: should be <=
        RETURN_ASSERT(false);
    }
    if (m_IfmapsAtomId < 0) {
        RETURN_ASSERT(false);
    }
    if (m_IfmapsOffsetInAtom < 0) {
        RETURN_ASSERT(false);
    }
    if (m_InDtype == "") {
        RETURN_ASSERT(false);
    }
    if (m_OfmapTileHeight < 0) {
        RETURN_ASSERT(false);
    }
    if (m_OfmapTileWidth < 0) { // TODO: should be <=
        RETURN_ASSERT(false);
    }
    if (m_OutDtype == "") {
        RETURN_ASSERT(false);
    }
    if (m_PsumBankId < 0) {
        RETURN_ASSERT(false);
    }
    if (m_PsumBankOffset < 0) {
        RETURN_ASSERT(false);
    }
    if (m_PsumXNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_PsumXStep < 1) {
        RETURN_ASSERT(false);
    }
    if (m_PsumYNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_PsumYStep < 1) {
        RETURN_ASSERT(false);
    }
    // "start": true,
    if (! m_WaveId.verify()) {
        RETURN_ASSERT(false);
    }
    if (m_WaveIdFormat == "") {
        RETURN_ASSERT(false);
    }
    if (m_WeightsAtomId < 0) {
        RETURN_ASSERT(false);
    }
    if (m_WeightsOffsetInAtom < -1) {
        RETURN_ASSERT(false);
    }
    return true;
}


bool
SerWaveOp::verifyPool() const
{
    if (m_DstSbAtomId < 0) {
        RETURN_ASSERT(false);
    }
    if (m_DstSbOffsetInAtom < 0) {
        RETURN_ASSERT(false);
    }
    if (m_DstXNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_DstXStep < 1) {
        RETURN_ASSERT(false);
    }
    if (m_DstYNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_DstYStep < 1) {
        RETURN_ASSERT(false);
    }
    if (m_DstZNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_DstZStep < 1) {
        RETURN_ASSERT(false);
    }
    if (m_InDtype == "") {
        RETURN_ASSERT(false);
    }
    // "layername": "1conv/i1",
    if (m_NumPartitions < 1) {
        RETURN_ASSERT(false);
    }
    if (m_OutDtype == "") {
        RETURN_ASSERT(false);
    }
    if (m_PoolFrequency	< 1) {
        RETURN_ASSERT(false);
    }
    if (m_PoolFunc == "") {
        RETURN_ASSERT(false);
    }

    // previouswaveops": [ 1conv/i1/MatMuln0m0h0w0c0r0s0" ]

    if (m_SrcWNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcWStep < 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcXNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcXStep < 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcYNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcYStep < 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcZNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcZStep < 1) {
        RETURN_ASSERT(false);
    }

    if (m_SrcIsPsum) {
        if (m_SrcPsumBankId < 0) {
            RETURN_ASSERT(false);
        }
        if (m_SrcPsumBankOffset < 0) {
            RETURN_ASSERT(false);
        }
    } else {
        if (m_SrcSbAtomId < 0) {
            RETURN_ASSERT(false);
        }
        if (m_SrcSbOffsetInAtom < 0) {
            RETURN_ASSERT(false);
        }
    }

    if (m_TileId.size() != 4) {
        RETURN_ASSERT(false);
    }
    for (auto n : m_TileId) {
        if (n < 0) {
            RETURN_ASSERT(false);
        }
    }
    if (m_TileIdFormat == "") {
        RETURN_ASSERT(false);
    }
    //waveopname": "1conv/i1/Pooln0m0h0w0",
    //waveoptype": "Pool"

    return true;
}


bool
SerWaveOp::verifyActivation() const
{
    if (m_ActivationFunc == "") {
        RETURN_ASSERT(false);
    }
    // m_BiasAddEn
    if (m_BiasAtomId < 0) {
        RETURN_ASSERT(false);
    }
    if (m_BiasOffsetInAtom < 0) {
        RETURN_ASSERT(false);
    }
    if (m_DstXNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_DstXStep < 1) {
        RETURN_ASSERT(false);
    }
    if (m_DstYNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_DstYStep < 1) {
        RETURN_ASSERT(false);
    }
    if (m_DstZNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_DstZStep < 1) {
        RETURN_ASSERT(false);
    }

    if (m_DstIsPsum) {
        if (m_DstPsumBankId < 0) {
            RETURN_ASSERT(false);
        }
    } else {
        if (m_DstSbAtomId < 0) {
            RETURN_ASSERT(false);
        }
        if (m_DstSbOffsetInAtom < 0) {
            RETURN_ASSERT(false);
        }
    }

    if (m_InDtype == "") {
        RETURN_ASSERT(false);
    }
    if (m_BiasDtype == "") {
        RETURN_ASSERT(false);
    }
    if (m_NumPartitions < 1) {
        RETURN_ASSERT(false);
    }
    if (m_OutDtype == "") {
        RETURN_ASSERT(false);
    }
    if (m_SrcPsumBankId < 0) {
        RETURN_ASSERT(false);
    }
    if (m_SrcXNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcXStep < 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcYNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcYStep < 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcZNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcZStep < 1) {
        RETURN_ASSERT(false);
    }

    if (m_TileId.size() != 4) {
        RETURN_ASSERT(false);
    }
    for (auto n : m_TileId) {
        if (n < 0) {
            RETURN_ASSERT(false);
        }
    }

    if (m_TileIdFormat == "") {
        RETURN_ASSERT(false);
    }
    return true;
}

bool
SerWaveOp::verifyResAdd() const
{
    if (m_InADtype == "") {
        RETURN_ASSERT(false);
    }
    if (m_InBDtype == "") {
        RETURN_ASSERT(false);
    }
    if (m_OutDtype == "") {
        RETURN_ASSERT(false);
    }
    if (m_NumPartitions < 1) {
        RETURN_ASSERT(false);
    }

    if (m_SrcAIsPsum) {
        if (m_SrcAPsumBankId < 0) {
            RETURN_ASSERT(false);
        }
        if (m_SrcAPsumBankOffset < 0) {
            RETURN_ASSERT(false);
        }
    } else {
        if (m_SrcASbAtomId < 0) {
            RETURN_ASSERT(false);
        }
        if (m_SrcASbOffsetInAtom < 0) {
            RETURN_ASSERT(false);
        }
    }
    if (m_SrcAXNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcAXStep < 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcAYNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcAYStep < 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcAZNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcAZStep < 1) {
        RETURN_ASSERT(false);
    }

    if (m_SrcBIsPsum) {
        if (m_SrcBPsumBankId < 0) {
            RETURN_ASSERT(false);
        }
        if (m_SrcBPsumBankOffset < 0) {
            RETURN_ASSERT(false);
        }
    } else {
        if (m_SrcBSbAtomId < 0) {
            RETURN_ASSERT(false);
        }
        if (m_SrcBSbOffsetInAtom < 0) {
            RETURN_ASSERT(false);
        }
    }
    if (m_SrcBXNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcBXStep < 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcBYNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcBYStep < 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcBZNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcBZStep < 1) {
        RETURN_ASSERT(false);
    }

    if (m_DstIsPsum) {
        if (m_DstPsumBankId < 0) {
            RETURN_ASSERT(false);
        }
        if (m_DstPsumBankOffset < 0) {
            RETURN_ASSERT(false);
        }
    } else {
        if (m_DstSbAtomId < 0) {
            RETURN_ASSERT(false);
        }
        if (m_DstSbOffsetInAtom < 0) {
            RETURN_ASSERT(false);
        }
    }
    if (m_DstXNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_DstXStep < 1) {
        RETURN_ASSERT(false);
    }
    if (m_DstYNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_DstYStep < 1) {
        RETURN_ASSERT(false);
    }
    if (m_DstZNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_DstZStep < 1) {
        RETURN_ASSERT(false);
    }

    return true;
}


bool
SerWaveOp::verify() const
{
    // Common
    if (m_WaveOpType == "") {
        RETURN_ASSERT(false);
    }
    if (m_WaveOpName == "") {
        RETURN_ASSERT(false);
    }
    if (m_LayerName == "") {
        RETURN_ASSERT(false);
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
    } else if (m_WaveOpType == WaveOpTypeStr_ResAdd) {
        return verifyResAdd();
    } else {
        RETURN_ASSERT(false);
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
    case ActivationFunc_LeakyRelu:
        return WaveOpKey_ActivationFunc_LeakyRelu;
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
    } else if (actType  == WaveOpKey_ActivationFunc_LeakyRelu) {
        return ActivationFunc_LeakyRelu;
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

