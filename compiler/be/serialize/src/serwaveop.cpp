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
    if (m_SbAddress < 0) {
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
    if (m_RefFileFormat != "NCHW" && m_RefFileFormat != "CRSM" && m_RefFileFormat != "HNWC") {
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
SerWaveOp::verifySbAtomLoad () const
{
    if (! this->verifySbAtom()) {
        RETURN_ASSERT(false);
    }
    if (m_NumPartitions<= 0) {
        RETURN_ASSERT(false);
    }
    if (m_IfmapReplicationNumRows < 0) {
        RETURN_ASSERT(false);
    }
    if (m_IfmapReplicationResolution < 0) {
        RETURN_ASSERT(false);
    }
    if (m_IfmapReplicationStepBytes < 0) {
        RETURN_ASSERT(false);
    }
    if (m_SrcStepElem < 0) {
        RETURN_ASSERT(false);
    }
    return true;
}

bool
SerWaveOp::verifySbAtomSave () const
{
    if (! this->verifySbAtom()) {
        RETURN_ASSERT(false);
    }
    if (m_NumPartitions  <= 0) {
        RETURN_ASSERT(false);
    }
    return true;
}

bool
SerWaveOp::verifyMatMul () const
{
    if (m_IfmapsSbAddress < 0) {
        RETURN_ASSERT(false);
    }
    if (m_InDtype == "") {
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
    if (m_PsumZNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_PsumZStep < 1) {
        RETURN_ASSERT(false);
    }
    if (m_WeightsSbAddress < -1) {
        RETURN_ASSERT(false);
    }
    if (m_IfmapReplicationNumRows < 0) {
        RETURN_ASSERT(false);
    }
    if (m_IfmapReplicationResolution < 0) {
        RETURN_ASSERT(false);
    }
    if (m_IfmapReplicationShiftAmnt < 0) {
        RETURN_ASSERT(false);
    }
    return true;
}


bool
SerWaveOp::verifyPool() const
{
    if (m_DstSbAddress < 0) {
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
    if (m_PoolFrequency < 1) {
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
        if (m_SrcSbAddress < 0) {
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
    if (m_BiasSbAddress < 0) {
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
        if (m_DstSbAddress < 0) {
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
    if (m_SrcIsPsum) {
        if (m_SrcPsumBankId < 0) {
            RETURN_ASSERT(false);
        }
    } else {
        if (m_SrcSbAddress < 0) {
            RETURN_ASSERT(false);
        }
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
        if (m_SrcASbAddress < 0) {
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
        if (m_SrcBSbAddress < 0) {
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
        if (m_DstSbAddress < 0) {
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
SerWaveOp::verifyBarrier() const
{
    return true;
}

bool
SerWaveOp::verifyNop() const
{
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

    if (m_WaveOpType == WaveOpTypeStr_SBAtomLoad) {
        return verifySbAtomLoad();
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
    } else if (m_WaveOpType == WaveOpTypeStr_Barrier) {
        return verifyBarrier();
    } else if (m_WaveOpType == WaveOpTypeStr_Nop) {
        return verifyNop();
    } else {
        RETURN_ASSERT(false);
    }
    return true;
}

std::string
SerWaveOp::activationType2Str(ActivationFunc actType)
{
    switch (actType) {
    case ActivationFunc::Identity:
        return WaveOpKey_ActivationFunc_Identity;
        break;
    case ActivationFunc::Relu:
        return WaveOpKey_ActivationFunc_Relu;
        break;
    case ActivationFunc::LeakyRelu:
        return WaveOpKey_ActivationFunc_LeakyRelu;
        break;
    case ActivationFunc::PRelu:
        return WaveOpKey_ActivationFunc_Prelu;
        break;
    case ActivationFunc::Sigmoid:
        return WaveOpKey_ActivationFunc_Sigmoid;
        break;
    case ActivationFunc::Tanh:
        return WaveOpKey_ActivationFunc_Tanh;
        break;
    case ActivationFunc::Exp:
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
        return ActivationFunc::Identity;
    } else if (actType  == WaveOpKey_ActivationFunc_Relu) {
        return ActivationFunc::Relu;
    } else if (actType  == WaveOpKey_ActivationFunc_LeakyRelu) {
        return ActivationFunc::LeakyRelu;
    } else if (actType  == WaveOpKey_ActivationFunc_Prelu) {
        return ActivationFunc::PRelu;
    } else if (actType  == WaveOpKey_ActivationFunc_Sigmoid) {
        return ActivationFunc::Sigmoid;
    } else if (actType  == WaveOpKey_ActivationFunc_Tanh) {
        return ActivationFunc::Tanh;
    } else if (actType  == WaveOpKey_ActivationFunc_Exp) {
        return ActivationFunc::Exp;
    } else {
        assert(false && "Wrong activation type");
    }
    return ActivationFunc::Invalid;
}


} // namespace serialize
} // namespace kcc

