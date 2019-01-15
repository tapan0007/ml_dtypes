#include <iostream>

#include "wave/inc/waveconsts.hpp"
#include "wave/inc/regshufflewaveop.hpp"
#include "serialize/inc/serwaveop.hpp"


namespace kcc {
namespace serialize {

SerWaveOp::SerWaveOp()
{
    m_RefFileShape.resize(utils::TensorParams::NUM_DIMS, 1);
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
    if (m_RefFileFormat != "NCHW" && m_RefFileFormat != "CRSM" &&
        m_RefFileFormat != "HNWC" && m_RefFileFormat != "CNcHW" &&
        m_RefFileFormat != "NHWC" && m_RefFileFormat != "NC") {
        RETURN_ASSERT(false);
    }
    //if (m_RefFileShape.size() != 4) {
    //    RETURN_ASSERT(false);
    //}
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
    if (m_PsumXStep == 0 && m_PsumXNum != 1) {
        RETURN_ASSERT(false);
    }
    if (m_PsumYNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_PsumYStep == 0 && m_PsumYNum != 1) {
        RETURN_ASSERT(false);
    }
    if (m_PsumZNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_PsumZStep == 0 && m_PsumZNum != 1) {
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
    if (m_DstXStep == 0 && m_DstXNum != 1) {
        RETURN_ASSERT(false);
    }
    if (m_DstYNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_DstYStep == 0 && m_DstYNum != 1) {
        RETURN_ASSERT(false);
    }
    if (m_DstZNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_DstZStep == 0 && m_DstZNum != 1) {
        RETURN_ASSERT(false);
    }
    if (m_InDtype == "") {
        RETURN_ASSERT(false);
    }
    if (m_NumPartitions < 1) {
        RETURN_ASSERT(false);
    }
    if (m_OutDtype == "") {
        RETURN_ASSERT(false);
    }
    if (m_PoolFunc == "") {
        RETURN_ASSERT(false);
    }

    // previouswaveops": [ 1conv/i1/MatMuln0m0h0w0c0r0s0" ]

    if (m_SrcWNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcWStep == 0 && m_SrcWNum != 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcXNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcXStep == 0 && m_SrcXNum != 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcYNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcYStep == 0 && m_SrcYNum != 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcZNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcZStep == 0 && m_SrcZNum != 1) {
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

    return true;
}


bool
SerWaveOp::verifyReciprocal() const
{
    if (m_DstSbAddress < 0) {
        RETURN_ASSERT(false);
    }
    if (m_DstXNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_DstXStep == 0 && m_DstXNum != 1) {
        RETURN_ASSERT(false);
    }
    if (m_DstYNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_DstYStep == 0 && m_DstYNum != 1) {
        RETURN_ASSERT(false);
    }
    if (m_DstZNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_DstZStep == 0 && m_DstZNum != 1) {
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

    // previouswaveops": [ 1conv/i1/MatMuln0m0h0w0c0r0s0" ]

    if (m_SrcXNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcXStep == 0 && m_SrcXNum != 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcYNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcYStep == 0 && m_SrcYNum != 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcZNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcZStep == 0 && m_SrcZNum != 1) {
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

    //waveopname": "1conv/i1/Pooln0m0h0w0",
    //waveoptype": "Pool"

    return true;
}


bool
SerWaveOp::verifyRegLoad() const
{
    if (m_InDtype == "") {
        RETURN_ASSERT(false);
    }
    if (m_NumPartitions < 1) {
        RETURN_ASSERT(false);
    }
    if (!m_ParallelMode && m_NumPartitions > 1) {
        RETURN_ASSERT(false);
    }

    if (m_SrcXNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcXStep == 0 && m_SrcXNum != 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcYNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcYStep == 0 && m_SrcYNum != 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcZNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcZStep == 0 && m_SrcZNum != 1) {
        RETURN_ASSERT(false);
    }

    if (m_SrcSbAddress < 0) {
        RETURN_ASSERT(false);
    }

    return true;
}

bool
SerWaveOp::verifyRegStore() const
{
    if (m_OutDtype == "") {
        RETURN_ASSERT(false);
    }
    if (m_NumPartitions < 1) {
        RETURN_ASSERT(false);
    }
    if (!m_ParallelMode && m_NumPartitions > 1) {
        RETURN_ASSERT(false);
    }

    if (m_DstXNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_DstXStep == 0 && m_DstXNum != 1) {
        RETURN_ASSERT(false);
    }
    if (m_DstYNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_DstYStep == 0 && m_DstYNum != 1) {
        RETURN_ASSERT(false);
    }
    if (m_DstZNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_DstZStep == 0 && m_DstZNum != 1) {
        RETURN_ASSERT(false);
    }

    if (m_DstSbAddress < 0) {
        RETURN_ASSERT(false);
    }

    return true;
}

bool
SerWaveOp::verifyRegShuffle() const
{
    if (m_StartReg < 0 || m_StartReg >= wave::RegShuffleWaveOp::MaxNumRegs) {
        RETURN_ASSERT(false);
    }
    for (auto k : m_InSel) {
        if (k < 0 || k >= wave::RegShuffleWaveOp::MaxNumRegs) {
            RETURN_ASSERT(false);
        }
    }

    return true;
}


bool
SerWaveOp::verifyActivation() const
{
    if (m_ActivationFunc == "") {
        RETURN_ASSERT(false);
    }
    if (m_BiasSbAddress < 0) {
        RETURN_ASSERT(false);
    }
    if (m_DstXNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_DstXStep == 0 && m_DstXNum != 1) {
        RETURN_ASSERT(false);
    }
    if (m_DstYNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_DstYStep == 0 && m_DstYNum != 1) {
        RETURN_ASSERT(false);
    }
    if (m_DstZNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_DstZStep == 0 && m_DstZNum != 1) {
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
        if (m_SrcPsumBankOffset < 0) {
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
    if (m_SrcXStep == 0 && m_SrcXNum != 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcYNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcYStep == 0 && m_SrcYNum != 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcZNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcZStep == 0 && m_SrcZNum != 1) {
        RETURN_ASSERT(false);
    }

    return true;
}

bool
SerWaveOp::verifyClipByValue() const
{
    if (m_DstXNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_DstXStep == 0 && m_DstXNum != 1) {
        RETURN_ASSERT(false);
    }
    if (m_DstYNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_DstYStep == 0 && m_DstYNum != 1) {
        RETURN_ASSERT(false);
    }
    if (m_DstZNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_DstZStep == 0 && m_DstZNum != 1) {
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

    if (m_InDtype == "") {
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
        if (m_SrcPsumBankOffset < 0) {
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
    if (m_SrcXStep == 0 && m_SrcXNum != 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcYNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcYStep == 0 && m_SrcYNum != 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcZNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcZStep == 0 && m_SrcZNum != 1) {
        RETURN_ASSERT(false);
    }
    return true;
}



bool
SerWaveOp::verifyTensor() const
{
    if (m_NumPartitions < 1) {
        RETURN_ASSERT(false);
    }
    if (m_OutDtype == "") {
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
    if (m_DstXStep == 0 && m_DstXNum != 1) {
        RETURN_ASSERT(false);
    }
    if (m_DstYNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_DstYStep == 0 && m_DstYNum != 1) {
        RETURN_ASSERT(false);
    }
    if (m_DstZNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_DstZStep == 0 && m_DstZNum != 1) {
        RETURN_ASSERT(false);
    }



    if (m_IsScalarOp) {
        if (m_InDtype == "") {
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
        if (m_SrcXNum < 1) {
            RETURN_ASSERT(false);
        }
        if (m_SrcXStep == 0 && m_SrcXNum != 1) {
            RETURN_ASSERT(false);
        }
        if (m_SrcYNum < 1) {
            RETURN_ASSERT(false);
        }
        if (m_SrcYStep == 0 && m_SrcYNum != 1) {
            RETURN_ASSERT(false);
        }
        if (m_SrcZNum < 1) {
            RETURN_ASSERT(false);
        }
        if (m_SrcZStep == 0 && m_SrcZNum != 1) {
            RETURN_ASSERT(false);
        }
    } else {
        if (m_InADtype == "") {
            RETURN_ASSERT(false);
        }
        if (m_InBDtype == "") {
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
        if (m_SrcAXStep == 0 && m_SrcAXNum != 1) {
            RETURN_ASSERT(false);
        }
        if (m_SrcAYNum < 1) {
            RETURN_ASSERT(false);
        }
        if (m_SrcAYStep == 0 && m_SrcAYNum != 1) {
            RETURN_ASSERT(false);
        }
        if (m_SrcAZNum < 1) {
            RETURN_ASSERT(false);
        }
        if (m_SrcAZStep == 0 && m_SrcAZNum != 1) {
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
        if (m_SrcBXStep == 0 && m_SrcBXNum != 1) {
            RETURN_ASSERT(false);
        }
        if (m_SrcBYNum < 1) {
            RETURN_ASSERT(false);
        }
        if (m_SrcBYStep == 0 && m_SrcBYNum != 1) {
            RETURN_ASSERT(false);
        }
        if (m_SrcBZNum < 1) {
            RETURN_ASSERT(false);
        }
        if (m_SrcBZStep == 0 && m_SrcBZNum != 1) {
            RETURN_ASSERT(false);
        }
    }

    return true;
}

bool
SerWaveOp::verifyTensorTensor() const
{
    if (m_NumPartitions < 1) {
        RETURN_ASSERT(false);
    }

    if (m_OutDtype == "") {
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
    if (m_DstXStep == 0 && m_DstXNum != 1) {
        RETURN_ASSERT(false);
    }
    if (m_DstYNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_DstYStep == 0 && m_DstYNum != 1) {
        RETURN_ASSERT(false);
    }
    if (m_DstZNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_DstZStep == 0 && m_DstZNum != 1) {
        RETURN_ASSERT(false);
    }


    if (m_InADtype == "") {
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
    if (m_SrcAXStep == 0 && m_SrcAXNum != 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcAYNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcAYStep == 0 && m_SrcAYNum != 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcAZNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcAZStep == 0 && m_SrcAZNum != 1) {
        RETURN_ASSERT(false);
    }


    if (m_InBDtype == "") {
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
    if (m_SrcBXStep == 0 && m_SrcBXNum != 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcBYNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcBYStep == 0 && m_SrcBYNum != 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcBZNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcBZStep == 0 && m_SrcBZNum != 1) {
        RETURN_ASSERT(false);
    }

    return true;
}

bool
SerWaveOp::verifyTensorScalar() const
{
    if (m_NumPartitions < 1) {
        RETURN_ASSERT(false);
    }

    if (m_OutDtype == "") {
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
    if (m_DstXStep == 0 && m_DstXNum != 1) {
        RETURN_ASSERT(false);
    }
    if (m_DstYNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_DstYStep == 0 && m_DstYNum != 1) {
        RETURN_ASSERT(false);
    }
    if (m_DstZNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_DstZStep == 0 && m_DstZNum != 1) {
        RETURN_ASSERT(false);
    }

    if (m_InDtype == "") {
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
    if (m_SrcXNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcXStep == 0 && m_SrcXNum != 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcYNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcYStep == 0 && m_SrcYNum != 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcZNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcZStep == 0 && m_SrcZNum != 1) {
        RETURN_ASSERT(false);
    }

    return true;
}

bool
SerWaveOp::verifyTpbCopy() const
{
    if (m_PairLoadWaveOp == "") {
        RETURN_ASSERT(false);
    }
    if (m_SrcSbAddress < 0) {
        RETURN_ASSERT(false);
    }
    if (m_DstSbAddress < 0) {
        RETURN_ASSERT(false);
    }
    if (m_SizeInBytes < 0) {
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
    if (m_SrcAXStep == 0 && m_SrcAXNum != 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcAYNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcAYStep == 0 && m_SrcAYNum != 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcAZNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcAZStep == 0 && m_SrcAZNum != 1) {
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
    if (m_SrcBXStep == 0 && m_SrcBXNum != 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcBYNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcBYStep == 0 && m_SrcBYNum != 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcBZNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcBZStep == 0 && m_SrcBZNum != 1) {
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
    if (m_DstXStep == 0 && m_DstXNum != 1) {
        RETURN_ASSERT(false);
    }
    if (m_DstYNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_DstYStep == 0 && m_DstYNum != 1) {
        RETURN_ASSERT(false);
    }
    if (m_DstZNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_DstZStep == 0 && m_DstZNum != 1) {
        RETURN_ASSERT(false);
    }

    return true;
}


bool
SerWaveOp::verifyScaleAdd() const
{
    if (m_InDtype == "") {
        RETURN_ASSERT(false);
    }
    if (m_OutDtype == "") {
        RETURN_ASSERT(false);
    }
    if (m_NumPartitions < 1) {
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
    if (m_SrcXNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcXStep == 0 && m_SrcXNum != 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcYNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcYStep == 0 && m_SrcYNum != 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcZNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcZStep == 0 && m_SrcZNum != 1) {
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
    if (m_DstXStep == 0 && m_DstXNum != 1) {
        RETURN_ASSERT(false);
    }
    if (m_DstYNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_DstYStep == 0 && m_DstYNum != 1) {
        RETURN_ASSERT(false);
    }
    if (m_DstZNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_DstZStep == 0 && m_DstZNum != 1) {
        RETURN_ASSERT(false);
    }

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

    if (m_WaveOpType == wave::WaveOpTypeStr::SBAtomLoad) {
        return verifySbAtomLoad();
    } else if (m_WaveOpType == wave::WaveOpTypeStr::SBAtomSave) {
        return verifySbAtomSave();
    } else if (m_WaveOpType == wave::WaveOpTypeStr::MatMul) {
        return verifyMatMul();
    } else if (m_WaveOpType == wave::WaveOpTypeStr::Pool) {
        return verifyPool();
    } else if (m_WaveOpType == wave::WaveOpTypeStr::Reciprocal) {
        return verifyReciprocal();
    } else if (m_WaveOpType == wave::WaveOpTypeStr::RegLoad) {
        return verifyRegLoad();
    } else if (m_WaveOpType == wave::WaveOpTypeStr::RegStore) {
        return verifyRegStore();
    } else if (m_WaveOpType == wave::WaveOpTypeStr::RegShuffle) {
        return verifyRegShuffle();
    } else if (m_WaveOpType == wave::WaveOpTypeStr::Activation) {
        return verifyActivation();
    } else if (m_WaveOpType == wave::WaveOpTypeStr::ClipByValue) {
        return verifyClipByValue();
    } else if (m_WaveOpType == wave::WaveOpTypeStr::TensorTensor) {
        return verifyTensorTensor();
    } else if (m_WaveOpType == wave::WaveOpTypeStr::TensorScalar) {
        return verifyTensorScalar();
    } else if (m_WaveOpType == wave::WaveOpTypeStr::Nop) {
        return verifyNop();
    } else if (m_WaveOpType == wave::WaveOpTypeStr::ScaleAdd) {
        return verifyScaleAdd();
    } else if (m_WaveOpType == wave::WaveOpTypeStr::ResAdd) {
        return verifyResAdd();
    } else if (m_WaveOpType == wave::WaveOpTypeStr::Maximum) {
        return verifyTensor();
    } else if (m_WaveOpType == wave::WaveOpTypeStr::Minimum) {
        return verifyTensor();
    } else if (m_WaveOpType == wave::WaveOpTypeStr::Multiply) {
        return verifyTensor();
    } else if (m_WaveOpType == wave::WaveOpTypeStr::Sub) {
        return verifyTensor();
    } else if (m_WaveOpType == wave::WaveOpTypeStr::Add) {
        return verifyTensor();
    } else if (m_WaveOpType == wave::WaveOpTypeStr::TpbCopy) {
        return verifyTpbCopy();
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
        return WaveOpKey::ActivationFunc_Identity;
        break;
    case ActivationFunc::Relu:
        return WaveOpKey::ActivationFunc_Relu;
        break;
    case ActivationFunc::LeakyRelu:
        return WaveOpKey::ActivationFunc_LeakyRelu;
        break;
    case ActivationFunc::PRelu:
        return WaveOpKey::ActivationFunc_Prelu;
        break;
    case ActivationFunc::Sigmoid:
        return WaveOpKey::ActivationFunc_Sigmoid;
        break;
    case ActivationFunc::Tanh:
        return WaveOpKey::ActivationFunc_Tanh;
        break;
    case ActivationFunc::Exp:
        return WaveOpKey::ActivationFunc_Exp;
        break;
    case ActivationFunc::Softplus:
        return WaveOpKey::ActivationFunc_Softplus;
        break;
    case ActivationFunc::Sqrt:
        return WaveOpKey::ActivationFunc_Sqrt;
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
    if (actType == WaveOpKey::ActivationFunc_Identity
        || actType == WaveOpKey::ActivationFunc_None /* until Jeff fixes none */) {
        return ActivationFunc::Identity;
    } else if (actType  == WaveOpKey::ActivationFunc_Relu) {
        return ActivationFunc::Relu;
    } else if (actType  == WaveOpKey::ActivationFunc_LeakyRelu) {
        return ActivationFunc::LeakyRelu;
    } else if (actType  == WaveOpKey::ActivationFunc_Prelu) {
        return ActivationFunc::PRelu;
    } else if (actType  == WaveOpKey::ActivationFunc_Sigmoid) {
        return ActivationFunc::Sigmoid;
    } else if (actType  == WaveOpKey::ActivationFunc_Tanh) {
        return ActivationFunc::Tanh;
    } else if (actType  == WaveOpKey::ActivationFunc_Exp) {
        return ActivationFunc::Exp;
    } else if (actType  == WaveOpKey::ActivationFunc_Softplus) {
        return ActivationFunc::Softplus;
    } else if (actType  == WaveOpKey::ActivationFunc_Sqrt) {
        return ActivationFunc::Sqrt;
    } else {
        assert(false && "Wrong activation type");
    }
    return ActivationFunc::Invalid;
}


//===================================================
SerWaveOp::Sync::Sync(events::EventSetMode setMode, events::EventId eventId, events::EventWaitMode waitMode)
    : m_WithEvent(true)
{
    new (&m_EventSync) EventSync(setMode, eventId, waitMode);
}

SerWaveOp::Sync::Sync(const char* que, kcc_int32 trigOrd)
    : m_WithEvent(false)
{
    new (&m_SemSync) SemSync(que, trigOrd);
}

SerWaveOp::Sync::Sync(const char* que, kcc_int32 trigOrd, const char* que1, kcc_int32 trigOrd1)
    : m_WithEvent(false)
{
    new (&m_SemSync) SemSync(que, trigOrd, que1, trigOrd1);
}

SerWaveOp::Sync::Sync(const Sync& rhs)
    : m_WithEvent(rhs.m_WithEvent)
{
    if (m_WithEvent) {
        new (&m_EventSync) EventSync(rhs.m_EventSync);
    } else {
        new (&m_SemSync) SemSync(rhs.m_SemSync);
    }
}

SerWaveOp::Sync::~Sync()
{
    if (m_WithEvent) {
        m_EventSync.~EventSync();
    } else {
        m_SemSync.~SemSync();
    }
}


//===================================================
void
SerWaveOp::addPreviousEventSync(events::EventSetMode setMode,
                                events::EventId eventId,
                                events::EventWaitMode waitMode)
{
    const Sync sync(setMode, eventId, waitMode);
    m_PreviousSyncs.push_back(sync);
}

void
SerWaveOp::addPreviousSemaphoreSync(const char* prevSemaphore, kcc_int32 trigOrd)
{
    const Sync sync(prevSemaphore, trigOrd);
    m_PreviousSyncs.push_back(sync);
}

void
SerWaveOp::addPreviousSemaphoreSync(const char* prevSemaphore, kcc_int32 trigOrd,
    const char* prevSemaphore1, kcc_int32 trigOrd1)
{
    const Sync sync(prevSemaphore, trigOrd, prevSemaphore1, trigOrd1);
    m_PreviousSyncs.push_back(sync);
}

} // namespace serialize
} // namespace kcc

