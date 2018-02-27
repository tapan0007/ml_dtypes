#include <sstream>



#include "utils/inc/datatype.hpp"

#include "arch/inc/arch.hpp"
#include "layers/inc/layer.hpp"
#include "nets/inc/network.hpp"
#include "wave/inc/resaddwaveop.hpp"

// #define RETURN_ASSERT(x) return (x)
#define RETURN_ASSERT(x)  assert(x); return (x)


namespace kcc {
namespace wave {

ResAddWaveOp::ResAddWaveOp(const ResAddWaveOp::Params& params,
                       const std::vector<WaveOp*>& prevWaveOps)
    : WaveOp(params, prevWaveOps)
    , m_InADtype(DataType::dataTypeId2DataType(params.m_InADtypeId))
    , m_InBDtype(DataType::dataTypeId2DataType(params.m_InBDtypeId))
    , m_OutDtype(DataType::dataTypeId2DataType(params.m_OutDtypeId))
{
    /* src_a */
    if (m_SrcAIsPsum) {
        m_SrcAPsumBankId        = params.m_SrcAPsumBankId;
        m_SrcAPsumBankOffset    = params.m_SrcAPsumBankOffset;
    } else {
        m_SrcASbAtomId          = params.m_SrcASbAtomId;
        m_SrcASbOffsetInAtom    = params.m_SrcASbOffsetInAtom;
    }
    m_SrcAXStep                 = params.m_SrcAXStep;
    m_SrcAXNum                  = params.m_SrcAXNum;
    m_SrcAYStep                 = params.m_SrcAYStep;
    m_SrcAYNum                  = params.m_SrcAYNum;
    m_SrcAZStep                 = params.m_SrcAZStep;
    m_SrcAZNum                  = params.m_SrcAZNum;

    /* src_b */
    if (m_SrcBIsPsum) {
        m_SrcBPsumBankId        = params.m_SrcBPsumBankId;
        m_SrcBPsumBankOffset    = params.m_SrcBPsumBankOffset;
    } else {
        m_SrcBSbAtomId          = params.m_SrcBSbAtomId;
        m_SrcBSbOffsetInAtom    = params.m_SrcBSbOffsetInAtom;
    }
    m_SrcBXStep                 = params.m_SrcBXStep;
    m_SrcBXNum                  = params.m_SrcBXNum;
    m_SrcBYStep                 = params.m_SrcBYStep;
    m_SrcBYNum                  = params.m_SrcBYNum;
    m_SrcBZStep                 = params.m_SrcBZStep;
    m_SrcBZNum                  = params.m_SrcBZNum;

    /* dst */
    if (m_DstIsPsum) {
        m_DstPsumBankId         = params.m_DstPsumBankId;
        m_DstPsumBankOffset     = params.m_DstPsumBankOffset;
    } else {
        m_DstSbAtomId           = params.m_DstSbAtomId;
        m_DstSbOffsetInAtom     = params.m_DstSbOffsetInAtom;
    }
    m_DstXStep                  = params.m_DstXStep;
    m_DstXNum                   = params.m_DstXNum;
    m_DstYStep                  = params.m_DstYStep;
    m_DstYNum                   = params.m_DstYNum;
    m_DstZStep                  = params.m_DstZStep;
    m_DstZNum                   = params.m_DstZNum;

    m_NumPartitions             = params.m_NumPartitions;

    assert(verify());
}


bool
ResAddWaveOp::verify() const
{
    if (! this->WaveOp::verify()) {
        RETURN_ASSERT(false);
    }
    const arch::PsumBuffer& psumBuf(arch::Arch::gArch().gPsumBuffer());


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
    if (m_SrcAIsPsum) {
        if (m_SrcAPsumBankId < 0 || m_SrcAPsumBankId >= psumBuf.gNumberBanks()) {
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
    if (m_SrcBIsPsum) {
        if (m_SrcBPsumBankId < 0 || m_SrcBPsumBankId >= psumBuf.gNumberBanks()) {
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
        if (m_DstPsumBankId < 0 || m_DstPsumBankId >= psumBuf.gNumberBanks()) {
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



    if (m_NumPartitions < 1) {
        RETURN_ASSERT(false);
    }

    return true;
}




bool
ResAddWaveOp::Params::verify() const
{
    return true;
}

}}

