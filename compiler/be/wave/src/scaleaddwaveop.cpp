#include <sstream>



#include "utils/inc/datatype.hpp"

#include "arch/inc/arch.hpp"
#include "nets/inc/network.hpp"
#include "wave/inc/waveconsts.hpp"
#include "wave/inc/scaleaddwaveop.hpp"


// #define RETURN_ASSERT(x) return (x)
#define RETURN_ASSERT(x)  assert(x); return (x)


namespace kcc {
namespace wave {

ScaleAddWaveOp::ScaleAddWaveOp(const ScaleAddWaveOp::Params& params,
                               const std::vector<WaveOp*>& prevWaveOps)
    : BaseClass(params, prevWaveOps)
    , m_DstSbAddress(params.m_DstSbAddress)
    , m_DstXNum(params.m_DstXNum)
    , m_DstXStep(params.m_DstXStep)
    , m_DstYNum(params.m_DstYNum)
    , m_DstYStep(params.m_DstYStep)
    , m_DstZNum(params.m_DstZNum)
    , m_DstZStep(params.m_DstZStep)
    , m_SrcIsPsum(params.m_SrcIsPsum)
    , m_Scale(params.m_Scale)
    , m_Offset(params.m_Offset)
{
    if (m_SrcIsPsum) {
        m_SrcPsumBankId     = params.m_SrcPsumBankId;
        m_SrcPsumBankOffset = params.m_SrcPsumBankOffset;
    } else {
        m_SrcSbAddress      = params.m_SrcSbAddress;
    }

    m_SrcXStep          = params.m_SrcXStep;
    m_SrcYNum           = params.m_SrcYNum;
    m_SrcYStep          = params.m_SrcYStep;
    m_SrcZNum           = params.m_SrcZNum;
    m_SrcZStep          = params.m_SrcZStep;
    assert(verify());
}

bool
ScaleAddWaveOp::verify() const
{
    const arch::PsumBuffer& psumBuf(arch::Arch::gArch().gPsumBuffer());
    if (! this->BaseClass::verify()) {
        RETURN_ASSERT(false);
    }
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
    if (m_NumPartitions < 1) {
        RETURN_ASSERT(false);
    }

    if (m_SrcIsPsum) {
        if (m_SrcPsumBankId < 0 || m_SrcPsumBankId >= psumBuf.gNumberBanks()) {
            RETURN_ASSERT(false);
        }
        if (m_SrcPsumBankOffset < 0 || m_SrcPsumBankOffset >= psumBuf.gNumberBankEntries()) {
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
ScaleAddWaveOp::Params::verify() const
{
    ScaleAddWaveOp::BaseClass::Params::verify();
    return true;
}

std::string
ScaleAddWaveOp::gTypeStrStatic()
{
    return WaveOpTypeStr_ScaleAdd;
}

}}

