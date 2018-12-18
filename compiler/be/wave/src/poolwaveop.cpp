#include <sstream>



#include "utils/inc/datatype.hpp"

#include "arch/inc/arch.hpp"
#include "nets/inc/network.hpp"
#include "wave/inc/waveconsts.hpp"
#include "wave/inc/poolwaveop.hpp"


// #define RETURN_ASSERT(x) return (x)
#define RETURN_ASSERT(x)  assert(x); return (x)


namespace kcc {
namespace wave {

PoolWaveOp::PoolWaveOp(const PoolWaveOp::Params& params,
                       const std::vector<WaveOp*>& prevWaveOps)
    : BaseClass(params, prevWaveOps)
    , m_InDtype(DataType::dataTypeId2DataType(params.m_InDtypeId))
    , m_DstStartAtMidPart(params.m_DstStartAtMidPart)

    , m_PoolFrequency(params.m_PoolFrequency)
    , m_PoolFunc(params.m_PoolFunc)

    , m_DstIsPsum(params.m_DstIsPsum)
    , m_SrcIsPsum(params.m_SrcIsPsum)
    , m_TileId(params.m_TileId)
    , m_TileIdFormat(params.m_TileIdFormat)
{
    if (m_SrcIsPsum) {
        m_SrcPsumBankId     = params.m_SrcPsumBankId;
        m_SrcPsumBankOffset = params.m_SrcPsumBankOffset;
    } else {
        m_SrcSbAddress      = params.m_SrcSbAddress;
        m_SrcStartAtMidPart      = params.m_SrcStartAtMidPart;
    }

    m_SrcWNum           = params.m_SrcWNum;
    m_SrcWStep          = params.m_SrcWStep;
    m_SrcXNum           = params.m_SrcXNum;
    m_SrcXStep          = params.m_SrcXStep;
    m_SrcYNum           = params.m_SrcYNum;
    m_SrcYStep          = params.m_SrcYStep;
    m_SrcZNum           = params.m_SrcZNum;
    m_SrcZStep          = params.m_SrcZStep;

    if (m_DstIsPsum) {
        m_DstPsumBankId     = params.m_DstPsumBankId;
        m_DstPsumBankOffset = params.m_DstPsumBankOffset;
    } else {
        m_DstSbAddress      = params.m_DstSbAddress;
        m_DstStartAtMidPart      = params.m_DstStartAtMidPart;
    }
    m_DstXNum = params.m_DstXNum;
    m_DstXStep = params.m_DstXStep;
    m_DstYNum = params.m_DstYNum;
    m_DstYStep = params.m_DstYStep;
    m_DstZNum = params.m_DstZNum;
    m_DstZStep = params.m_DstZStep;

    assert(verify());
}

bool
PoolWaveOp::verify() const
{
    const arch::PsumBuffer& psumBuf(arch::Arch::gArch().gPsumBuffer());
    if (! this->BaseClass::verify()) {
        RETURN_ASSERT(false);
    }

    if (m_DstIsPsum) {
        if (m_DstPsumBankId < 0 || m_DstPsumBankId >= psumBuf.gNumberBanks()) {
            RETURN_ASSERT(false);
        }
        if (m_DstPsumBankOffset < 0
                || m_DstPsumBankOffset >= psumBuf.gNumberBankEntries(gOutDtype().gDataTypeId()))
        {
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
    if (m_PoolFrequency < 1) {
        RETURN_ASSERT(false);
    }
    switch (m_PoolFunc) {
    case PoolType::Max:
    case PoolType::Avg:
        break;
    default:
        RETURN_ASSERT(false);
    }
    // previouswaveops: [ 1conv/i1/MatMuln0m0h0w0c0r0s0" ]

    if (m_SrcIsPsum) {
        if (m_SrcPsumBankId < 0 || m_SrcPsumBankId >= psumBuf.gNumberBanks()) {
            RETURN_ASSERT(false);
        }
        if (m_SrcPsumBankOffset < 0
                || m_SrcPsumBankOffset >= psumBuf.gNumberBankEntries(gInDtype().gDataTypeId()))
        {
            RETURN_ASSERT(false);
        }
    } else {
        if (m_SrcSbAddress < 0) {
            RETURN_ASSERT(false);
        }
    }

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

    for (auto n : m_TileId) {
        if (n < 0) {
            RETURN_ASSERT(false);
        }
    }
    if (m_TileIdFormat == "") {
        RETURN_ASSERT(false);
    }
    //waveopname;
    //waveoptype;
    return true;
}




bool
PoolWaveOp::Params::verify() const
{
    PoolWaveOp::BaseClass::Params::verify();
    return true;
}

std::string
PoolWaveOp::gTypeStrStatic()
{
    return WaveOpTypeStr::Pool;
}

}}

