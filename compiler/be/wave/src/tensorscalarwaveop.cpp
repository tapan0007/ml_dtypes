#include <sstream>



#include "utils/inc/datatype.hpp"

#include "arch/inc/arch.hpp"
#include "nets/inc/network.hpp"
#include "wave/inc/waveconsts.hpp"
#include "wave/inc/tensorscalarwaveop.hpp"

// #define RETURN_ASSERT(x) return (x)
#define RETURN_ASSERT(x)  assert(x); return (x)


namespace kcc {
namespace wave {

TensorScalarWaveOp::TensorScalarWaveOp(
                        const TensorScalarWaveOp::Params& params,
                        const std::vector<WaveOp*>& prevWaveOps)
    : BaseClass(params, prevWaveOps)
    , m_InDtype(DataType::dataTypeId2DataType(params.m_InDtypeId))
    , m_SrcIsPsum(params.m_SrcIsPsum)
{
    Assert(params.m_InDtypeId != DataTypeId::None, "None in data type");

    for (auto i = 0; i < 2; ++i) {
        m_AluOp[i] = params.m_AluOp[i];
        m_Reverse[i] = params.m_Reverse[i];
        m_ImmVal[i] = params.m_ImmVal[i];
    }

    /* src_a */
    if (m_SrcIsPsum) {
        m_SrcPsumBankId        = params.m_SrcPsumBankId;
        m_SrcPsumBankOffset    = params.m_SrcPsumBankOffset;
    } else {
        m_SrcSbAddress         = params.m_SrcSbAddress;
        m_SrcStartAtMidPart    = params.m_SrcStartAtMidPart;
    }
    m_SrcXStep                 = params.m_SrcXStep;
    m_SrcXNum                  = params.m_SrcXNum;
    m_SrcYStep                 = params.m_SrcYStep;
    m_SrcYNum                  = params.m_SrcYNum;
    m_SrcZStep                 = params.m_SrcZStep;
    m_SrcZNum                  = params.m_SrcZNum;

    /* dst */
    if (m_DstIsPsum) {
        m_DstPsumBankId         = params.m_DstPsumBankId;
        m_DstPsumBankOffset     = params.m_DstPsumBankOffset;
    } else {
        m_DstSbAddress          = params.m_DstSbAddress;
        m_DstStartAtMidPart          = params.m_DstStartAtMidPart;
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
TensorScalarWaveOp::verify() const
{
    if (! this->BaseClass::verify()) {
        RETURN_ASSERT(false);
    }
    const arch::PsumBuffer& psumBuf(arch::Arch::gArch().gPsumBuffer());

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

    return true;
}




bool
TensorScalarWaveOp::Params::verify() const
{
    TensorScalarWaveOp::BaseClass::Params::verify();
    return true;
}

}}

