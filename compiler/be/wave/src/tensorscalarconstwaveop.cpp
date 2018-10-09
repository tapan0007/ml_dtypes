#include <sstream>



#include "utils/inc/datatype.hpp"

#include "arch/inc/arch.hpp"
#include "layers/inc/layer.hpp"
#include "nets/inc/network.hpp"
#include "wave/inc/waveconsts.hpp"
#include "wave/inc/tensorscalarconstwaveop.hpp"

// #define RETURN_ASSERT(x) return (x)
#define RETURN_ASSERT(x)  assert(x); return (x)


namespace kcc {
namespace wave {

TensorScalarConstWaveOp::TensorScalarConstWaveOp(
                        const TensorScalarConstWaveOp::Params& params,
                        const std::vector<WaveOp*>& prevWaveOps)
    : PoolEngWaveOp(params, prevWaveOps)
    , m_TypeStr(params.m_WaveOpType)
    , m_InDtype(DataType::dataTypeId2DataType(params.m_InDtypeId))
    , m_SrcIsPsum(params.m_SrcIsPsum)
    , m_DstIsPsum(params.m_DstIsPsum)
{
    Assert(params.m_InDtypeId != DataTypeId::None, "None in data type");

    m_AluOp[0] = params.m_AluOp[0];
    m_AluOp[1] = params.m_AluOp[1];
    m_ImmVal[0] = params.m_ImmVal[0];
    m_ImmVal[1] = params.m_ImmVal[1];

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
TensorScalarConstWaveOp::verify() const
{
    if (! this->WaveOp::verify()) {
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
        if (m_SrcPsumBankOffset < 0) {
            RETURN_ASSERT(false);
        }
    } else {
        if (m_SrcSbAddress < 0) {
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
    if (m_DstIsPsum) {
        if (m_DstPsumBankId < 0 || m_DstPsumBankId >= psumBuf.gNumberBanks()) {
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


    if (m_NumPartitions < 1) {
        RETURN_ASSERT(false);
    }
    if (m_TypeStr == "") {
        RETURN_ASSERT(false);
    }

    return true;
}




bool
TensorScalarConstWaveOp::Params::verify() const
{
    return true;
}

std::string
TensorScalarConstWaveOp::gTypeStr() const
{
    return m_TypeStr;
}

}}
