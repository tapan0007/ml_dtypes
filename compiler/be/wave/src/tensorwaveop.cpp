#include <sstream>



#include "utils/inc/datatype.hpp"

#include "arch/inc/arch.hpp"
#include "nets/inc/network.hpp"
#include "wave/inc/waveconsts.hpp"
#include "wave/inc/tensorwaveop.hpp"

// #define RETURN_ASSERT(x) return (x)
#define RETURN_ASSERT(x)  assert(x); return (x)


namespace kcc {
namespace wave {

TensorWaveOp::TensorWaveOp(
                        const TensorWaveOp::Params& params,
                        const std::vector<WaveOp*>& prevWaveOps)
    : PoolEngWaveOp(params, prevWaveOps)
    , m_TypeStr(params.m_WaveOpType)
    , m_DstIsPsum(params.m_DstIsPsum)
{
    Assert(params.m_OutDtypeId != DataTypeId::None, "None out data type");
    m_NumPartitions             = params.m_NumPartitions;

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

    assert(verify());
}


bool
TensorWaveOp::verify() const
{
    if (! this->WaveOp::verify()) {
        RETURN_ASSERT(false);
    }

    if (m_NumPartitions < 1) {
        RETURN_ASSERT(false);
    }
    if (m_TypeStr == "") {
        RETURN_ASSERT(false);
    }
    const arch::PsumBuffer& psumBuf(arch::Arch::gArch().gPsumBuffer());


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


    return true;
}




bool
TensorWaveOp::Params::verify() const
{
    return true;
}

std::string
TensorWaveOp::gTypeStr() const
{
    return m_TypeStr; // from JSON
}

}}

