#include <sstream>



#include "utils/inc/datatype.hpp"

#include "arch/inc/arch.hpp"
#include "wave/inc/waveconsts.hpp"
#include "layers/inc/layer.hpp"
#include "nets/inc/network.hpp"
#include "wave/inc/activationwaveop.hpp"

// #define RETURN_ASSERT(x) return (x)
#define RETURN_ASSERT(x)  assert(x); return (x)


namespace kcc {
namespace wave {

ActivationWaveOp::ActivationWaveOp(const ActivationWaveOp::Params& params,
                       const std::vector<WaveOp*>& prevWaveOps)
    : WaveOp(params, prevWaveOps)
    , m_ActivationFunc(params.m_ActivationFunc)
    , m_BiasAddEn(params.m_BiasAddEn)
    , m_BiasSbAddress(params.m_BiasSbAddress)
    , m_BiasStartAtMidPart(params.m_BiasStartAtMidPart)
    , m_DstIsPsum(params.m_DstIsPsum)
    , m_InDtype(DataType::dataTypeId2DataType(params.m_InDtypeId))
    , m_BiasDtype(DataType::dataTypeId2DataType(params.m_BiasDtypeId))
    , m_NumPartitions(params.m_NumPartitions)
    , m_OutDtype(DataType::dataTypeId2DataType(params.m_OutDtypeId))
    , m_SrcIsPsum(params.m_SrcIsPsum)
    , m_SrcXNum(params.m_SrcXNum)
    , m_SrcXStep(params.m_SrcXStep)
    , m_SrcYNum(params.m_SrcYNum)
    , m_SrcYStep(params.m_SrcYStep)
    , m_SrcZNum(params.m_SrcZNum)
    , m_SrcZStep(params.m_SrcZStep)
    , m_TileId(params.m_TileId)
    , m_TileIdFormat(params.m_TileIdFormat)
{
    m_DstXNum           = params.m_DstXNum;
    m_DstXStep          = params.m_DstXStep;
    m_DstYNum           = params.m_DstYNum;
    m_DstYStep          = params.m_DstYStep;
    m_DstZNum           = params.m_DstZNum;
    m_DstZStep          = params.m_DstZStep;

    if (m_DstIsPsum) {
        m_DstPsumBankId     = params.m_DstPsumBankId;
        m_DstPsumBankOffset     = params.m_DstPsumBankOffset;
    } else {
        m_DstSbAddress      = params.m_DstSbAddress;
        m_DstStartAtMidPart      = params.m_DstStartAtMidPart;
    }
    if (m_SrcIsPsum) {
        m_SrcPsumBankId     = params.m_SrcPsumBankId;
        m_SrcPsumBankOffset     = params.m_SrcPsumBankOffset;
    } else {
        m_SrcSbAddress      = params.m_SrcSbAddress;
        m_SrcStartAtMidPart      = params.m_SrcStartAtMidPart;
    }
    assert(verify());
}

bool
ActivationWaveOp::verify() const
{
    if (! this->WaveOp::verify()) {
        RETURN_ASSERT(false);
    }
    const arch::PsumBuffer& psumBuf(arch::Arch::gArch().gPsumBuffer());

    switch (m_ActivationFunc) {
    case ActivationFunc::Identity:
    case ActivationFunc::Relu:
    case ActivationFunc::LeakyRelu:
    case ActivationFunc::PRelu:
    case ActivationFunc::Sigmoid:
    case ActivationFunc::Tanh:
    case ActivationFunc::Exp:
    case ActivationFunc::Softplus:
        break;
    default:
        RETURN_ASSERT(false);
    }
    // m_BiasAddEn
    if (m_BiasSbAddress < 0) {
        RETURN_ASSERT(false);
    }

    // m_DstPsumBankOffset is assumed to be 0
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
        if (m_DstPsumBankOffset < 0 || m_DstPsumBankOffset >= psumBuf.gNumberBankEntries()) {
            RETURN_ASSERT(false);
        }
    } else {
        if (m_DstSbAddress < 0) {
            RETURN_ASSERT(false);
        }
    }

    // m_InDtype
    if (m_NumPartitions < 1) {
        RETURN_ASSERT(false);
    }
    // m_OutDtype
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
    // issues.amazon.com/issues/kaena-198
    if (m_SrcZStep == 0 && m_SrcZNum != 1) {
        RETURN_ASSERT(false);
    }

    if (m_SrcIsPsum) {
        if (m_SrcPsumBankId < 0 || m_SrcPsumBankId >= psumBuf.gNumberBanks()) {
            RETURN_ASSERT(false);
        }
        if (m_SrcPsumBankOffset < 0 || m_SrcPsumBankOffset >= psumBuf.gNumberBanks()) {
            RETURN_ASSERT(false);
        }
    } else {
        if (m_SrcSbAddress < 0) {
            RETURN_ASSERT(false);
        }
    }

    if (m_TileIdFormat == "") {
        RETURN_ASSERT(false);
    }

    for (auto n : m_TileId) {
        if (n < 0) {
            RETURN_ASSERT(false);
        }
    }

    return true;
}

TONGA_ISA_TPB_ACTIVATION_FUNC
ActivationWaveOp::gSimActivationFunc() const
{
    switch(gActivationFunc()) {
    case ActivationFunc::Identity:
        return TONGA_ISA_TPB_ACTIVATION_FUNC::TONGA_ISA_TPB_ACTIVATION_FUNC_IDENTITY;
        break;
    case ActivationFunc::Relu:
        return TONGA_ISA_TPB_ACTIVATION_FUNC::TONGA_ISA_TPB_ACTIVATION_FUNC_RELU;
        break;
    case ActivationFunc::LeakyRelu:
        return TONGA_ISA_TPB_ACTIVATION_FUNC::TONGA_ISA_TPB_ACTIVATION_FUNC_LEAKY_RELU;
        break;
    case ActivationFunc::PRelu:
        return TONGA_ISA_TPB_ACTIVATION_FUNC::TONGA_ISA_TPB_ACTIVATION_FUNC_PARAMETRIC_RELU;
        break;
    case ActivationFunc::Sigmoid:
        return TONGA_ISA_TPB_ACTIVATION_FUNC::TONGA_ISA_TPB_ACTIVATION_FUNC_SIGMOID;
        break;
    case ActivationFunc::Tanh:
        return TONGA_ISA_TPB_ACTIVATION_FUNC::TONGA_ISA_TPB_ACTIVATION_FUNC_TANH;
        break;
    case ActivationFunc::Exp:
        return TONGA_ISA_TPB_ACTIVATION_FUNC::TONGA_ISA_TPB_ACTIVATION_FUNC_EXP;
        break;
    case ActivationFunc::Softplus:
        return TONGA_ISA_TPB_ACTIVATION_FUNC::TONGA_ISA_TPB_ACTIVATION_FUNC_SOFTPLUS;
        break;
    default:
        break;
    }
    return TONGA_ISA_TPB_ACTIVATION_FUNC::TONGA_ISA_TPB_ACTIVATION_FUNC_INVALID;
}



bool
ActivationWaveOp::Params::verify() const
{
    return true;
}

std::string
ActivationWaveOp::gTypeStrStatic()
{
    return WaveOpTypeStr_Activation;
}

}}

