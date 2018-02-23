#include <sstream>



#include "utils/inc/datatype.hpp"

#include "arch/inc/arch.hpp"
#include "layers/inc/layer.hpp"
#include "nets/inc/network.hpp"
#include "wave/inc/activationwaveop.hpp"



namespace kcc {
namespace wave {

ActivationWaveOp::ActivationWaveOp(const ActivationWaveOp::Params& params,
                       const std::vector<WaveOp*>& prevWaveOps)
    : WaveOp(params, prevWaveOps)
    , m_ActivationFunc(params.m_ActivationFunc)
    , m_BiasAddEn(params.m_BiasAddEn)
    , m_BiasAtomId(params.m_BiasAtomId)
    , m_BiasOffsetInAtom(params.m_BiasOffsetInAtom)
    , m_DstPsumBankId(params.m_DstPsumBankId)
    , m_DstXNum(params.m_DstXNum)
    , m_DstXStep(params.m_DstXStep)
    , m_DstYNum(params.m_DstYNum)
    , m_DstYStep(params.m_DstYStep)
    , m_DstZNum(params.m_DstZNum)
    , m_DstZStep(params.m_DstZStep)
    , m_InDtype(DataType::dataTypeId2DataType(params.m_InDtypeId))
    , m_NumPartitions(params.m_NumPartitions)
    , m_OutDtype(DataType::dataTypeId2DataType(params.m_OutDtypeId))
    , m_SrcPsumBankId(params.m_SrcPsumBankId)
    , m_SrcXNum(params.m_SrcXNum)
    , m_SrcXStep(params.m_SrcXStep)
    , m_SrcYNum(params.m_SrcYNum)
    , m_SrcYStep(params.m_SrcYStep)
    , m_SrcZNum(params.m_SrcZNum)
    , m_SrcZStep(params.m_SrcZStep)
    , m_TileId(params.m_TileId)
    , m_TileIdFormat(params.m_TileIdFormat)
{
    assert(verify());
}

bool
ActivationWaveOp::verify() const
{
    if (! this->WaveOp::verify()) {
        return false;
    }
    const arch::PsumBuffer& psumBuf(arch::Arch::gArch().gPsumBuffer());
    switch (m_ActivationFunc) {
    case ActivationFunc_Identity:
    case ActivationFunc_Relu:
    case ActivationFunc_LRelu:
    case ActivationFunc_PRelu:
    case ActivationFunc_Sigmoid:
    case ActivationFunc_Tanh:
    case ActivationFunc_Exp:
        return true;
    default:
        return false;
    }
    // m_BiasAddEn
    if (m_BiasAtomId < 0) {
        return false;
    }
    if (m_BiasOffsetInAtom < 0) {
        return false;
    }
    if (m_DstPsumBankId < 0 || m_DstPsumBankId >= psumBuf.gNumberBanks()) {
        return false;
    }
    if (m_DstXNum < 1) {
        return false;
    }
    if (m_DstXStep < 1) {
        return false;
    }
    if (m_DstYNum < 1) {
        return false;
    }
    if (m_DstYStep < 1) {
        return false;
    }
    if (m_DstZNum < 1) {
        return false;
    }
    if (m_DstZStep < 1) {
        return false;
    }
    // m_InDtype
    if (m_NumPartitions < 1) {
        return false;
    }
    // m_OutDtype
    if (m_SrcPsumBankId < 1) {
        return false;
    }
    if (m_SrcXNum < 1) {
        return false;
    }
    if (m_SrcXStep < 1) {
        return false;
    }
    if (m_SrcYNum < 1) {
        return false;
    }
    if (m_SrcYStep < 1) {
        return false;
    }
    if (m_SrcZNum < 1) {
        return false;
    }
    // issues.amazon.com/issues/kaena-198
    if (m_SrcZStep <1 ){
        return false;
    }
    std::array<kcc_int32, 4>    m_TileId;
    if (m_TileIdFormat == "") {
        return false;
    }

    for (auto n : m_TileId) {
        if (n < 0) {
            return false;
        }
    }

    return true;
}




bool
ActivationWaveOp::Params::verify() const
{
    return true;
}

}}

