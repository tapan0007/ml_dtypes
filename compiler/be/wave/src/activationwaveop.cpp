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
    , m_ActType(params.m_ActType)
    , m_BiasAddEn(params.m_BiasAddEn)
    , m_BiasAtomId(params.m_BiasAtomId)
    , m_BiasOffsetInAtom(params.m_BiasOffsetInAtom)
    , m_PsumBankIdDst(params.m_PsumBankIdDst)
    , m_PsumBankIdSrc(params.m_PsumBankIdSrc)
    , m_TileIdFormat(params.m_TileIdFormat)
    , m_TileId(params.m_TileId)
{
    assert(verify());
}

bool 
ActivationWaveOp::verify() const
{
    const arch::PsumBuffer& psumBuf(arch::Arch::gArch().gPsumBuffer());
    if (! this->WaveOp::verify()) {
        return false;
    }
    switch (m_ActType) {
    case ActivationType_Identity:
    case ActivationType_Relu:
    case ActivationType_LRelu:
    case ActivationType_PRelu:
    case ActivationType_Sigmoid:
    case ActivationType_Tanh:
    case ActivationType_Exp:
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
    if (m_PsumBankIdDst < 0 || m_PsumBankIdDst >= psumBuf.gNumberBanks()) {
        return false;
    }
    if (m_PsumBankIdSrc < 0 || m_PsumBankIdSrc >= psumBuf.gNumberBanks()) {
        return false;
    }
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

