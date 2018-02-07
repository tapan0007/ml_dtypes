#include <sstream>



#include "utils/inc/datatype.hpp"
#include "layers/inc/layer.hpp"
#include "wave/inc/matmulwaveop.hpp"
#include "nets/inc/network.hpp"



namespace kcc {
namespace wave {

MatMulWaveOp::MatMulWaveOp(const MatMulWaveOp::Params& params,
                           const std::vector<WaveOp*>& prevWaveOps)
    : WaveOp(params.m_WaveOpParams, prevWaveOps)
    , m_WaveId(params.m_WaveId)
    , m_WaveIdFormat(params.m_WaveIdFormat)
    , m_IfmapsAtomId(params.m_IfmapsAtomId)
    , m_IfmapsOffsetInAtom(params.m_IfmapsOffsetInAtom)
    , m_WeightsAtomId(params.m_WeightsAtomId)
    , m_WeightsOffsetInAtom(params.m_WeightsOffsetInAtom)
    , m_PsumBankId(params.m_PsumBankId)
    , m_Start(params.m_Start)
{}

bool 
MatMulWaveOp::Params::verify() const
{
    if (! m_WaveOpParams.verify()) {
        return false;
    }
    if (m_WaveIdFormat == "") {
        return false;
    }
    if (m_IfmapsAtomId < 0) {
        return false;
    }
    if (m_IfmapsOffsetInAtom < 0) {
        return false;
    }
    if (m_WeightsAtomId < 0) {
        return false;
    }
    if (m_WeightsOffsetInAtom < 0) {
        return false;
    }
    if (m_PsumBankId < 0) {
        return false;
    }
    return true;
}

}}

