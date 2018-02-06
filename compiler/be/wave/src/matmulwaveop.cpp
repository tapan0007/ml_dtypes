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
    , m_WaveIdFormat(params.m_WaveIdFormat)
    , m_IfmapsAtomId(params.m_IfmapsAtomId)
    , m_IfmapsOffsetInAtom(params.m_IfmapsOffsetInAtom)
    , m_PsumBankId(params.m_PsumBankId)
    , m_Start(params.m_Start)
{}

}}

