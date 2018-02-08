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
{
    assert(params.verify());
}


bool
MatMulWaveOp::verify() const
{
    if (! m_WaveId.verify()) {
        return false;
    }
    if (m_WaveIdFormat == "") {
        return false;
    }
    if (m_IfmapsAtomId == -1) {
        return false;
    }
    if (m_IfmapsOffsetInAtom == -1) {
        return false;
    }
    if (m_WeightsAtomId == -1) {
        return false;
    }
    // m_WeightsOffsetInAtom is negative for waves that do NOT reload weights
    if (m_PsumBankId == -1) {
        return false;
    }
    return true;
}






bool 
MatMulWaveOp::WaveId::verify() const
{
    if (m_BatchIdx < 0) {
        return false;
    }
    if (m_OfmapFoldIdx < 0) {
        return false;
    }
    if (m_TileY < 0) {
        return false;
    }
    if (m_TileX < 0) {
        return false;
    }
    if (m_IfmapFoldIdx < 0) {
        return false;
    }
    if (m_FilterPixelX < 0) {
        return false;
    }
    if (m_FilterPixelY < 0) {
        return false;
    }
    return true;
}


bool 
MatMulWaveOp::Params::verify() const
{
    if (! m_WaveOpParams.verify()) {
        return false;
    }
    if (! m_WaveId.verify()) {
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
    // m_WeightsOffsetInAtom is negative for waves that do NOT reload weights
    if (m_PsumBankId < 0) {
        return false;
    }
    return true;
}


}}

