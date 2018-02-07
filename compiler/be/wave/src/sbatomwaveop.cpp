
#include <sstream>



#include "utils/inc/datatype.hpp"
#include "layers/inc/layer.hpp"
#include "wave/inc/sbatomwaveop.hpp"
#include "nets/inc/network.hpp"



namespace kcc {
namespace wave {

SbAtomWaveOp::SbAtomWaveOp(const SbAtomWaveOp::Params& params,
                           const std::vector<WaveOp*>& prevWaveOps)
    : WaveOp(params.m_WaveOpParams, prevWaveOps)
    , m_RefFileName(params.m_RefFileName)
    , m_AtomId(params.m_AtomId)
    , m_IfmapsFoldIdx(params.m_IfmapsFoldIdx)
    , m_Length(params.m_Length)
    , m_OffsetInFile(params.m_OffsetInFile)
    , m_IfmapsReplicate(params.m_IfmapsReplicate)
{
    assert(params.verify());
}

bool 
SbAtomWaveOp::Params::verify() const
{
    if (! m_WaveOpParams.verify()) {
        return false;
    }
    if (m_RefFileName == "") {
        return false;
    }
    if (m_AtomId < 0) {
        return false;
    }
    if (m_IfmapsFoldIdx < 0) {
        return false;
    }
    if (m_Length < 0) {
        return false;
    }
    if (m_OffsetInFile < 0) {
        return false;
    }
    return true;
}


}}

