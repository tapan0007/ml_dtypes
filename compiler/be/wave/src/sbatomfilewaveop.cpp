
#include <sstream>


#include "utils/inc/datatype.hpp"
#include "layers/inc/layer.hpp"
#include "wave/inc/sbatomfilewaveop.hpp"
#include "nets/inc/network.hpp"



namespace kcc {
namespace wave {

SbAtomFileWaveOp::SbAtomFileWaveOp(
        const SbAtomFileWaveOp::Params& params,
        const std::vector<WaveOp*>& prevWaveOps)
    : SbAtomWaveOp(params, prevWaveOps)
    , m_IfmapCount(params.m_IfmapCount)
    , m_IfmapsFoldIdx(params.m_IfmapsFoldIdx)
    , m_IfmapsReplicate(params.m_IfmapsReplicate)
{
    assert(params.verify());
}

bool
SbAtomFileWaveOp::verify() const
{
    if (! this->SbAtomWaveOp::verify()) {
        return false;
    }
    if (m_IfmapCount < 1) {
        return false;
    }
    if (m_IfmapsFoldIdx < 0) {
        return false;
    }
    // bool m_IfmapsReplicate
    return true;
}





bool
SbAtomFileWaveOp::Params::verify() const
{
    if (! this->SbAtomWaveOp::Params::verify()) {
        return false;
    }
    if (m_IfmapsFoldIdx < 0) {
        return false;
    }
    // bool m_IfmapsReplicate
    return true;
}


}}


