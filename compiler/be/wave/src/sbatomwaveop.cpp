
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
{}

}}

