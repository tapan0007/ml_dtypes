
#include <sstream>



#include "utils/inc/datatype.hpp"
#include "layers/inc/layer.hpp"
#include "wave/inc/sbatomwaveop.hpp"
#include "nets/inc/network.hpp"



namespace kcc {
namespace wave {

SbAtomWaveOp::SbAtomWaveOp(const SbAtomWaveOp::Params& params,
                           const std::vector<WaveOp*>& prevWaveOps)
    : WaveOp(params, prevWaveOps)
    , m_RefFileName(params.m_RefFileName)
    , m_BatchFoldIdx(params.m_BatchFoldIdx)
    , m_AtomId(params.m_AtomId)
    , m_Length(params.m_Length)
    , m_OffsetInFile(params.m_OffsetInFile)
{
    assert(params.verify());
}


bool 
SbAtomWaveOp::verify() const
{
    if (! this-> WaveOp::verify()) {
        return false;
    }
    if (m_RefFileName == "") {
        return false;
    }
    if (m_BatchFoldIdx < 0) {
        return false;
    }
    if (m_AtomId < 0) {
        return false;
    }
    if (m_Length < 0) { // TODO: should be <= 0
        return false;
    }
    if (m_OffsetInFile < 0) {
        return false;
    }
    return true;
}



bool 
SbAtomWaveOp::Params::verify() const
{
    if (! this-> WaveOp::Params::verify()) {
        return false;
    }
    if (m_RefFileName == "") {
        return false;
    }
    if (m_BatchFoldIdx < 0) {
        return false;
    }
    if (m_AtomId < 0) {
        return false;
    }
    if (m_Length < 0) {  // TODO: should be <=
        return false;
    }
    if (m_OffsetInFile < 0) {
        return false;
    }
    return true;
}


}}

