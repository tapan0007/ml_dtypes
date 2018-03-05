
#include <sstream>



#include "utils/inc/datatype.hpp"
#include "layers/inc/layer.hpp"
#include "wave/inc/sbatomsavewaveop.hpp"
#include "nets/inc/network.hpp"



namespace kcc {
namespace wave {

SbAtomSaveWaveOp::SbAtomSaveWaveOp(const SbAtomSaveWaveOp::Params& params,
                           const std::vector<WaveOp*>& prevWaveOps)
    : SbAtomWaveOp(params, prevWaveOps)
    , m_OfmapCount(params.m_OfmapCount)
    , m_OfmapsFoldIdx(params.m_OfmapsFoldIdx)
{
    assert(params.verify());
}


bool
SbAtomSaveWaveOp::verify() const
{
    if (! this->SbAtomWaveOp::verify()) {
        return false;
    }
    if (m_OfmapCount < 1) {
        return false;
    }
    if (m_OfmapsFoldIdx < 0) {
        return false;
    }
    return true;
}



kcc_int64
SbAtomSaveWaveOp::gSaveDataSizeInBytes() const
{
    kcc_int64 numPySize = gDataType().gSizeInBytes();
    for (int i = 0; i < 4; ++i) {
        numPySize *= gRefFileShape()[i];
    }
    return numPySize;
}


bool
SbAtomSaveWaveOp::Params::verify() const
{
    if (! this->SbAtomWaveOp::Params::verify()) {
        return false;
    }
    if (m_OfmapsFoldIdx < 0) {
        return false;
    }
    return true;
}


}}

