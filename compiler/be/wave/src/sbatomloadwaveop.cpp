
#include <sstream>


#include "utils/inc/datatype.hpp"

#include "layers/inc/layer.hpp"
#include "layers/inc/convlayer.hpp"
#include "layers/inc/inputlayer.hpp"
#include "layers/inc/constlayer.hpp"

#include "wave/inc/sbatomloadwaveop.hpp"
#include "nets/inc/network.hpp"



namespace kcc {
namespace wave {

SbAtomLoadWaveOp::SbAtomLoadWaveOp(
        const SbAtomLoadWaveOp::Params& params,
        const std::vector<WaveOp*>& prevWaveOps)
    : SbAtomWaveOp(params, prevWaveOps)
    , m_IfmapCount(params.m_IfmapCount)
    , m_IfmapsFoldIdx(params.m_IfmapsFoldIdx)
    , m_IfmapsReplicate(params.m_IfmapsReplicate)
    , m_ContainWeights(params.m_ContainWeights)
{
    assert(params.verify());
}

kcc_int64
SbAtomLoadWaveOp::gLoadDataSizeInBytes () const
{
    kcc_int64 numPySize = gDataType().gSizeInBytes();

    for (int i = 0; i < 4; ++i) {
        numPySize *= gRefFileShape()[i];
    }
    return numPySize;
}


bool
SbAtomLoadWaveOp::verify() const
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
SbAtomLoadWaveOp::Params::verify() const
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


