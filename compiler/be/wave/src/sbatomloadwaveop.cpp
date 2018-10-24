
#include <sstream>


#include "utils/inc/datatype.hpp"

#include "layers/inc/layer.hpp"
#include "layers/inc/convlayer.hpp"
#include "layers/inc/inputlayer.hpp"
#include "layers/inc/constlayer.hpp"

#include "wave/inc/waveconsts.hpp"
#include "wave/inc/sbatomloadwaveop.hpp"
#include "nets/inc/network.hpp"



namespace kcc {
namespace wave {

SbAtomLoadWaveOp::SbAtomLoadWaveOp(
        const SbAtomLoadWaveOp::Params& params,
        const std::vector<WaveOp*>& prevWaveOps)
    : SbAtomWaveOp(params, prevWaveOps)
    , m_ContainWeights(params.m_ContainWeights)
    , m_IfmapReplicationNumRows(params.m_IfmapReplicationNumRows)
    , m_IfmapReplicationResolution(params.m_IfmapReplicationResolution)
    , m_IfmapReplicationStepBytes(params.m_IfmapReplicationStepBytes)
    , m_SrcStepElem(params.m_SrcStepElem)
{
    assert(params.verify());
}

kcc_int64
SbAtomLoadWaveOp::gLoadDataSizeInBytes () const
{
    kcc_int64 numPySize = gDataType().gSizeInBytes();

//    for (int i = 0; i < 4; ++i) {
//        numPySize *= gRefFileShape()[i];
    for (auto n : gRefFileShape()) {
        numPySize *= n;
    }
    return numPySize;
}


bool
SbAtomLoadWaveOp::verify() const
{
    if (! this->SbAtomWaveOp::verify()) {
        return false;
    }
    if (m_IfmapReplicationNumRows < 0) {
        return false;
    }
    if (m_IfmapReplicationResolution < 0) {
        return false;
    }
    if (m_IfmapReplicationStepBytes < 0) {
        return false;
    }
    if (m_SrcStepElem < 0) {
        return false;
    }
    return true;
} // SbAtomLoadWaveOp::verify





bool
SbAtomLoadWaveOp::Params::verify() const
{
    if (! this->SbAtomWaveOp::Params::verify()) {
        return false;
    }
    // bool m_IfmapsReplicate
    return true;
}

std::string
SbAtomLoadWaveOp::gTypeStrStatic()
{
    return WaveOpTypeStr_SBAtomLoad;
}

}}


