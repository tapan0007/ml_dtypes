
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
    , m_NumPartitions(params.m_NumPartitions)
    , m_FinalLayerOfmap(params.m_FinalLayerOfmap)
{
    assert(params.verify());
}


bool
SbAtomSaveWaveOp::verify() const
{
    if (! this->SbAtomWaveOp::verify()) {
        return false;
    }
    if (m_NumPartitions < 1) {
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
    return true;
}


}}

