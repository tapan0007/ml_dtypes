
#include <sstream>



#include "utils/inc/datatype.hpp"
#include "layers/inc/layer.hpp"
#include "wave/inc/sbatomwaveop.hpp"
#include "nets/inc/network.hpp"



namespace kcc {
namespace wave {

SbAtomWaveOp::SbAtomWaveOp(const Params& params, const FmapDesc& fmapDesc,
                           const std::vector<WaveOp*>& prevWaveOps)
    : WaveOp(params, fmapDesc, prevWaveOps)
{}

}}

