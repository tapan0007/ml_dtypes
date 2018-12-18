#include <sstream>



#include "utils/inc/datatype.hpp"

#include "arch/inc/arch.hpp"
#include "nets/inc/network.hpp"
#include "wave/inc/waveconsts.hpp"
#include "wave/inc/regwaveop.hpp"


// #define RETURN_ASSERT(x) return (x)
#define RETURN_ASSERT(x)  assert(x); return (x)


namespace kcc {
namespace wave {

RegWaveOp::RegWaveOp(const RegWaveOp::Params& params,
                       const std::vector<WaveOp*>& prevWaveOps)
    : BaseClass(params, prevWaveOps)
{
    m_ParallelMode  = params.m_ParallelMode;
    m_NumPartitions = params.m_NumPartitions;

    assert(verify());
}

bool
RegWaveOp::verify() const
{
    //const arch::PsumBuffer& psumBuf(arch::Arch::gArch().gPsumBuffer());
    if (! this->BaseClass::verify()) {
        RETURN_ASSERT(false);
    }

    if (m_NumPartitions < 1) {
        RETURN_ASSERT(false);
    }
    return true;
}




bool
RegWaveOp::Params::verify() const
{
    RegWaveOp::BaseClass::Params::verify();
    return true;
}


}}

