#include <sstream>



#include "utils/inc/datatype.hpp"

#include "arch/inc/arch.hpp"
#include "arch/inc/poolingeng.hpp"

#include "nets/inc/network.hpp"

#include "wave/inc/waveconsts.hpp"
#include "wave/inc/regshufflewaveop.hpp"


// #define RETURN_ASSERT(x) return (x)
#define RETURN_ASSERT(x)  assert(x); return (x)


namespace kcc {
namespace wave {

RegShuffleWaveOp::RegShuffleWaveOp(const RegShuffleWaveOp::Params& params,
                       const std::vector<WaveOp*>& prevWaveOps)
    : BaseClass(params, prevWaveOps)
    , m_StartReg(params.m_StartReg)
    , m_InSel(params.m_InSel)
{
    assert(verify());
}

bool
RegShuffleWaveOp::verify() const
{
    if (! this->BaseClass::verify()) {
        RETURN_ASSERT(false);
    }

    const arch::PoolingEng& poolEng(arch::Arch::gArch().gPoolingEng());


    if (m_StartReg > poolEng.gNumChannels()) {
        RETURN_ASSERT(false);
    }
    if (m_StartReg % poolEng.gNumChannels() != 0) {
        RETURN_ASSERT(false);
    }

    for (auto k : m_InSel) {
        if (k < 0 || k >  poolEng.gNumChannels()) {
            RETURN_ASSERT(false);
        }
    }


    return true;
}


std::string
RegShuffleWaveOp::gTypeStrStatic()
{
    return WaveOpTypeStr::RegShuffle;
}



bool
RegShuffleWaveOp::Params::verify() const
{
    RegShuffleWaveOp::BaseClass::Params::verify();
    return true;
}

}}

