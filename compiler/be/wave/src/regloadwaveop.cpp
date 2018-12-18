#include <sstream>



#include "utils/inc/datatype.hpp"

#include "arch/inc/arch.hpp"
#include "nets/inc/network.hpp"
#include "wave/inc/waveconsts.hpp"
#include "wave/inc/regloadwaveop.hpp"


// #define RETURN_ASSERT(x) return (x)
#define RETURN_ASSERT(x)  assert(x); return (x)


namespace kcc {
namespace wave {

RegLoadWaveOp::RegLoadWaveOp(const RegLoadWaveOp::Params& params,
                       const std::vector<WaveOp*>& prevWaveOps)
    : BaseClass(params, prevWaveOps)
    , m_InDtype(DataType::dataTypeId2DataType(params.m_InDtypeId))
{
    m_SrcSbAddress      = params.m_SrcSbAddress;
    m_SrcStartAtMidPart = params.m_SrcStartAtMidPart;

    m_SrcXNum           = params.m_SrcXNum;
    m_SrcXStep          = params.m_SrcXStep;
    m_SrcYNum           = params.m_SrcYNum;
    m_SrcYStep          = params.m_SrcYStep;
    m_SrcZNum           = params.m_SrcZNum;
    m_SrcZStep          = params.m_SrcZStep;

    assert(verify());
}

bool
RegLoadWaveOp::verify() const
{
    //const arch::PsumBuffer& psumBuf(arch::Arch::gArch().gPsumBuffer());
    if (! this->BaseClass::verify()) {
        RETURN_ASSERT(false);
    }

    if (m_SrcSbAddress < 0) {
        RETURN_ASSERT(false);
    }

    if (m_SrcXNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcXStep == 0 && m_SrcXNum != 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcYNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcYStep == 0 && m_SrcYNum != 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcZNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcZStep == 0 && m_SrcZNum != 1) {
        RETURN_ASSERT(false);
    }

    return true;
}




bool
RegLoadWaveOp::Params::verify() const
{
    RegLoadWaveOp::BaseClass::Params::verify();
    return true;
}

std::string
RegLoadWaveOp::gTypeStrStatic()
{
    return WaveOpTypeStr::RegLoad;
}

}}

