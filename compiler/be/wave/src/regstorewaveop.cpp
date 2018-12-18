#include <sstream>



#include "utils/inc/datatype.hpp"

#include "arch/inc/arch.hpp"
#include "nets/inc/network.hpp"
#include "wave/inc/waveconsts.hpp"
#include "wave/inc/regstorewaveop.hpp"


// #define RETURN_ASSERT(x) return (x)
#define RETURN_ASSERT(x)  assert(x); return (x)


namespace kcc {
namespace wave {

RegStoreWaveOp::RegStoreWaveOp(const RegStoreWaveOp::Params& params,
                       const std::vector<WaveOp*>& prevWaveOps)
    : BaseClass(params, prevWaveOps)
    , m_OutDtype(DataType::dataTypeId2DataType(params.m_OutDtypeId))
{
    m_DstSbAddress      = params.m_DstSbAddress;
    m_DstStartAtMidPart = params.m_DstStartAtMidPart;

    m_DstXNum           = params.m_DstXNum;
    m_DstXStep          = params.m_DstXStep;
    m_DstYNum           = params.m_DstYNum;
    m_DstYStep          = params.m_DstYStep;
    m_DstZNum           = params.m_DstZNum;
    m_DstZStep          = params.m_DstZStep;

    assert(verify());
}

bool
RegStoreWaveOp::verify() const
{
    //const arch::PsumBuffer& psumBuf(arch::Arch::gArch().gPsumBuffer());
    if (! this->BaseClass::verify()) {
        RETURN_ASSERT(false);
    }

    if (m_DstSbAddress < 0) {
        RETURN_ASSERT(false);
    }

    if (m_DstXNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_DstXStep == 0 && m_DstXNum != 1) {
        RETURN_ASSERT(false);
    }
    if (m_DstYNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_DstYStep == 0 && m_DstYNum != 1) {
        RETURN_ASSERT(false);
    }
    if (m_DstZNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_DstZStep == 0 && m_DstZNum != 1) {
        RETURN_ASSERT(false);
    }

    return true;
}




bool
RegStoreWaveOp::Params::verify() const
{
    RegStoreWaveOp::BaseClass::Params::verify();
    return true;
}

std::string
RegStoreWaveOp::gTypeStrStatic()
{
    return WaveOpTypeStr::RegStore;
}

}}

