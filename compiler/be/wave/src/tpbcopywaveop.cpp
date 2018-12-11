
#include <sstream>


#include "utils/inc/datatype.hpp"

#include "wave/inc/waveconsts.hpp"
#include "wave/inc/tpbcopywaveop.hpp"
#include "nets/inc/network.hpp"

#define ASSERT_RETURN(x) assert(x); return (x)


namespace kcc {
namespace wave {

TpbCopyWaveOp::TpbCopyWaveOp(
        const TpbCopyWaveOp::Params& params,
        const std::vector<WaveOp*>& prevWaveOps)
    : BaseClass(params, prevWaveOps)
    , m_PairLoadWaveOp(params.m_PairLoadWaveOp)
    , m_PrevCopyWaveOp(params.m_PrevCopyWaveOp)
    , m_SrcSbAddress(params.m_SrcSbAddress)
    , m_DstSbAddress(params.m_DstSbAddress)
    , m_SizeInBytes(params.m_SizeInBytes)
{
    assert(params.verify());
}

bool
TpbCopyWaveOp::verify() const
{
    if (! this->BaseClass::verify()) {
        ASSERT_RETURN(false);
    }
    if (! m_PairLoadWaveOp) {
        ASSERT_RETURN(false);
    }
    if (m_SizeInBytes <= 0) {
        ASSERT_RETURN(false);
    }
    return true;
} // TpbCopyWaveOp::verify





bool
TpbCopyWaveOp::Params::verify() const
{
    if (! this->TpbCopyWaveOp::BaseClass::Params::verify()) {
        ASSERT_RETURN(false);
    }
    // bool m_IfmapsReplicate
    return true;
}

std::string
TpbCopyWaveOp::gTypeStrStatic()
{
    return WaveOpTypeStr::TpbCopy;
}

}}


