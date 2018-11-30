#include <sstream>



#include "utils/inc/datatype.hpp"

#include "arch/inc/arch.hpp"
#include "nets/inc/network.hpp"
#include "wave/inc/poolengwaveop.hpp"


#define RETURN_ASSERT(x)  assert(x); return (x)

namespace kcc {
namespace wave {

PoolEngWaveOp::PoolEngWaveOp(const PoolEngWaveOp::Params& params,
                             const std::vector<WaveOp*>& prevWaveOps)
    : BaseClass(params, prevWaveOps)
    , m_OutDtype(DataType::dataTypeId2DataType(params.m_OutDtypeId))
    , m_NumPartitions(params.m_NumPartitions)
{
    Assert(params.m_OutDtypeId != DataTypeId::None, "None out data type");
    Assert(params.m_NumPartitions >= 1, "Num partitions is ", params.m_NumPartitions, ". Must be >0");
}


bool
PoolEngWaveOp::verify() const
{
    if (! this->BaseClass::verify()) {
        RETURN_ASSERT(false);
    }
    if (m_NumPartitions < 1) {
        RETURN_ASSERT(false);
    }
    return true;
}

}}

