#include <sstream>



#include "utils/inc/datatype.hpp"

#include "arch/inc/arch.hpp"
#include "layers/inc/layer.hpp"
#include "nets/inc/network.hpp"
#include "wave/inc/poolengwaveop.hpp"



namespace kcc {
namespace wave {

PoolEngWaveOp::PoolEngWaveOp(const PoolEngWaveOp::Params& params,
                             const std::vector<WaveOp*>& prevWaveOps)
    : WaveOp(params, prevWaveOps)
    , m_OutDtype(DataType::dataTypeId2DataType(params.m_OutDtypeId))
{
    Assert(params.m_OutDtypeId != DataTypeId::None, "None out data type");
}


}}

