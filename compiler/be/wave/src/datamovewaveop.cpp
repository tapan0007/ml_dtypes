#include <sstream>



#include "utils/inc/datatype.hpp"
#include "wave/inc/datamovewaveop.hpp"
#include "nets/inc/network.hpp"

//#define ASSERT_RETURN(x) return(x)
#define ASSERT_RETURN(x) assert(x); return (x)


namespace kcc {
namespace wave {

DataMoveWaveOp::DataMoveWaveOp(const DataMoveWaveOp::Params& params,
                           const std::vector<WaveOp*>& prevWaveOps)
    : BaseClass(params, prevWaveOps)
{
    assert(params.verify());
}


bool
DataMoveWaveOp::verify() const
{
    if (! this->BaseClass::verify()) {
        ASSERT_RETURN(false);
    }
    return true;
}



bool
DataMoveWaveOp::Params::verify() const
{
    if (! this->DataMoveWaveOp::BaseClass::Params::verify()) {
        return false;
    }
    return true;
}


}}

