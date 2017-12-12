
#include "datatype.hpp"


namespace kcc {
namespace utils {

ARBPRECTYPE
DataTypeInt8::gTypeId() const 
{
    return ARBPRECTYPE::INT8;
}

ARBPRECTYPE
DataTypeInt16::gTypeId() const 
{
    return ARBPRECTYPE::INT16;
}

ARBPRECTYPE
DataTypeFloat16::gTypeId() const 
{
    return ARBPRECTYPE::FP16;
}


}}

