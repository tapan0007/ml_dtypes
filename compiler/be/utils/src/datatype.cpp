
#include "datatype.hpp"


namespace kcc {
namespace utils {


ARBPRECTYPE
DataTypeUint8::gTypeId() const
{
    return ARBPRECTYPE::UINT8;
}

ARBPRECTYPE
DataTypeUint16::gTypeId() const
{
    return ARBPRECTYPE::UINT16;
}

ARBPRECTYPE
DataTypeFloat16::gTypeId() const
{
    return ARBPRECTYPE::FP16;
}

ARBPRECTYPE
DataTypeFloat32::gTypeId() const
{
    return ARBPRECTYPE::FP32;
}


}}

