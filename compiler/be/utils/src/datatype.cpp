
#include "utils/inc/datatype.hpp"


namespace kcc {
namespace utils {

DataTypeId
DataTypeUint8::gDataTypeId () const {
    return DataTypeId_Uint8;
}

ARBPRECTYPE
DataTypeUint8::gSimTypeId() const
{
    return ARBPRECTYPE::UINT8;
}

DataTypeId
DataTypeUint16::gDataTypeId () const {
    return DataTypeId_Uint16;
}

ARBPRECTYPE
DataTypeUint16::gSimTypeId() const
{
    return ARBPRECTYPE::UINT16;
}

DataTypeId
DataTypeFloat16::gDataTypeId () const {
    return DataTypeId_Float16;
}

ARBPRECTYPE
DataTypeFloat16::gSimTypeId() const
{
    return ARBPRECTYPE::FP16;
}

DataTypeId
DataTypeFloat32::gDataTypeId () const {
    return DataTypeId_Float32;
}

ARBPRECTYPE
DataTypeFloat32::gSimTypeId() const
{
    return ARBPRECTYPE::FP32;
}

DataTypeId
DataTypeInt32::gDataTypeId () const {
    return DataTypeId_Int32;
}

ARBPRECTYPE
DataTypeInt32::gSimTypeId() const
{
    return ARBPRECTYPE::INT32;
}

DataTypeId
DataTypeInt64::gDataTypeId () const {
    return DataTypeId_Int64;
}

ARBPRECTYPE
DataTypeInt64::gSimTypeId() const
{
    return ARBPRECTYPE::INT64;
}



const DataType&
DataType::dataTypeId2DataType(DataTypeId typeId)
{
    static const DataTypeUint8      typeUint8;
    static const DataTypeUint16     typeUint16;
    static const DataTypeFloat16    typeFloat16;
    static const DataTypeFloat32    typeFloat32;
    static const DataTypeInt32      typeInt32;
    static const DataTypeInt64      typeInt64;

    switch (typeId) {
    case DataTypeId_Uint8:
        return typeUint8;
        break;
    case DataTypeId_Uint16:
        return typeUint16;
        break;
    case DataTypeId_Float16:
        return typeFloat16;
        break;
    case DataTypeId_Float32:
        return typeFloat32;
        break;
    case DataTypeId_Int32:
        return typeInt32;
        break;
    case DataTypeId_Int64:
        return typeInt64;
        break;
    default:
        assert(false && "Wrong DataTypeId");
        break;
    }
    assert(false && "Wrong DataTypeId");
    return typeFloat16;
}

DataTypeId
DataType::dataTypeStr2Id(const std::string& dataTypeStr)
{
    if (dataTypeStr == DataTypeUint8::gNameStatic()) {
        return DataTypeId_Uint8;
    } else if (dataTypeStr == DataTypeUint16::gNameStatic()) {
        return DataTypeId_Uint16;
    } else if (dataTypeStr == DataTypeFloat16::gNameStatic()) {
        return DataTypeId_Float16;
    } else if (dataTypeStr == DataTypeFloat32::gNameStatic()) {
        return DataTypeId_Float32;
    } else if (dataTypeStr == DataTypeInt32::gNameStatic()) {
        return DataTypeId_Int32;
    } else if (dataTypeStr == DataTypeInt64::gNameStatic()) {
        return DataTypeId_Int64;
    } else {
        assert(false && "Wrong DataTypeId");
    }
    assert(false && "Wrong DataTypeId");
    return DataTypeId_Float32;
}

const
std::string&
DataType::dataTypeId2Str(DataTypeId typeId)
{
    static const std::string    uint8Str(DataTypeUint8::gNameStatic());
    static const std::string    uint16Str(DataTypeUint16::gNameStatic());
    static const std::string    float16Str(DataTypeFloat16::gNameStatic());
    static const std::string    float32Str(DataTypeFloat32::gNameStatic());
    static const std::string    int32Str(DataTypeInt32::gNameStatic());
    static const std::string    int64Str(DataTypeInt64::gNameStatic());

    switch (typeId) {
    case DataTypeId_Uint8:
        return uint8Str;
        break;
    case DataTypeId_Uint16:
        return uint16Str;
        break;
    case DataTypeId_Float16:
        return float16Str;
        break;
    case DataTypeId_Float32:
        return float32Str;
        break;
    case DataTypeId_Int32:
        return int32Str;
        break;
    case DataTypeId_Int64:
        return int64Str;
        break;
    default:
        assert(false && "Wrong DataTypeId");
        break;
    }
    assert(false && "Wrong DataTypeId");
    return float32Str;
}

}}

