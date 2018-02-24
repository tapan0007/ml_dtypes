#pragma once

#ifndef KCC_UTILS_DATATYPE_H
#define KCC_UTILS_DATATYPE_H

#include <cereal/archives/json.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/map.hpp>

#include "tpb_isa.hpp"

#include "utils/inc/types.hpp"



namespace kcc {
namespace utils {

enum DataTypeId {
    DataTypeId_None,
    DataTypeId_Uint8,
    DataTypeId_Uint16,
    DataTypeId_Float16,
    DataTypeId_Float32,
    DataTypeId_Int32,
    DataTypeId_Int64,
};

//########################################################
class DataType {
public:
    virtual ~DataType()
    {}

    static const DataType& dataTypeId2DataType(DataTypeId typeId);
    static DataTypeId dataTypeStr2Id(const std::string& dataTypeStr);
    static const std::string& dataTypeId2Str(DataTypeId typeId);

    virtual kcc_int64 gSizeInBytes() const = 0;

    virtual DataTypeId  gDataTypeId() const = 0;
    virtual ARBPRECTYPE gSimTypeId() const = 0;

    virtual const char* gName() const = 0;

    virtual const char* gTccName() const = 0;
};


//########################################################
class DataTypeUint8 : public DataType {
private:
    using data_type = kcc_uint8;
public:
    DataTypeUint8()
        : DataType()
    {
    }

    DataTypeId  gDataTypeId() const override;
    ARBPRECTYPE gSimTypeId() const override;

    kcc_int64 gSizeInBytes() const override
    {
        static_assert(sizeof(data_type)==1, "sizeof(uint8) != 1");
        return sizeof(data_type);
    }

    static const char* gNameStatic()
    {
        return "uint8";
    }
    const char* gName() const override
    {
        return gNameStatic();
    }

    const char* gTccName() const override
    {
        return "UINT8";
    }
};

//########################################################
class DataTypeUint16 : public DataType {
private:
    using data_type = kcc_uint16;
public:
    DataTypeUint16()
        : DataType()
    {
    }

    DataTypeId  gDataTypeId() const override;
    ARBPRECTYPE gSimTypeId() const override;

    kcc_int64 gSizeInBytes() const override
    {
        static_assert(sizeof(data_type)==2, "sizeof(uint16) != 2");
        return sizeof(data_type);
    }

    static const char* gNameStatic()
    {
        return "uint16";
    }

    const char* gName() const override
    {
        return gNameStatic();
    }

    const char* gTccName() const override
    {
        return "INT16";
    }
};

//########################################################
class DataTypeFloat16 : public DataType {
public:
    DataTypeFloat16()
        : DataType()
    {
    }

    DataTypeId  gDataTypeId() const override;
    ARBPRECTYPE gSimTypeId() const override;

    kcc_int64 gSizeInBytes() const override
    {
        return 2; // No float16 in C++
    }

    static const char* gNameStatic()
    {
        return "float16";
    }

    const char* gName() const override
    {
        return gNameStatic();
    }

    const char* gTccName() const override
    {
        return "FP16";
    }
};


//########################################################
class DataTypeFloat32 : public DataType {
private:
    using data_type = kcc_float32;
public:
    DataTypeFloat32()
        : DataType()
    {
    }

    DataTypeId  gDataTypeId() const override;
    ARBPRECTYPE gSimTypeId() const override;

    kcc_int64 gSizeInBytes() const override
    {
        static_assert(sizeof(data_type)==4, "sizeof(float32) != 4");
        return sizeof(float);
    }

    static const char* gNameStatic()
    {
        return "float32";
    }

    const char* gName() const override
    {
        return gNameStatic();
    }

    const char* gTccName() const override
    {
        return "FP32";
    }
};


//########################################################
class DataTypeInt32 : public DataType {
private:
    using data_type = kcc_int32;
public:
    DataTypeInt32()
        : DataType()
    {
    }

    DataTypeId  gDataTypeId() const override;
    ARBPRECTYPE gSimTypeId() const override;

    kcc_int64 gSizeInBytes() const override
    {
        static_assert(sizeof(data_type)==4, "sizeof(int32) != 4");
        return sizeof(data_type);
    }

    static const char* gNameStatic()
    {
        return "int32";
    }

    const char* gName() const override
    {
        return gNameStatic();
    }

    const char* gTccName() const override
    {
        assert(false && "Cannot use int32 as TCC dtype");
        return nullptr;
    }
};

//########################################################
class DataTypeInt64 : public DataType {
private:
    using data_type = kcc_int32;
public:
    DataTypeInt64()
        : DataType()
    {
    }

    DataTypeId  gDataTypeId() const override;
    ARBPRECTYPE gSimTypeId() const override;

    kcc_int64 gSizeInBytes() const override
    {
        static_assert(sizeof(data_type)==4, "sizeof(int32) != 4");
        return sizeof(data_type);
    }

    static const char* gNameStatic()
    {
        return "int32";
    }

    const char* gName() const override
    {
        return gNameStatic();
    }

    const char* gTccName() const override
    {
        assert(false && "Cannot use int32 as TCC dtype");
        return nullptr;
    }
};

} // namespace utils
} // namespace kcc

#endif

