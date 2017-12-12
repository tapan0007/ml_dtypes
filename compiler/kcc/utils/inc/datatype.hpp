#pragma once

#ifndef KCC_UTILS_DATATYPE_H
#define KCC_UTILS_DATATYPE_H

#include <cereal/archives/json.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/map.hpp>

#include "tpb_isa.hpp"

#include "types.hpp"



namespace kcc {
namespace utils {


//########################################################
class DataType {
public:
    virtual int64 gSizeInBytes() const = 0;

    virtual ARBPRECTYPE gTypeId() const = 0;

    virtual const char* gName() const = 0;

    virtual const char* gTccName() const = 0;
};

//########################################################
class DataTypeInt8 : public DataType {
private:
    typedef int8 data_type;
public:
    DataTypeInt8() {
    }

    ARBPRECTYPE gTypeId() const override;

    int64 gSizeInBytes() const override
    {
        return sizeof(data_type);
    }

    static const char* gNameStatic()
    {
        return "int8";
    }
    const char* gName() const override
    {
        return gNameStatic();
    }

    const char* gTccName() const override
    {
        return "INT8";
    }
};

//########################################################
class DataTypeInt16 : public DataType {
private:
    typedef int8 data_type;
public:
    DataTypeInt16() {
    }

    ARBPRECTYPE gTypeId() const override;

    int64 gSizeInBytes() const
    {
        return sizeof(data_type);
    }

    static const char* gNameStatic()
    {
        return "int16";
    }

    const char* gName() const
    {
        return gNameStatic();
    }

    const char* gTccName() const
    {
        return "INT16";
    }
};

//########################################################
class DataTypeFloat16 : public DataType {
public:
    DataTypeFloat16() {
    }

    ARBPRECTYPE gTypeId() const override;

    int64 gSizeInBytes() const
    {
        return 2;
    }

    static const char* gNameStatic()
    {
        return "float16";
    }

    const char* gName() const
    {
        return gNameStatic();
    }

    const char* gTccName() const
    {
        return "FP16";
    }
};

} // namespace utils
} // namespace kcc

#endif

