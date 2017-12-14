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
    virtual ~DataType()
    {}

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
        static_assert(sizeof(data_type)==1, "sizeof(float) != 1");
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
    typedef short data_type;
public:
    DataTypeInt16() {
    }

    ARBPRECTYPE gTypeId() const override;

    int64 gSizeInBytes() const override
    {
        static_assert(sizeof(data_type)==2, "sizeof(int16) != 2");
        return sizeof(data_type);
    }

    static const char* gNameStatic()
    {
        return "int16";
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
    DataTypeFloat16() {
    }

    ARBPRECTYPE gTypeId() const override;

    int64 gSizeInBytes() const override
    {
        return 2;
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
    typedef float data_type;
public:
    DataTypeFloat32() {
    }

    ARBPRECTYPE gTypeId() const override;

    int64 gSizeInBytes() const override
    {
        static_assert(sizeof(data_type)==4, "sizeof(float) != 4");
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


} // namespace utils
} // namespace kcc

#endif

