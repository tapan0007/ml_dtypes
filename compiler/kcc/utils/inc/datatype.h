#ifndef KCC_UTILS_DATATYPE_H
#define KCC_UTILS_DATATYPE_H

#include "types.h"

namespace kcc {
namespace utils {


//########################################################
class DataType {
public:
    virtual int64 gSizeInBytes() const = 0;

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

    int64 gSizeInBytes() const
    {
        return sizeof(data_type);
    }

    const char* gName() const
    {
        return "int8";
    }

    const char* gTccName() const
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

    int64 gSizeInBytes() const
    {
        return sizeof(data_type);
    }

    const char* gName() const
    {
        return "int16";
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

    int64 gSizeInBytes() const
    {
        return 2;
    }

    const char* gName() const
    {
        return "float16";
    }

    const char* gTccName() const
    {
        return "FP16";
    }
};

} // namespace utils
} // namespace kcc

#endif

