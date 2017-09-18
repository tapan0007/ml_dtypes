#include "types.h"
#include <assert.h>



ARBPRECTYPE 
get_upcast(ARBPRECTYPE type) {
    switch (type) {
        case UINT8:
            return UINT32;
        case INT8:
            return INT32;
        case UINT16:
            return UINT32;
        case INT16:
            return INT32;
        case FP16:
            return FP32;
        default:
            assert(0);
    }
    assert(0);
    return INVALID_ARBPRECTYPE;
}

size_t
sizeofArbPrecType(ARBPRECTYPE type)
{
    size_t size;
    switch (type) {
        case UINT8:
            size = sizeof(uint8_t);
            break;
        case UINT32:
            size = sizeof(uint32_t);
            break;
        case INT32:
            size = sizeof(int32_t);
            break;
        case UINT64:
            size = sizeof(uint64_t);
            break;
        case INT64:
            size = sizeof(int64_t);
            break;
        case FP32:
            size = sizeof(float);
            break;
        default:
            assert(0);
    }
    return size;
}


