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
