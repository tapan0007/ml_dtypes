#include "arb_prec.h"

size_t
sizeofArbPrecType(ARBPRECTYPE type)
{
    size_t size;
    switch (type) {
        case UINT8:
            size = sizeof(uint8_t);
            break;
        case R_UINT32:
            size = sizeof(uint32_t);
            break;
        case R_FP32:
            size = sizeof(float);
            break;
        default:
            assert(0);
    }
    return size;
}


