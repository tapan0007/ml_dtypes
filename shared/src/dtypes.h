#ifndef _TYPES_H
#define _TYPES_H

#include "stdint.h"
#include "stdlib.h"
#include "assert.h"

enum ARBPRECTYPE {INVALID_ARBPRECTYPE = 0, 
    INT8 = 2,   UINT8 = 3, 
    INT16 = 4,  UINT16 = 5, 
    FP16 = 7,
    INT32,      UINT32,
    FP32, 
    INT64 = 12, UINT64 = 13, 
    NUM_ARBPRECTYPE = 16};

inline
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

inline
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


enum ACTIVATIONFUNCTION {INVALID_ACTIVATIONFUNCTION=-1, IDENTITY, RELU, LEAKY_RELU, SIGMIOD, TANH,  NUM_ACTIVATIONFUNCTION};


enum POOLFUNC {
    MAX_POOL = 0, 
    AVG_POOL = 1, 
    IDENTITY_POOL = 2,
    NUM_POOLTYPE=3};

typedef uint64_t addr_t;
#define MAX_ADDR UINTMAX_MAX


#endif
