#ifndef TPB_ISA_H
#define TPB_ISA_H

#include <stdlib.h>
#include <stdint.h>
#include "isa_common.h"
#include "tpb_isa_ldweights.h"
#include "tpb_isa_matmul.h"
#include "tpb_isa_pool.h"
#include "tpb_isa_simrdifmap.h"
#include "tpb_isa_simrdfilter.h"
#include "tpb_isa_simwrofmap.h"

#define TPB_OPCODE(x) BITFIELD_EXTRACT(x, 0, 8)

#define INSTRUCTION_NBYTES 256

typedef uint32_t addr_t;

enum TPB_CMD_TYPE {
    LDWEIGHTS_OPC  = 0x00,
    MATMUL_OPC  = 0x01,
    POOL_OPC    = 0x80,
    SIM_WROFMAP_OPC = 0xFC,
    SIM_RDFILTER_OPC = 0xFD,
    SIM_RDIFMAP_OPC = 0xFE,
};

/* todo: move out to activation isa defintion*/
enum ACTIVATIONFUNCTION {
    INVALID_ACTIVATIONFUNCTION=0x00, 
    IDENTITY, 
    RELU, 
    LEAKY_RELU, 
    SIGMIOD, 
    TANH, 
    NUM_ACTIVATIONFUNCTION
};


enum ARBPRECTYPE {
    INVALID_ARBPRECTYPE = 0,
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

#endif


