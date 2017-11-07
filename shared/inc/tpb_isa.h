#ifndef TPB_ISA_H
#define TPB_ISA_H

#include <stdlib.h>
#include <stdint.h>
#include "isa_common.h"

#define TPB_OPCODE(x) BITFIELD_EXTRACT(x, 1, 8)

#define INSTRUCTION_NBYTES 256

typedef uint32_t addr_t;

enum ARBPRECTYPE {
    INVALID_ARBPRECTYPE = 0,
    INT8 = 2,   UINT8 = 3,
    INT16 = 4,  UINT16 = 5,
    FP16 = 7,
    INT32,      UINT32,
    FP32,
    INT64 = 12, UINT64 = 13,
    NUM_ARBPRECTYPE = 16};


enum TPB_CMD_TYPE {
    LDWEIGHTS_OPC  = 0x00,
    MATMUL_OPC  = 0x01,
    POOL_OPC    = 0x02,
    SIM_WROFMAP_OPC = 0x7C,
    SIM_RDFILTER_OPC = 0x7D,
    SIM_RDIFMAP_OPC = 0x7E,
};

class TPB_CMD_HEADER {
    public:
        /* on little endian machine, bit field order is reversed from code-list order*/
        uint8_t         opcode  : 7;
        uint8_t         phase   : 1;
        uint8_t         inst_word_len = {0};
        void            set_phase(uint8_t ph) {
            phase=ph & 0x1;
        };
        TPB_CMD_HEADER(uint8_t _opcode, size_t szof) : 
            opcode(_opcode), phase(0), inst_word_len(szof) {}
} TONGA_PACKED;

struct TPB_CMD_SYNCH {
    uint8_t         wait_event_mode   : 4;
    uint8_t         set_event_mode    : 4;
    uint8_t         wait_event_id = {0};
    uint8_t         set_event_id  = {0};
    TPB_CMD_SYNCH() : wait_event_mode(0), set_event_mode(0) {}
} TONGA_PACKED;

struct TPB_CMD_DEQUANT {
    uint8_t         dequant_table_idx = {0xff};
    uint8_t         quant_data_size   = {0xff};
    uint8_t         dequant_data_type = {INVALID_ARBPRECTYPE};
} TONGA_PACKED;

#include "tpb_isa_ldweights.h"
#include "tpb_isa_matmul.h"
#include "tpb_isa_pool.h"
#include "tpb_isa_simrdifmap.h"
#include "tpb_isa_simrdfilter.h"
#include "tpb_isa_simwrofmap.h"

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
            return FP16;
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
        case FP16: /* FIXME HACK!! */ 
            size = 2;
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


