#ifndef TPB_ISA_LDWEIGHTS_H
#define TPB_ISA_LDWEIGHTS_H

#include "tpb_isa.h"


struct LDWEIGHTS {
    uint8_t         opcode;        
    uint32_t        address;       
    uint8_t         dtype;         
    int16_t         x_step;        // 1's complement, granularity of dtype
    uint8_t         x_num;         
    int16_t         y_step;        
    uint8_t         y_num;         
    uint8_t         last_row;      // id of last partition to read from
} TONGA_PACKED;



#endif
