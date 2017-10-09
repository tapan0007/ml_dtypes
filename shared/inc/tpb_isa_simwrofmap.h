#ifndef TPB_ISA_SIMWROFMAP_H
#define TPB_ISA_SIMWROFMAP_H

#include "tpb_isa.h"


struct SIM_WROFMAP {
    uint8_t         opcode;        
    char            fname[128];    
    uint32_t        address;       
    uint16_t        i_n;           
    uint16_t        w_c;           
    uint16_t        w_m;           
    uint16_t        o_rows;        
    uint16_t        o_cols;        
    uint8_t         word_size;     
} TONGA_PACKED;


#endif
