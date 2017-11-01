#ifndef TPB_ISA_SIMWROFMAP_H
#define TPB_ISA_SIMWROFMAP_H

#include "tpb_isa.h"


struct SIM_WROFMAP {
    uint8_t         opcode;        
    char            fname[128];    
    uint32_t        address;       
    uint16_t        dims[4];           
    uint8_t         dtype;
} TONGA_PACKED;


#endif
