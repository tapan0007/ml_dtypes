#ifndef TPB_ISA_SIMRDIFMAP_H
#define TPB_ISA_SIMRDIFMAP_H

#include "tpb_isa.h"


struct SIM_RDIFMAP {
    uint8_t         opcode;        
    uint32_t        address;       
    char            fname[128];    
} TONGA_PACKED;



#endif
