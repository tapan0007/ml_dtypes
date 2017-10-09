#ifndef TPB_ISA_SIMRDFILTER_H
#define TPB_ISA_SIMRDFILTER_H

#include "tpb_isa.h"

struct SIM_RDFILTER {
    uint8_t         opcode;        
    uint32_t        address;       
    char            fname[128];    
} TONGA_PACKED;



#endif
