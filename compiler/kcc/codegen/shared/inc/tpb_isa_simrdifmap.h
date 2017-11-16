#ifndef TPB_ISA_SIMRDIFMAP_H
#define TPB_ISA_SIMRDIFMAP_H

#include "tpb_isa.h"


struct SIM_RDIFMAP {
    TPB_CMD_HEADER  hdr;
    uint32_t        address    = {0};       
    char            fname[128] = {0};    
    SIM_RDIFMAP() : hdr(SIM_RDIFMAP_OPC, sizeof(*this)) {};
} TONGA_PACKED;



#endif
