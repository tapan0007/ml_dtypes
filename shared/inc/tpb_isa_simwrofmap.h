#ifndef TPB_ISA_SIMWROFMAP_H
#define TPB_ISA_SIMWROFMAP_H

#include "tpb_isa.h"


struct SIM_WROFMAP {
    TPB_CMD_HEADER  hdr;
    char            fname[128] = {0};    
    uint32_t        address    = {0};       
    uint16_t        dims[4]    = {0};           
    uint8_t         dtype      = {0};
    SIM_WROFMAP() : hdr(SIM_WROFMAP_OPC, sizeof(*this)) {}
} TONGA_PACKED;


#endif
