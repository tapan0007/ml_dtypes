#ifndef TPB_ISA_SIMRDFILTER_H
#define TPB_ISA_SIMRDFILTER_H

#include "tpb_isa.h"

struct SIM_RDFILTER {
    TPB_CMD_HEADER  hdr;
    uint32_t        address    = {0};
    char            fname[128] = {0};
    SIM_RDFILTER() : hdr(SIM_RDFILTER_OPC, sizeof(*this)) {}
} TONGA_PACKED;



#endif
