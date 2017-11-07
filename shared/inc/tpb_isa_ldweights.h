#ifndef TPB_ISA_LDWEIGHTS_H
#define TPB_ISA_LDWEIGHTS_H

#include "tpb_isa.h"

struct LDWEIGHTS 
{
    public:
        TPB_CMD_HEADER<LDWEIGHTS>    hdr;
        struct TPB_CMD_SYNCH         synch;
        struct TPB_CMD_DEQUANT       dquant;
        uint32_t        start_addr = {0};       
        int16_t         x_step = {0};      // 1's complement, granularity of dtype
        uint8_t         x_num  = {0};         
        uint8_t         num_row_partitions = {0}; 
        LDWEIGHTS() : hdr(LDWEIGHTS_OPC) {}
} TONGA_PACKED;




#endif
