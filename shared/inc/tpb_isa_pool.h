#ifndef TPB_ISA_POOL_H
#define TPB_ISA_POOL_H

#include "tpb_isa.h"

enum POOLFUNC {
    MAX_POOL = 0,
    AVG_POOL = 1,
    IDENTITY_POOL = 2,
    NUM_POOLTYPE  = 3
};


struct POOL {
    TPB_CMD_HEADER               hdr;
    struct TPB_CMD_SYNCH         synch;
    struct TPB_CMD_DEQUANT       dquant;
    uint8_t         pool_func = {IDENTITY_POOL};     
    uint8_t         in_dtype  = {INVALID_ARBPRECTYPE};      
    uint8_t         out_dtype = {INVALID_ARBPRECTYPE};     
    uint32_t        src_start_addr = {0};   
    /* src* describes ONE pooling src */  
    int16_t         src_x_step = {0};    
    uint16_t        src_x_num  = {0};     
    int16_t         src_y_step = {0};    
    uint16_t        src_y_num  = {0};     
    uint32_t        dst_start_addr = {0};   
    /* dst* describes ONE pooling result */ 
    int16_t         dst_x_step = {0};    
    uint16_t        dst_x_num  = {0};     
    int16_t         dst_y_step = {0};    
    uint16_t        dst_y_num  = {0};     
    /* str* describes interpooling strides */
    int16_t         str_x_step = {0};    
    uint16_t        str_x_num  = {0};
    int16_t         str_y_step = {0};    
    uint16_t        str_y_num  = {0};
    uint8_t         num_partitions = {0};
    POOL() : hdr(POOL_OPC, sizeof(*this)) {}
} TONGA_PACKED;

#endif


