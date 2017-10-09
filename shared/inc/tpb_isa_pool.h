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
    uint8_t         opcode;        
    uint8_t         pool_func;     
    uint8_t         in_dtype;      
    uint8_t         out_dtype;     
    uint32_t        src_start_addr;   
    int16_t         src_w_step;    // 1's complement, granularity of dtype
    uint16_t        src_w_num;     
    int16_t         src_x_step;    
    uint16_t        src_x_num;     
    int16_t         src_y_step;    
    uint16_t        src_y_num;     
    int16_t         src_z_step;    
    uint16_t        src_z_num;     
    uint32_t        dst_start_addr;   
    int16_t         dst_w_step;    
    uint16_t        dst_w_num;     
    int16_t         dst_x_step;    
    uint16_t        dst_x_num;     
    int16_t         dst_y_step;    
    uint16_t        dst_y_num;     
    int16_t         dst_z_step;    
    uint16_t        dst_z_num;     
    uint8_t         event_func;    
    uint8_t         event_id;      
} TONGA_PACKED;



#endif
