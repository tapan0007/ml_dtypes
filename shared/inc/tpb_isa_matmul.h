#ifndef TPB_ISA_MATMUL_H
#define TPB_ISA_MATMUL_H

#include "tpb_isa.h"


struct MATMUL {
    uint8_t         opcode;        
    uint8_t         dequant_table_idx;   
    uint8_t         quant_data_size;   
    uint8_t         dequant_data_size;   
    uint8_t         start_tensor_calc;   
    uint8_t         stop_tensor_calc;   
    uint8_t         reserved;      
    uint8_t         dtype;         
    uint32_t        fmap_start_addr;   
    int16_t         fmap_x_step;   // 1's complement, granularity of dtype
    uint8_t         fmap_x_num;    
    int16_t         fmap_y_step;   
    uint8_t         fmap_y_num;    
    uint8_t         num_row_partitions;      
    uint8_t         n_pad;         
    uint8_t         w_pad;         
    uint8_t         e_pad;         
    uint8_t         s_pad;         
    uint32_t        psum_start_addr;   
    uint8_t         num_column_partitions;      
    int8_t          psum_step;     
    /* use most recently loaded weights */
    uint8_t         toggle_weight;
    uint8_t         event_func;    
    uint8_t         event_id;      
} TONGA_PACKED;



#endif
