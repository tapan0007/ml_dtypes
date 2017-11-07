#ifndef TPB_ISA_MATMUL_H
#define TPB_ISA_MATMUL_H

#include "tpb_isa.h"


struct MATMUL {
    TPB_CMD_HEADER               hdr;
    struct TPB_CMD_SYNCH         synch;
    struct TPB_CMD_DEQUANT       dquant;
    /* clear psum */ 
    uint8_t         start_tensor_calc = {0}; 
    /* debug */ 
    uint8_t         stop_tensor_calc  = {0}; 
    //      dtype is in spec, but is it necessary w/ dquant?
    //        uint8_t         dtype;         
    uint32_t        fmap_start_addr   = {0xffffffff};   
    int16_t         fmap_x_step       = {0};  
    uint8_t         fmap_x_num        = {0};    
    int16_t         fmap_y_step       = {0};   
    uint8_t         fmap_y_num        = {0};    
    uint8_t         num_row_partitions = {0};      
    /* Verdict out?: should HW support padding? */
    uint8_t         n_pad = {0};
    uint8_t         w_pad = {0};         
    uint8_t         e_pad = {0};         
    uint8_t         s_pad = {0};         
    uint32_t        psum_start_addr = {0};   
    int8_t          psum_step = {0};     
    uint8_t         num_column_partitions = {0};      
    /* use Most-recently loaded weight */
    uint8_t         toggle_weight = {0}; 
    MATMUL() : hdr(MATMUL_OPC, sizeof(*this)) {}
} TONGA_PACKED;



#endif
