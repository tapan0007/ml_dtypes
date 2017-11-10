#ifndef TPB_ISA_ACTIVATION_H
#define TPB_ISA_ACTIVATION_H

#include "tpb_isa.h"


/* todo: move out to activation isa defintion*/
enum ACTIVATIONFUNC {
    INVALID_ACTIVATIONFUNC=0x00,
    IDENTITY,
    RELU,
    LEAKY_RELU,
    SIGMOID,
    TANH,
    NUM_ACTIVATIONFUNC
};

struct ACTIVATION {
    TPB_CMD_HEADER           hdr;
    struct TPB_CMD_SYNCH     synch;
    uint8_t    in_data_type   = {INVALID_ARBPRECTYPE};      
    uint8_t    out_data_type  = {INVALID_ARBPRECTYPE};
    uint8_t    activation_func  = {INVALID_ACTIVATIONFUNC};
    uint32_t   src_start_addr = {0x0};
    uint16_t   src_x_step     = {0};
    uint8_t    src_x_num_elements = {0};
    uint16_t   src_y_step         = {0};
    uint8_t    src_y_num_elements = {0};
    uint32_t   dst_start_addr     = {0};
    uint16_t   dst_x_step         = {0};
    uint8_t    dst_x_num_elements = {0};
    uint16_t   dst_y_step         = {0};
    uint8_t    dst_y_num_elements = {0};
    uint8_t    num_partitions     = {0};
    ACTIVATION() : hdr(ACTIVATION_OPC, sizeof(*this)) {}
} TONGA_PACKED;

#endif


