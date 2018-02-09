/*
 * Copyright 2018, Amazon.com, Inc. or its affiliates. All Rights Reserved
 */
#ifndef __AWS_HAL_TPB_PE_HPP__
#define __AWS_HAL_TPB_PE_HPP__
/**
 * Tensor Processing Block (TPB) / PE-ARRAY
 *
 *  +-------+       +--------------------------------------+
 *  |       |  (W)  | PE | PE | PE |                       |
 *  |       |------>+----+----+----+                       |
 *  |       |       | PE | PE |                            |
 *  |   S   |       |----+----+     PE-ARRAY               |
 *  |   T   |  (X)  | PE |                                 |
 *  |   A   |------>|----+                                 |
 *  |   T   |       |                                      |
 *  |   E   |       +--------------------------------------+
 *  |       |          |   |   |   |   |    . . .    |   |
 *  |   B   |       +--------------------------------------+
 *  |   U   |       |            PSUM BUFFER               |
 *  |   F   |       +--------------------------------------+
 *  |   F   |                      |          |  
 *  |   E   |    +--------------+  |          |  
 *  |   R   |<---|  ACTIVATION  |<-+          |  
 *  |       |    +--------------+             |  
 *  |       |    +--------------+             |  
 *  |       |<---|   POOLING    |<------------+  
 *  +-------+    +--------------+                
 *
 */


#include "aws_hal_tpb_common.h"

#define AWS_HAL_TPB_PE_PE_PROFILE_SIZE 64

enum aws_hal_tpb_pe_profile_id {
    TPB_PE_PROFILE_ID_MAT_MUL = 0,
    TPB_PE_PROFILE_ID_WEIGHT_LOAD = 1,
    TPB_PE_PROFILE_ID_NOP = 2,
    TPB_PE_PROFILE_ID_SET_EVENT = 3,
    TPB_PE_PROFILE_ID_CLEAR_EVENT = 4,
    TPB_PE_PROFILE_ID_WAIT_EVENT = 5,
    TPB_PE_PROFILE_ID_WRITE = 6,
    // 7-31: Reserved
    TPB_PE_NUM_PROFILES
};

/*
 * PE-Array Profile Table Entry:
 * ============================
 */
struct aws_hal_tpb_pe_profile_table_params {
    // Common fields
    struct aws_hal_tpb_common_profile_table_params common_params;
};

/*
 * PE-Array init:
 * =============
 */
int aws_hal_tpb_pe_init (void* tpb_mem_handle);


#endif

