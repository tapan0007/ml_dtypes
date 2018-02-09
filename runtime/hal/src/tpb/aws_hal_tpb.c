/*
 * Copyright 2018, Amazon.com, Inc. or its affiliates. All Rights Reserved
 */

/**
 * Tensor Processing Block (TPB)
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

#include "aws_hal_tpb.h"

/*
 * TPB init:
 * =============
 */
int aws_hal_tpb_init (void* tpb_mem_handle)
{
    int ret = 0;
    ret += aws_hal_tpb_pe_init (tpb_mem_handle);
    ret += aws_hal_tpb_pool_init (tpb_mem_handle);
    //TODO ret += aws_hal_tpb_act_init (tpb_mem_handle);

    return ret;
}


