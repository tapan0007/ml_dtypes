/*
 * Copyright 2018, Amazon.com, Inc. or its affiliates. All Rights Reserved
 */
#ifndef __AWS_HAL_TPB_COMMON_MMAP_HPP__
#define __AWS_HAL_TPB_COMMON_MMAP_HPP__
/**
 * Tensor Processing Block (TPB) / Common Data Structures.
 * This file defines the common structures reused accross TPB:
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

/*
 * TPB Memory Map:
 * ==============
 */
#define  TPB_MMAP_STATE_BUF_BASE                     0x00000000
#define  TPB_MMAP_STATE_BUF_SIZE                     0x01000000
#define  TPB_MMAP_PSUM_BUF_BASE                      0x02000000
#define  TPB_MMAP_PSUM_BUF_SIZE                      0x00100000
#define  TPB_MMAP_PSUM_BUF_BANK0_BASE                0x02000000
#define  TPB_MMAP_PSUM_BUF_BANK1_BASE                0x02040000
#define  TPB_MMAP_PSUM_BUF_BANK2_BASE                0x02080000
#define  TPB_MMAP_PSUM_BUF_BANK3_BASE                0x020C0000
#define  TPB_MMAP_PSUM_BUF_BANK_SIZE                 0x00040000
#define  TPB_MMAP_ACTIVATION_BASE                    0x02400000
#define  TPB_MMAP_ACTIVATION_SIZE                    0x00100000
#define  TPB_MMAP_ACTIVATION_INST_BUF_BASE           0x02400000
#define  TPB_MMAP_ACTIVATION_INST_BUF_SIZE           0x00004000
#define  TPB_MMAP_ACTIVATION_INST_BUF_TAIL_PTR       0x02404000
#define  TPB_MMAP_ACTIVATION_PROFILE_TABLE_BASE      0x02420000
#define  TPB_MMAP_ACTIVATION_PROFILE_TABLE_SIZE      0x00002000
#define  TPB_MMAP_ACTIVATION_PROFILE_CAM_BASE        0x02440000
#define  TPB_MMAP_ACTIVATION_PROFILE_CAM_SIZE        0x00000400
#define  TPB_MMAP_ACTIVATION_PWL_TABLE_BASE          0x02460000
#define  TPB_MMAP_ACTIVATION_PWL_TABLE_SIZE          0x00004000
#define  TPB_MMAP_ACTIVATION_PWL_CONTROL_TABLE_BASE  0x02480000
#define  TPB_MMAP_ACTIVATION_PWL_CONTROL_TABLE_SIZE  0x00004000
#define  TPB_MMAP_POOLING_BASE                       0x02500000
#define  TPB_MMAP_POOLING_SIZE                       0x00100000
#define  TPB_MMAP_POOLING_INST_BUF_BASE              0x02500000
#define  TPB_MMAP_POOLING_INST_BUF_SIZE              0x00004000
#define  TPB_MMAP_POOLING_INST_BUF_TAIL_PTR          0x02504000
#define  TPB_MMAP_POOLING_PROFILE_TABLE_BASE         0x02520000
#define  TPB_MMAP_POOLING_PROFILE_TABLE_SIZE         0x00002000
#define  TPB_MMAP_POOLING_PROFILE_CAM_BASE           0x02540000
#define  TPB_MMAP_POOLING_PROFILE_CAM_SIZE           0x00000400
#define  TPB_MMAP_PE_ARRAY_BASE                      0x02600000
#define  TPB_MMAP_PE_ARRAY_SIZE                      0x00100000
#define  TPB_MMAP_PE_ARRAY_INST_BUF_BASE             0x02600000
#define  TPB_MMAP_PE_ARRAY_INST_BUF_SIZE             0x00004000
#define  TPB_MMAP_PE_ARRAY_INST_BUF_TAIL_PTR         0x02604000
#define  TPB_MMAP_PE_ARRAY_PROFILE_TABLE_BASE        0x02620000
#define  TPB_MMAP_PE_ARRAY_PROFILE_TABLE_SIZE        0x00002000
#define  TPB_MMAP_PE_ARRAY_PROFILE_CAM_BASE          0x02640000
#define  TPB_MMAP_PE_ARRAY_PROFILE_CAM_SIZE          0x00000400
#define  TPB_MMAP_EVENTS_BASE                        0x02700000
#define  TPB_MMAP_EVENTS_SIZE                        0x00100000

#endif
