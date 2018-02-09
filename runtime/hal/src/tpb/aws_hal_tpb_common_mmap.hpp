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
 * TODO (Ilya/Rich to comment) - consider moving to a central location.
 * TODO (Ilya/Rich to comment) - Consider formatting in another way (#defines maybe)
 */

// TODO - temporary. to be removed later (with some central definitions)
typedef char   uint8_t;
typedef int    uint32_t;
typedef float  uint64_t;
//


namespace TPB_MMAP {
    namespace STATE_BUF {
        uint64_t base = 0x00000000;
        uint64_t size = 0x01000000;
    };
    namespace PSUM_BUF {
        uint64_t base = 0x02000000;
        uint64_t size = 0x00100000;
        namespace BANK_0 {
            uint64_t base = 0x02000000; // TODO - this is ugly... need a way to make address incremental, so that a change to PSUM_BUF base doesn't influence all the banks
            uint64_t size = 0x00000800;
        }
        namespace BANK_1 {
            uint64_t base = 0x02000800;
            uint64_t size = 0x00000800;
        }
        namespace BANK_2 {
            uint64_t base = 0x02001000;
            uint64_t size = 0x00000800;
        }
        namespace BANK_3 {
            uint64_t base = 0x02001800;
            uint64_t size = 0x00000800;
        }
    };
    namespace ACTIVATION {
        uint64_t base = 0x02400000;
        uint64_t size = 0x00100000;
        namespace INST_BUF {
            uint64_t base = 0x02400000;
            uint64_t size = 0x00004000;
        }
        namespace INST_BUF_TAIL_PTR {
            uint64_t base = 0x02404000;
            uint64_t size = 0x00000004;
        }
        namespace PROFILE_TABLE {
            uint64_t base = 0x02420000;
            uint64_t size = 0x00002000;
        }
        namespace PROFILE_CAM {
            uint64_t base = 0x02440000;
            uint64_t size = 0x00000400;
        }
        namespace PWL_TABLE {
            uint64_t base = 0x02460000;
            uint64_t size = 0x00004000;
        }
        namespace PWL_CONTROL_TABLE {
            uint64_t base = 0x02480000;
            uint64_t size = 0x00004000;
        }
    };
    namespace POOLING {
        uint64_t base = 0x02500000;
        uint64_t size = 0x00100000;
        namespace INST_BUF {
            uint64_t base = 0x02500000;
            uint64_t size = 0x00004000;
        }
        namespace INST_BUF_TAIL_PTR {
            uint64_t base = 0x02504000;
            uint64_t size = 0x00000004;
        }
        namespace PROFILE_TABLE {
            uint64_t base = 0x02520000;
            uint64_t size = 0x00002000;
        }
        namespace PROFILE_CAM {
            uint64_t base = 0x02540000;
            uint64_t size = 0x00000400;
        }
    };
    namespace PE_ARRAY {
        uint64_t base = 0x02600000;
        uint64_t size = 0x00100000;
        namespace INST_BUF {
            uint64_t base = 0x02600000;
            uint64_t size = 0x00004000;
        }
        namespace INST_BUF_TAIL_PTR {
            uint64_t base = 0x02604000;
            uint64_t size = 0x00000004;
        }
        namespace PROFILE_TABLE {
            uint64_t base = 0x02620000;
            uint64_t size = 0x00002000;
        }
        namespace PROFILE_CAM {
            uint64_t base = 0x02640000;
            uint64_t size = 0x00000400;
        }
        namespace DEQUANT_TABLES {
            uint64_t base = 0x02660000;
            uint64_t size = 0x00004000;
        }
    };
    namespace EVENTS {
        uint64_t base = 0x02700000;
        uint64_t size = 0x00100000;
    };
};

#endif
