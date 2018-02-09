/*
 * Copyright 2018, Amazon.com, Inc. or its affiliates. All Rights Reserved
 */
#ifndef __AWS_HAL_TPB_PE_PROFILES_HPP__
#define __AWS_HAL_TPB_PE_PROFILES_HPP__

/**
 * Tensor Processing Block (TPB) / PE-ARRAY / Profiles
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

#include "aws_hal_tpb_pe.hpp"

/* MatMul */
static struct aws_hal_tpb_pe_profile_table_params pe_profile_MatMul =
{
    // TODO - most constants below will be eventually replaced with ISA-probe calls,
    //        which get the offset&size of a certain field in the instruction from the ISA itself.
    //        once we get there, we can create these instructions with a for-loop, just passing for each step
    //        the instruction and the address in the profile table, and probing for all the relevant info.
    .common_params =
    {
        // Software self-check fields
        .profile_entry_type = TPB_ENGINE_PE_ARRAY,
        .addr = TPB_PE_PROFILE_ID_MAT_MUL,

        // Opcode
        .opcode = 0x1,

        // Control fields
        .inst_length        = {.offset=1,  .size=1},
        // TODO - after we align the ISA, this will be replaced with:
        // .inst_length = {.offset=tpb_isa.MatMul.get_offset(), .size=tpb_isa.MatMul.get_size}
        // (and same for most fields below)

        .misc_imm_vals[0]      = {.offset=7,  .size=1}, // performance optimizations (double-row / double-col / double-pixel)
        .misc_imm_vals[1]      = {.offset=8,  .size=2}, // instruction control bits (start/end tensor-computation)
        .misc_imm_vals[2]      = {.offset=10, .size=2}, // quantization offset
        .misc_imm_vals[3]      = {.offset=0,  .size=0}, // unused
        .misc_imm_vals[4]      = {.offset=0,  .size=0}, // unused
        .misc_imm_vals[5]      = {.offset=0,  .size=0}, // unused
        .in_data_type[0]       = {.offset=6,  .size=1},
        .in_data_type[1]       = {.offset=0,  .size=0}, // unused
        .out_data_type         = {.offset=0,  .size=0}, // unused
        .num_partitions        = {.offset=28, .size=1},

        // Read memory access-pattern
        .rd_start_addr[0]      = {.offset=12, .size=4},
        .rd_step[0][0]         = {.offset=16, .size=2},
        .rd_num_elements[0][0] = {.offset=18, .size=2},
        .rd_step[0][1]         = {.offset=20, .size=2},
        .rd_num_elements[0][1] = {.offset=22, .size=2},
        .rd_step[0][2]         = {.offset=24, .size=2},
        .rd_num_elements[0][2] = {.offset=26, .size=2},
        .rd_step[0][3]         = {.offset=0,  .size=0}, // unused (3D read)
        .rd_num_elements[0][3] = {.offset=0,  .size=0}, // unused (3D read)

        // Write memory access-pattern
        .wr_start_addr         = {.offset=29, .size=4},
        .wr_step[0]            = {.offset=33, .size=2},
        .wr_num_elements[0]    = {.offset=35, .size=2},
        .wr_step[1]            = {.offset=37, .size=2},
        .wr_num_elements[1]    = {.offset=39, .size=2},
        .wr_step[2]            = {.offset=0,  .size=0}, // unused (2D write)
        .wr_num_elements[2]    = {.offset=0,  .size=0}, // unused (2D write)
        .wr_step[3]            = {.offset=0,  .size=0}, // unused (2D write)
        .wr_num_elements[3]    = {.offset=0,  .size=0}, // unused (2D write)

        // Control flow (events)
        .event_trigger_condition = TPB_EVENT_TRIGGER_LAST_READ_ELEMENT_DIM_W, // trigger event when finished reading full wave (last dimension).
                                                            // for MatMul, event will get delayed by the PE-array latency, to assure next engine
                                                            // only gets the event after last result was written to the psum-buffer
        .wait_event_mode    = {.offset=2, .size=1},
        .wait_event_idx     = {.offset=3, .size=1},
        .set_event_mode     = {.offset=4, .size=1},
        .set_event_idx      = {.offset=5, .size=1}
    }
};

/* WeightLoad */
static struct aws_hal_tpb_pe_profile_table_params pe_profile_WeightLoad =
{
    .common_params =
    {
        // Software self-check fields
        .profile_entry_type = TPB_ENGINE_PE_ARRAY,
        .addr = TPB_PE_PROFILE_ID_WEIGHT_LOAD,

        // Opcode
        .opcode = 0x0,
    
        // Control fields
        .inst_length        = {.offset=1,  .size=1},
        .misc_imm_vals[0]   = {.offset=7,  .size=1}, // performance optimizations (double-row / double-col / double-pixel)
        .misc_imm_vals[1]   = {.offset=12, .size=2}, // quantization offset
        .misc_imm_vals[2]   = {.offset=0,  .size=0}, // unused
        .misc_imm_vals[3]   = {.offset=0,  .size=0}, // unused
        .misc_imm_vals[4]   = {.offset=0,  .size=0}, // unused
        .misc_imm_vals[5]   = {.offset=0,  .size=0}, // unused
        .in_data_type[0]    = {.offset=6,  .size=1},
        .in_data_type[1]    = {.offset=0,  .size=0}, // unused
        .out_data_type      = {.offset=6,  .size=1},
        .num_partitions     = {.offset=1,  .size=1},

        // Read memory access-pattern
        .rd_start_addr[0]      = {.offset=14, .size=4},
        .rd_step[0][0]         = {.offset=18, .size=2},
        .rd_num_elements[0][0] = {.offset=20, .size=2},
        .rd_step[0][1]         = {.offset=0,  .size=0}, // unused (1D read)
        .rd_num_elements[0][1] = {.offset=0,  .size=0}, // unused (1D read)
        .rd_step[0][2]         = {.offset=0,  .size=0}, // unused (1D read)
        .rd_num_elements[0][2] = {.offset=0,  .size=0}, // unused (1D read)
        .rd_step[0][3]         = {.offset=0,  .size=0}, // unused (1D read)
        .rd_num_elements[0][3] = {.offset=0,  .size=0}, // unused (1D read)

        // Write memory access-pattern
        .wr_start_addr      = {.offset=0,  .size=0}, // unused (no writes)
        .wr_step[0]         = {.offset=0,  .size=0}, // unused (no writes)
        .wr_num_elements[0] = {.offset=0,  .size=0}, // unused (no writes)
        .wr_step[1]         = {.offset=0,  .size=0}, // unused (no writes)
        .wr_num_elements[1] = {.offset=0,  .size=0}, // unused (no writes)
        .wr_step[2]         = {.offset=0,  .size=0}, // unused (no writes)
        .wr_num_elements[2] = {.offset=0,  .size=0}, // unused (no writes)
        .wr_step[3]         = {.offset=0,  .size=0}, // unused (no writes)
        .wr_num_elements[3] = {.offset=0,  .size=0}, // unused (no writes)

        // Control flow (events)
        .event_trigger_condition = TPB_EVENT_TRIGGER_LAST_READ_ELEMENT_DIM_W, // trigger event when finished reading full wave (last dimension).
        .wait_event_mode    = {.offset=2, .size=1},
        .wait_event_idx     = {.offset=3, .size=1},
        .set_event_mode     = {.offset=4, .size=1},
        .set_event_idx      = {.offset=5, .size=1}
    }
};

/* Nop */
static struct aws_hal_tpb_pe_profile_table_params pe_profile_Nop =
{
    .common_params =
    {
        // Software self-check fields
        .profile_entry_type = TPB_ENGINE_PE_ARRAY,
        .addr = TPB_PE_PROFILE_ID_NOP,

        // Opcode
        .opcode = 0xE,
    
        // Control fields
        .inst_length        = {.offset=1,  .size=1},
        .misc_imm_vals[0]   = {.offset=6,  .size=1}, // num cycles
        .misc_imm_vals[1]   = {.offset=0,  .size=0}, // unused
        .misc_imm_vals[2]   = {.offset=0,  .size=0}, // unused
        .misc_imm_vals[3]   = {.offset=0,  .size=0}, // unused
        .misc_imm_vals[4]   = {.offset=0,  .size=0}, // unused
        .misc_imm_vals[5]   = {.offset=0,  .size=0}, // unused
        .in_data_type[0]    = {.offset=0,  .size=0}, // unused
        .in_data_type[1]    = {.offset=0,  .size=0}, // unused
        .out_data_type      = {.offset=0,  .size=0}, // unused
        .num_partitions     = {.offset=0,  .size=0}, // unused

        // Read memory access-pattern
        .rd_start_addr[0]      = {.offset=0,  .size=0}, // unused
        .rd_step[0][0]         = {.offset=0,  .size=0}, // unused
        .rd_num_elements[0][0] = {.offset=0,  .size=0}, // unused
        .rd_step[0][1]         = {.offset=0,  .size=0}, // unused
        .rd_num_elements[0][1] = {.offset=0,  .size=0}, // unused
        .rd_step[0][2]         = {.offset=0,  .size=0}, // unused
        .rd_num_elements[0][2] = {.offset=0,  .size=0}, // unused
        .rd_step[0][3]         = {.offset=0,  .size=0}, // unused
        .rd_num_elements[0][3] = {.offset=0,  .size=0}, // unused

        // Write memory access-pattern
        .wr_start_addr      = {.offset=0,  .size=0}, // unused
        .wr_step[0]         = {.offset=0,  .size=0}, // unused
        .wr_num_elements[0] = {.offset=0,  .size=0}, // unused
        .wr_step[1]         = {.offset=0,  .size=0}, // unused
        .wr_num_elements[1] = {.offset=0,  .size=0}, // unused
        .wr_step[2]         = {.offset=0,  .size=0}, // unused
        .wr_num_elements[2] = {.offset=0,  .size=0}, // unused
        .wr_step[3]         = {.offset=0,  .size=0}, // unused
        .wr_num_elements[3] = {.offset=0,  .size=0}, // unused

        // Control flow (events)
        .event_trigger_condition = TPB_EVENT_TRIGGER_INST_DONE,
        .wait_event_mode    = {.offset=2, .size=1},
        .wait_event_idx     = {.offset=3, .size=1},
        .set_event_mode     = {.offset=4, .size=1},
        .set_event_idx      = {.offset=5, .size=1}
    }
};

#endif

