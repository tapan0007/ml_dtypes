/*
 * Copyright 2018, Amazon.com, Inc. or its affiliates. All Rights Reserved
 */
#ifndef __AWS_HAL_TPB_POOL_PROFILES_HPP__
#define __AWS_HAL_TPB_POOL_PROFILES_HPP__

/**
 * Tensor Processing Block (TPB) / POOLING / Profiles
 * This file contains the profile-configuration for the Pooling engine
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

#include "aws_hal_tpb_pool.h"

/* MaxPool (phase-1, pass) */
static struct aws_hal_tpb_pool_profile_table_params pool_profile_MaxPool_1 =
{
    // TODO
};
/* MaxPool (phase-2, max) */
static struct aws_hal_tpb_pool_profile_table_params pool_profile_MaxPool_2 =
{
    // TODO
};

/* AveragePool (phase-1, pass) */
static struct aws_hal_tpb_pool_profile_table_params pool_profile_AveragePool_1 =
{
    // TODO
};
/* AveragePool (phase-2, add) */
static struct aws_hal_tpb_pool_profile_table_params pool_profile_AveragePool_2 =
{
    // TODO
};

/* VectorAdd */
static struct aws_hal_tpb_pool_profile_table_params pool_profile_VectorAdd =
{
    // TODO
};

/* ScaleAndAdd */
static struct aws_hal_tpb_pool_profile_table_params pool_profile_ScaleAndAdd =
{
    // TODO
};

/* Nop */
static struct aws_hal_tpb_pool_profile_table_params pool_profile_Nop =
{
    .common_params =
    {
        // Software self-check fields
        .profile_entry_type = TPB_ENGINE_POOLING,
        .addr = TPB_POOL_PROFILE_ID_NOP,

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

/* SetEvent */
static struct aws_hal_tpb_pool_profile_table_params pool_profile_SetEvent =
{
    .common_params =
    {
        // Software self-check fields
        .profile_entry_type = TPB_ENGINE_POOLING,
        .addr = TPB_POOL_PROFILE_ID_SET_EVENT,

        // Opcode
        .opcode = 0x11,
    
        // Control fields
        .inst_length        = {.offset=1,  .size=1},
        .misc_imm_vals[0]   = {.offset=0,  .size=0}, // unused
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
        .wait_event_mode    = {.offset=0, .size=0}, // unused
        .wait_event_idx     = {.offset=0, .size=0}, // unused
        .set_event_mode     = {.offset=0, .size=0}, // unused
        .set_event_idx      = {.offset=2, .size=1}  // event index
        // TODO - we will add a PT option to take event-mode from PT and not from INST
    }
};

/* ClearEvent */
static struct aws_hal_tpb_pool_profile_table_params pool_profile_ClearEvent =
{
    .common_params =
    {
        // Software self-check fields
        .profile_entry_type = TPB_ENGINE_POOLING,
        .addr = TPB_POOL_PROFILE_ID_SET_EVENT,

        // Opcode
        .opcode = 0x12,
    
        // Control fields
        .inst_length        = {.offset=1,  .size=1},
        .misc_imm_vals[0]   = {.offset=0,  .size=0}, // unused
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
        .wait_event_mode    = {.offset=0, .size=0}, // unused
        .wait_event_idx     = {.offset=0, .size=0}, // unused
        .set_event_mode     = {.offset=0, .size=0}, // unused
        .set_event_idx      = {.offset=2, .size=1}  // event index
        // TODO - we will add a PT option to take event-mode from PT and not from INST
    }
};

/* WaitEvent */
static struct aws_hal_tpb_pool_profile_table_params pool_profile_WaitEvent =
{
    .common_params =
    {
        // Software self-check fields
        .profile_entry_type = TPB_ENGINE_POOLING,
        .addr = TPB_POOL_PROFILE_ID_SET_EVENT,

        // Opcode
        .opcode = 0x10,
    
        // Control fields
        .inst_length        = {.offset=1,  .size=1},
        .misc_imm_vals[0]   = {.offset=0,  .size=0}, // unused
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
        .wait_event_mode    = {.offset=0, .size=0}, // unused
        .wait_event_idx     = {.offset=2, .size=1}, // event index
        .set_event_mode     = {.offset=0, .size=0}, // unused
        .set_event_idx      = {.offset=0, .size=0}  // unused
        // TODO - we should have a PT option for WaitEventMode
    }
};

/* Write */
static struct aws_hal_tpb_pool_profile_table_params pool_profile_Write =
{
    .common_params =
    {
        // Software self-check fields
        .profile_entry_type = TPB_ENGINE_POOLING,
        .addr = TPB_POOL_PROFILE_ID_SET_EVENT,

        // Opcode
        .opcode = 0x10,
    
        // Control fields
        .inst_length        = {.offset=1,  .size=1},
        .misc_imm_vals[0]   = {.offset=2,  .size=4}, // write address
        .misc_imm_vals[1]   = {.offset=6,  .size=8}, // write data
        .misc_imm_vals[2]   = {.offset=14, .size=1}, // write size (in bytes)
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
        .wait_event_mode    = {.offset=0, .size=0}, // unused
        .wait_event_idx     = {.offset=0, .size=0}, // unused
        .set_event_mode     = {.offset=0, .size=0}, // unused
        .set_event_idx      = {.offset=0, .size=0}  // unused
    }
};

#endif

