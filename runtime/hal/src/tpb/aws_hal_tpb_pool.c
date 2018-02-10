/*
 * Copyright 2018, Amazon.com, Inc. or its affiliates. All Rights Reserved
 */

/**
 * Tensor Processing Block (TPB) / Pooling
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
#include "aws_hal_tpb_pool_profiles.h"

/*
 * Pooling set profile entry:
 * =========================
 */
static int aws_hal_tpb_pool_write_profile (void* tpb_mem_handle, uint8_t profile_table_idx, struct aws_hal_tpb_pool_profile_table_params profile_entry)
{
    uint8_t* cam_addr = (uint8_t*)tpb_mem_handle + TPB_MMAP_PE_ARRAY_PROFILE_CAM_BASE + profile_table_idx*AWS_HAL_TPB_POOL_PROFILE_SIZE;
    uint8_t* profile_addr = (uint8_t*)tpb_mem_handle + TPB_MMAP_PE_ARRAY_PROFILE_TABLE_BASE + profile_table_idx*AWS_HAL_TPB_POOL_PROFILE_SIZE;
    uint8_t byte_idx;

    /* CAM.data */
    al_reg_write8(&cam_addr[0], profile_entry.common_params.opcode);
    al_reg_write8(&cam_addr[1], 0); // unused
    al_reg_write8(&cam_addr[2], 0); // unused
    al_reg_write8(&cam_addr[3], 0); // unused

    /* CAM.mask */
    al_reg_write8(&cam_addr[4], 0xFF);
    al_reg_write8(&cam_addr[5], 0); // unused
    al_reg_write8(&cam_addr[6], 0); // unused
    al_reg_write8(&cam_addr[7], 0); // unused

    /* CAM.entry_valid */
    al_reg_write8(&cam_addr[8], 1);
    
    /* Profile.Instruction_Decode */
    byte_idx = AWS_HAL_TPB_PROFILE_ID_LSB;
    al_reg_write8(&profile_addr[byte_idx++], profile_entry.common_params.inst_length.offset);
    al_reg_write8(&profile_addr[byte_idx++], profile_entry.common_params.inst_length.size);
    for (int k=0 ; k<AWS_HAL_TPB_PROFILE_NUM_IMM_VALS ; k++) {
        al_reg_write8(&profile_addr[byte_idx++], profile_entry.common_params.misc_imm_vals[k].offset);
        al_reg_write8(&profile_addr[byte_idx++], profile_entry.common_params.misc_imm_vals[k].size);
    }

    /* Profile.Data_Types */
    byte_idx = AWS_HAL_TPB_PROFILE_DTYPE_LSB;
    al_reg_write8(&profile_addr[byte_idx++], profile_entry.common_params.in_data_type[0].offset);
    al_reg_write8(&profile_addr[byte_idx++], profile_entry.common_params.in_data_type[0].size);
    al_reg_write8(&profile_addr[byte_idx++], profile_entry.common_params.in_data_type[1].offset);
    al_reg_write8(&profile_addr[byte_idx++], profile_entry.common_params.in_data_type[1].size);
    al_reg_write8(&profile_addr[byte_idx++], profile_entry.alu_data_type_sel);
    al_reg_write8(&profile_addr[byte_idx++], profile_entry.alu_data_type.offset);
    al_reg_write8(&profile_addr[byte_idx++], profile_entry.alu_data_type.size);
    al_reg_write8(&profile_addr[byte_idx++], profile_entry.common_params.out_data_type.offset);
    al_reg_write8(&profile_addr[byte_idx++], profile_entry.common_params.out_data_type.size);
    al_reg_write8(&profile_addr[byte_idx++], 0); // reserved
    al_reg_write8(&profile_addr[byte_idx++], 0); // reserved
    al_reg_write8(&profile_addr[byte_idx++], profile_entry.common_params.num_partitions.offset);
    al_reg_write8(&profile_addr[byte_idx++], profile_entry.common_params.num_partitions.size);

    /* Profile.Memory_Read_Controller */
    byte_idx = AWS_HAL_TPB_PROFILE_MEM_RD_CTRL_LSB;
    for (int j=0 ; j<2 ; j++) { // repeat for 2 read channels
        al_reg_write8(&profile_addr[byte_idx++], profile_entry.common_params.rd_start_addr[j].offset);
        al_reg_write8(&profile_addr[byte_idx++], profile_entry.common_params.rd_start_addr[j].size);
        for (int k=0 ; k<AWS_HAL_TPB_MEM_NUM_DIMENSIONS ; k++) {
            al_reg_write8(&profile_addr[byte_idx++], profile_entry.common_params.rd_step[j][k].offset);
            al_reg_write8(&profile_addr[byte_idx++], profile_entry.common_params.rd_step[j][k].size);
        }
        for (int k=0 ; k<AWS_HAL_TPB_MEM_NUM_DIMENSIONS ; k++) {
            al_reg_write8(&profile_addr[byte_idx++], profile_entry.common_params.rd_num_elements[j][k].offset);
            al_reg_write8(&profile_addr[byte_idx++], profile_entry.common_params.rd_num_elements[j][k].size);
        }
        al_reg_write8(&profile_addr[byte_idx++], 0); // reserved
    }

    /* Profile.Memory_Write_Controller */
    byte_idx = AWS_HAL_TPB_PROFILE_MEM_WR_CTRL_LSB;
    al_reg_write8(&profile_addr[byte_idx++], profile_entry.common_params.wr_start_addr.offset);
    al_reg_write8(&profile_addr[byte_idx++], profile_entry.common_params.wr_start_addr.size);
    for (int k=0 ; k<AWS_HAL_TPB_MEM_NUM_DIMENSIONS ; k++) {
        al_reg_write8(&profile_addr[byte_idx++], profile_entry.common_params.wr_step[k].offset);
        al_reg_write8(&profile_addr[byte_idx++], profile_entry.common_params.wr_step[k].size);
    }
    for (int k=0 ; k<AWS_HAL_TPB_MEM_NUM_DIMENSIONS ; k++) {
        al_reg_write8(&profile_addr[byte_idx++], profile_entry.common_params.wr_num_elements[k].offset);
        al_reg_write8(&profile_addr[byte_idx++], profile_entry.common_params.wr_num_elements[k].size);
    }
    
    /* ALU Commands */
    // TODO

    /* Output Selection */
    // TODO

    /* Step Management */
    // TODO

    /* Profile.Event_Management */
    byte_idx = AWS_HAL_TPB_PROFILE_EVENT_MGMT_LSB;
    al_reg_write8(&profile_addr[byte_idx++], profile_entry.common_params.event_trigger_condition);
    al_reg_write8(&profile_addr[byte_idx++], profile_entry.common_params.wait_event_mode.offset);
    al_reg_write8(&profile_addr[byte_idx++], profile_entry.common_params.wait_event_mode.size);
    al_reg_write8(&profile_addr[byte_idx++], profile_entry.common_params.wait_event_idx.offset);
    al_reg_write8(&profile_addr[byte_idx++], profile_entry.common_params.wait_event_idx.size);
    al_reg_write8(&profile_addr[byte_idx++], profile_entry.common_params.set_event_mode.offset);
    al_reg_write8(&profile_addr[byte_idx++], profile_entry.common_params.set_event_mode.size);
    al_reg_write8(&profile_addr[byte_idx++], profile_entry.common_params.set_event_idx.offset);
    al_reg_write8(&profile_addr[byte_idx++], profile_entry.common_params.set_event_idx.size);
    
    /* Profile.FSM_Management */
    // TODO


    return 0;
}

/*
 * Pooling init:
 * ============
 */
int aws_hal_tpb_pool_init (void* tpb_mem_handle)
{
    /* CSRs */
    // TODO

    /* Profile CAM and Table */
    int ret = 0;
    ret += aws_hal_tpb_pool_write_profile (tpb_mem_handle, TPB_POOL_PROFILE_ID_MAX_POOL_STEP1, pool_profile_MaxPool_1);
    ret += aws_hal_tpb_pool_write_profile (tpb_mem_handle, TPB_POOL_PROFILE_ID_MAX_POOL_STEP2, pool_profile_MaxPool_2);
    ret += aws_hal_tpb_pool_write_profile (tpb_mem_handle, TPB_POOL_PROFILE_ID_AVERAGE_POOL_STEP1, pool_profile_AveragePool_1);
    ret += aws_hal_tpb_pool_write_profile (tpb_mem_handle, TPB_POOL_PROFILE_ID_AVERAGE_POOL_STEP2, pool_profile_AveragePool_2);
    ret += aws_hal_tpb_pool_write_profile (tpb_mem_handle, TPB_POOL_PROFILE_ID_VECTOR_ADD, pool_profile_VectorAdd);
    ret += aws_hal_tpb_pool_write_profile (tpb_mem_handle, TPB_POOL_PROFILE_ID_SCALE_AND_ADD, pool_profile_ScaleAndAdd);
    ret += aws_hal_tpb_pool_write_profile (tpb_mem_handle, TPB_POOL_PROFILE_ID_NOP, pool_profile_Nop);
    ret += aws_hal_tpb_pool_write_profile (tpb_mem_handle, TPB_POOL_PROFILE_ID_SET_EVENT, pool_profile_SetEvent);
    ret += aws_hal_tpb_pool_write_profile (tpb_mem_handle, TPB_POOL_PROFILE_ID_CLEAR_EVENT, pool_profile_ClearEvent);
    ret += aws_hal_tpb_pool_write_profile (tpb_mem_handle, TPB_POOL_PROFILE_ID_WAIT_EVENT, pool_profile_WaitEvent);
    ret += aws_hal_tpb_pool_write_profile (tpb_mem_handle, TPB_POOL_PROFILE_ID_WRITE, pool_profile_Write);

    return ret;
}

