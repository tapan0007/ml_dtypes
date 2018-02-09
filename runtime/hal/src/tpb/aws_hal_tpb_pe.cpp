/*
 * Copyright 2018, Amazon.com, Inc. or its affiliates. All Rights Reserved
 */

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

#include "aws_hal_tpb_pe.hpp"
#include "aws_hal_tpb_pe_profiles.hpp"

/*
 * PE-Array set profile entry:
 * ==========================
 */
static int aws_hal_tpb_pe_write_profile (void* tpb_base_addr, uint8_t profile_table_idx, struct aws_hal_tpb_pe_profile_table_params profile_entry)
{
    uint8_t* cam_addr = (uint8_t*)tpb_base_addr + TPB_MMAP::PE_ARRAY::PROFILE_CAM::base + profile_table_idx*AWS_HAL_TPB_PE_PE_PROFILE_SIZE;
    uint8_t* profile_addr = (uint8_t*)tpb_base_addr + TPB_MMAP::PE_ARRAY::PROFILE_TABLE::base + profile_table_idx*AWS_HAL_TPB_PE_PE_PROFILE_SIZE;
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
    profile_addr[byte_idx++] = profile_entry.common_params.inst_length.offset;
    profile_addr[byte_idx++] = profile_entry.common_params.inst_length.size;
    for (int k=0 ; k<AWS_HAL_TPB_PROFILE_NUM_IMM_VALS ; k++) {
        profile_addr[byte_idx++] = profile_entry.common_params.misc_imm_vals[k].offset;
        profile_addr[byte_idx++] = profile_entry.common_params.misc_imm_vals[k].size;
    }

    /* Profile.Data_Types */
    byte_idx = AWS_HAL_TPB_PROFILE_DTYPE_LSB;
    profile_addr[byte_idx++] = profile_entry.common_params.in_data_type[0].offset;
    profile_addr[byte_idx++] = profile_entry.common_params.in_data_type[0].size;
    profile_addr[byte_idx++] = 0; // in_data_type[1] offset - unused (only one input stream in PE-array)
    profile_addr[byte_idx++] = 0; // in_data_type[1] size   - unused (only one input stream in PE-array)
    profile_addr[byte_idx++] = 0; // alu_data_type src sel  - unused (no ALUs in PE-array)
    profile_addr[byte_idx++] = 0; // alu_data_type offset   - unused (no ALUs in PE-array)
    profile_addr[byte_idx++] = 0; // alu_data_type size     - unused (no ALUs in PE-array)
    profile_addr[byte_idx++] = profile_entry.common_params.out_data_type.offset;
    profile_addr[byte_idx++] = profile_entry.common_params.out_data_type.size;
    profile_addr[byte_idx++] = 0; // reserved
    profile_addr[byte_idx++] = 0; // reserved
    profile_addr[byte_idx++] = profile_entry.common_params.num_partitions.offset;
    profile_addr[byte_idx++] = profile_entry.common_params.num_partitions.size;

    /* Profile.Memory_Read_Controller */
    byte_idx = AWS_HAL_TPB_PROFILE_MEM_RD_CTRL_LSB;
    for (int j=0 ; j<2 ; j++) { // repeat for 2 read channels
        profile_addr[byte_idx++] = profile_entry.common_params.rd_start_addr[j].offset;
        profile_addr[byte_idx++] = profile_entry.common_params.rd_start_addr[j].size;
        for (int k=0 ; k<AWS_HAL_TPB_MEM_NUM_DIMENSIONS ; k++) {
            profile_addr[byte_idx++] = profile_entry.common_params.rd_step[j][k].offset;
            profile_addr[byte_idx++] = profile_entry.common_params.rd_step[j][k].size;
        }
        for (int k=0 ; k<AWS_HAL_TPB_MEM_NUM_DIMENSIONS ; k++) {
            profile_addr[byte_idx++] = profile_entry.common_params.rd_num_elements[j][k].offset;
            profile_addr[byte_idx++] = profile_entry.common_params.rd_num_elements[j][k].size;
        }
        profile_addr[byte_idx++] = 0; // reserved
    }

    /* Profile.Memory_Write_Controller */
    byte_idx = AWS_HAL_TPB_PROFILE_MEM_WR_CTRL_LSB;
    profile_addr[byte_idx++] = profile_entry.common_params.wr_start_addr.offset;
    profile_addr[byte_idx++] = profile_entry.common_params.wr_start_addr.size;
    for (int k=0 ; k<AWS_HAL_TPB_MEM_NUM_DIMENSIONS ; k++) {
        profile_addr[byte_idx++] = profile_entry.common_params.wr_step[k].offset;
        profile_addr[byte_idx++] = profile_entry.common_params.wr_step[k].size;
    }
    for (int k=0 ; k<AWS_HAL_TPB_MEM_NUM_DIMENSIONS ; k++) {
        profile_addr[byte_idx++] = profile_entry.common_params.wr_num_elements[k].offset;
        profile_addr[byte_idx++] = profile_entry.common_params.wr_num_elements[k].size;
    }
    
    /* Profile.Event_Management */
    byte_idx = AWS_HAL_TPB_PROFILE_EVENT_MGMT_LSB;
    profile_addr[byte_idx++] = profile_entry.common_params.event_trigger_condition;
    profile_addr[byte_idx++] = profile_entry.common_params.wait_event_mode.offset;
    profile_addr[byte_idx++] = profile_entry.common_params.wait_event_mode.size;
    profile_addr[byte_idx++] = profile_entry.common_params.wait_event_idx.offset;
    profile_addr[byte_idx++] = profile_entry.common_params.wait_event_idx.size;
    profile_addr[byte_idx++] = profile_entry.common_params.set_event_mode.offset;
    profile_addr[byte_idx++] = profile_entry.common_params.set_event_mode.size;
    profile_addr[byte_idx++] = profile_entry.common_params.set_event_idx.offset;
    profile_addr[byte_idx++] = profile_entry.common_params.set_event_idx.size;
    
    /* Profile.FSM_Management */
    // TODO


    return 0;
}

/*
 * PE-Array init:
 * =============
 */
int aws_hal_tpb_pe_init (void* tpb_base_addr)
{
    /* CSRs */
    // TODO

    /* Profile CAM and Table */
    int ret = 0;
    ret += aws_hal_tpb_pe_write_profile (tpb_base_addr, TPB_PE_PROFILE_ID_MAT_MUL, pe_profile_MatMul);
    ret += aws_hal_tpb_pe_write_profile (tpb_base_addr, TPB_PE_PROFILE_ID_WEIGHT_LOAD, pe_profile_WeightLoad);
    ret += aws_hal_tpb_pe_write_profile (tpb_base_addr, TPB_PE_PROFILE_ID_NOP, pe_profile_Nop);
    ret += aws_hal_tpb_pe_write_profile (tpb_base_addr, TPB_PE_PROFILE_ID_SET_EVENT, pe_profile_SetEvent);
    ret += aws_hal_tpb_pe_write_profile (tpb_base_addr, TPB_PE_PROFILE_ID_CLEAR_EVENT, pe_profile_ClearEvent);
    ret += aws_hal_tpb_pe_write_profile (tpb_base_addr, TPB_PE_PROFILE_ID_WAIT_EVENT, pe_profile_WaitEvent);
    ret += aws_hal_tpb_pe_write_profile (tpb_base_addr, TPB_PE_PROFILE_ID_WRITE, pe_profile_Write);

    return ret;
}


