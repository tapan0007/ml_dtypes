/*
 * Copyright 2018, Amazon.com, Inc. or its affiliates. All Rights Reserved
 */
#ifndef __AWS_HAL_TPB_COMMON_HPP__
#define __AWS_HAL_TPB_COMMON_HPP__
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


#include "aws_hal_tpb_common_mmap.hpp"
#include "aws_hal_tmp_file.hpp" // TODO - remove
// TODO - #include "aws_isa_tpb.hpp"

/* General Defines */
#define AWS_HAL_TPB_MEM_NUM_DIMENSIONS 4

/* Profile Entry Defines */
#define AWS_HAL_TPB_PROFILE_ID_LSB 0
#define AWS_HAL_TPB_PROFILE_DTYPE_LSB 16
#define AWS_HAL_TPB_PROFILE_MEM_RD_CTRL_LSB 29
#define AWS_HAL_TPB_PROFILE_MEM_WR_CTRL_LSB 75
#define AWS_HAL_TPB_PROFILE_EVENT_MGMT_LSB 131
#define AWS_HAL_TPB_PROFILE_NUM_IMM_VALS 6

/*
 * Execution Engines
 * TPB has 3 embedded execution engines: PE-Array, Activation and Pooling
 */
enum aws_hal_tpb_exec_engine {
    TPB_ENGINE_PE_ARRAY,
    TPB_ENGINE_ACTIVATION,
    TPB_ENGINE_POOLING
};

/*
 * Event Generation Condition
 * each TPB instruction can generate a local event, for synchronizing between the execution engines.
 * the possible conditions for trigger the event are defined below
 */
enum aws_hal_tpb_common_event_trigger_condition {
    TPB_EVENT_TRIGGER_NONE = 0, // don't generate event
    TPB_EVENT_TRIGGER_LAST_READ_X = 1,
    TPB_EVENT_TRIGGER_LAST_READ_Y = 2,
    TPB_EVENT_TRIGGER_LAST_READ_Z = 3,
    TPB_EVENT_TRIGGER_LAST_READ_W = 4,
    TPB_EVENT_TRIGGER_LAST_WRITTEN_X = 5,
    TPB_EVENT_TRIGGER_LAST_WRITTEN_Y = 6,
    TPB_EVENT_TRIGGER_LAST_WRITTEN_Z = 7,
    TPB_EVENT_TRIGGER_LAST_WRITTEN_W = 8,
    TPB_EVENT_TRIGGER_INST_DONE = 9 // set event unconditionally when instruction is done (relevant for 'Write'/'SetLocalEvent'/'ClearLocalEvent' instructions)
    // 10-15: Reserved
};
    

/*
 * Field-extract:
 * ==============
 * The field-extract is reused in all TPB execution engines (PE-array, Activation, Pooling).
 * It is used for extracting a field from an ISA instruction, using the (offset, size) tuple.
 * Offset and size are given in bytes (fields in the instructions are byte aligned)
 */
struct aws_hal_tpb_inst_field_extract {
	/** Offset (in bytes) for a field in the instruction */
	unsigned int offset;
	/** Size (in bytes) of the extracted field */
	unsigned int size;
};

/*
 * TPB Profile Table Entry:
 * ====================
 * Profile-tables are used across the different execution engines to parse and execture instructions.
 * The struct below defines the common fields that are shared between profile tables of the different execution engines.
 * Engine-unique fields are defined in the corresponding header files.
 *
 * Note: Many of the fields are field-extractors for parsing different fields of the TPB ISA instructions.
 *       The semantics of these different fields is described in the ISA, and thus not repeated here.
 */
struct aws_hal_tpb_common_profile_table_params {
    // Software self-check fields
    enum aws_hal_tpb_exec_engine profile_entry_type; // which execution engine this entry belongs to (sw-only field, used for validity checks)
    uint8_t addr; // entry's address in the profile-table
    uint8_t opcode;
    
    // Control fields
    struct aws_hal_tpb_inst_field_extract inst_length; // instruction-length field
    struct aws_hal_tpb_inst_field_extract misc_imm_vals[AWS_HAL_TPB_PROFILE_NUM_IMM_VALS]; // miscellaneous immediate-values extracted from the instruction
    struct aws_hal_tpb_inst_field_extract in_data_type[2]; // input data-type (2 input-streams in the general case (e.g. residue-add/bias-add)
    struct aws_hal_tpb_inst_field_extract out_data_type; // output data-type
    struct aws_hal_tpb_inst_field_extract num_partitions; // number of partitions participating in computation (relevant for both reads and writes)

    // Read memory access-pattern
    struct aws_hal_tpb_inst_field_extract rd_start_addr[2]; // read start-address
    struct aws_hal_tpb_inst_field_extract rd_step[2][AWS_HAL_TPB_MEM_NUM_DIMENSIONS]; // step-size, across the different dimensions
    struct aws_hal_tpb_inst_field_extract rd_num_elements[2][AWS_HAL_TPB_MEM_NUM_DIMENSIONS]; // number of elements, across the different dimensions

    // Write memory access-pattern
    struct aws_hal_tpb_inst_field_extract wr_start_addr; // write start-address
    struct aws_hal_tpb_inst_field_extract wr_step[AWS_HAL_TPB_MEM_NUM_DIMENSIONS]; // step-size, across the different dimensions
    struct aws_hal_tpb_inst_field_extract wr_num_elements[AWS_HAL_TPB_MEM_NUM_DIMENSIONS]; // number of elements, across the different dimensions

    // Control flow (events)
    enum aws_hal_tpb_common_event_trigger_condition event_trigger_condition; // trigger for generating an event
    struct aws_hal_tpb_inst_field_extract wait_event_mode; // waitLocalEvent mode
    struct aws_hal_tpb_inst_field_extract wait_event_idx; // waitLocalEvent index
    struct aws_hal_tpb_inst_field_extract set_event_mode; // setLocalEvent mode
    struct aws_hal_tpb_inst_field_extract set_event_idx; // setLocalEvent index

    // TODO FSM management

};

#endif
