/*
 * Copyright 2018, Amazon.com, Inc. or its affiliates. All Rights Reserved
 */
#ifndef __AWS_HAL_TPB_POOL_HPP__
#define __AWS_HAL_TPB_POOL_HPP__
/**
 * Tensor Processing Block (TPB) / Pooling Engine
 * Terminology:
 *  + TPB has a single Pooling Engine
 +  + The pooling engine within the TPB instansiates 64 Pooling channels
 *  + Each pooling channel is built out of 2 ALUs
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

#define AWS_HAL_TPB_POOL_ENGINE_NUM_CHANNELS 64 // 64 Pooling-Channels in the Pooling-Engine
#define AWS_HAL_TPB_POOL_CHANNEL_NUM_ALUS 2 // 2 ALUs in each Pooling-Channel
#define AWS_HAL_TPB_POOL_CHANNEL_NUM_ALU_INPUTS 2 // 2 inputs to each ALU
#define AWS_HAL_TPB_POOL_PROFILE_SIZE 64

enum aws_hal_tpb_pool_profile_id {
    TPB_POOL_PROFILE_ID_MAX_POOL_STEP1 = 0,
    TPB_POOL_PROFILE_ID_MAX_POOL_STEP2 = 1,
    TPB_POOL_PROFILE_ID_AVERAGE_POOL_STEP1 = 2,
    TPB_POOL_PROFILE_ID_AVERAGE_POOL_STEP2 = 3,
    TPB_POOL_PROFILE_ID_VECTOR_ADD = 4,
    TPB_POOL_PROFILE_ID_SCALE_AND_ADD = 5,
    TPB_POOL_PROFILE_ID_NOP = 6,
    TPB_POOL_PROFILE_ID_SET_EVENT = 7,
    TPB_POOL_PROFILE_ID_CLEAR_EVENT = 8,
    TPB_POOL_PROFILE_ID_WAIT_EVENT = 9,
    TPB_POOL_PROFILE_ID_WRITE = 10,
    // 7-31: Reserved
    TPB_POOL_NUM_PROFILES
};




enum aws_hal_tpb_pool_alu_in_sel {
    TPB_POOL_ALU_IN_PROFILE_TABLE_CONST_0 = 0,
    TPB_POOL_ALU_IN_PROFILE_TABLE_CONST_1 = 1,
    TPB_POOL_ALU_IN_INGRESS_FIFO_0 = 2,
    TPB_POOL_ALU_IN_INGRESS_FIFO_1 = 3,
    TPB_POOL_ALU_IN_INGRESS_FIFO_REL = 4,
    // 5-7: Reserved
    TPB_POOL_ALU_IN_CURRENT_ALU_RESULT = 8,
    TPB_POOL_ALU_IN_ALU_0_RESULT = 9,
    TPB_POOL_ALU_IN_ALU_1_RESULT = 10,
    TPB_POOL_ALU_IN_CONCAT_POOL_ALU_0_RESULT = 11,
    TPB_POOL_ALU_IN_CONCAT_POOL_ALU_1_RESULT = 12,
    TPB_POOL_ALU_IN_INST_IMM_VAL_0 = 16,
    TPB_POOL_ALU_IN_INST_IMM_VAL_1 = 17,
    TPB_POOL_ALU_IN_INST_IMM_VAL_2 = 18,
    TPB_POOL_ALU_IN_INST_IMM_VAL_3 = 19,
    TPB_POOL_ALU_IN_INST_IMM_VAL_4 = 20,
    TPB_POOL_ALU_IN_INST_IMM_VAL_5 = 21
    // 22-31: Reserved
};

enum aws_hal_tpb_pool_alu_data_type_sel {
    TPB_POOL_ALU_DTYPE_SEL_FROM_INST = 0,
    TPB_POOL_ALU_DTYPE_SEL_FP32 = 1,
    TPB_POOL_ALU_DTYPE_SEL_BITVECTOR = 2
    // 3: Reserved
};

enum aws_hal_tpb_pool_alu_opcode {
    TPB_POOL_ALU_OPCODE_PASS_A = 0, // out = A
    TPB_POOL_ALU_OPCODE_INV_A = 1, // out = ~A
    TPB_POOL_ALU_OPCODE_SHIFT_LEFT = 2, // out = A<<B
    TPB_POOL_ALU_OPCODE_SHIFT_RIGHT = 3, // out = A>>B
    TPB_POOL_ALU_OPCODE_ADD = 4, // out = A+B
    TPB_POOL_ALU_OPCODE_SUB = 5, // out = A-B
    TPB_POOL_ALU_OPCODE_MULT = 6, // out = A*B  (only supported in ALU[1])
    TPB_POOL_ALU_OPCODE_DIV = 7, // out = A/B  (only supported in POOL[0].ALU[1])
    TPB_POOL_ALU_OPCODE_MAX = 8, // out = max{A,B}
    TPB_POOL_ALU_OPCODE_MIN = 9, // out = min{A,B}
    TPB_POOL_ALU_OPCODE_BITWISE_OR = 10, // out = A | B
    TPB_POOL_ALU_OPCODE_BITWISE_AND = 11, // out = A & B
    TPB_POOL_ALU_OPCODE_BITWISE_XOR = 12 // out = A ^ B
    // 13-15: Reserved
};

enum aws_hal_tpb_pool_step_mgmt_func {
    TPB_POOL_STEP_MGMT_INC_ALWAYS_NO_CLR = 0,
    TPB_POOL_STEP_MGMT_INC_ALWAYS_CLR_ON_LAST_READ_X = 1,
    TPB_POOL_STEP_MGMT_INC_ALWAYS_CLR_ON_LAST_READ_Y = 2,
    TPB_POOL_STEP_MGMT_INC_ON_LAST_READ_X_CLR_ON_LAST_READ_Y = 3,
    TPB_POOL_STEP_MGMT_INC_ALWAYS_CLR_ON_LAST_READ_Z = 4,
    TPB_POOL_STEP_MGMT_INC_ON_LAST_READ_X_CLR_ON_LAST_READ_Z = 5,
    TPB_POOL_STEP_MGMT_INC_ON_LAST_READ_Y_CLR_ON_LAST_READ_Z = 6,
    TPB_POOL_STEP_MGMT_INC_ALWAYS_CLR_ON_LAST_READ_W = 7,
    TPB_POOL_STEP_MGMT_INC_ON_LAST_READ_X_CLR_ON_LAST_READ_W = 8,
    TPB_POOL_STEP_MGMT_INC_ON_LAST_READ_Y_CLR_ON_LAST_READ_W = 9,
    TPB_POOL_STEP_MGMT_INC_ON_LAST_READ_Z_CLR_ON_LAST_READ_W = 10,
    TPB_POOL_STEP_MGMT_INC_ALWAYS_CLR_WHEN_REACH_THRESHOLD = 11,
    TPB_POOL_STEP_MGMT_INC_ON_LAST_READ_X_CLR_WHEN_REACH_THRESHOLD = 12,
    TPB_POOL_STEP_MGMT_INC_ON_LAST_READ_Y_CLR_WHEN_REACH_THRESHOLD = 13,
    TPB_POOL_STEP_MGMT_INC_ON_LAST_READ_Z_CLR_WHEN_REACH_THRESHOLD = 14
    // 15: Reserved
};

enum aws_hal_tpb_pool_step_threshold_sel {
    TPB_POOL_STEP_THR_SEL_INST_IMM_VAL_0 = 0,
    TPB_POOL_STEP_THR_SEL_INST_IMM_VAL_1 = 1,
    TPB_POOL_STEP_THR_SEL_INST_IMM_VAL_2 = 2,
    TPB_POOL_STEP_THR_SEL_INST_IMM_VAL_3 = 3,
    TPB_POOL_STEP_THR_SEL_INST_IMM_VAL_4 = 4,
    TPB_POOL_STEP_THR_SEL_INST_IMM_VAL_5 = 5,
    TPB_POOL_STEP_THR_SEL_PROF_TABLE_CONST = 7
};

enum aws_hal_tpb_pool_alu_res_cache_command {
    TPB_POOL_CACHE_RES_NEVER = 0, // never cache ALU results
    TPB_POOL_CACHE_RES_ALWAYS = 1, // cache all ALU results (every valid result)
    TPB_POOL_CACHE_RES_LAST_X = 2, // cache when reading the last element of dim-x
    TPB_POOL_CACHE_RES_LAST_Y = 3, // cache when reading the last element of dim-y
    TPB_POOL_CACHE_RES_LAST_Z = 4, // cache when reading the last element of dim-z
    TPB_POOL_CACHE_RES_LAST_W = 5  // cache when reading the last element of dim-w
    // 6-7: Reserved
};

enum aws_hal_tpb_pool_out_sel {
    TPB_POOL_OUT_SEL_ALU_0 = 0,
    TPB_POOL_OUT_SEL_ALU_1 = 1,
    TPB_POOL_OUT_SEL_ALU_0_CACHED = 2,
    TPB_POOL_OUT_SEL_ALU_1_CACHED = 3
};

/*
 * Pooling Profile Table Entry:
 * ============================
 */
struct aws_hal_tpb_pool_profile_table_params {
    // Common fields
    struct aws_hal_tpb_common_profile_table_params common_params;

    // Pooling fields
    // -- Step management (only relevant for pooling, as it is the only planar engine)
    enum aws_hal_tpb_pool_step_mgmt_func step_mgmt_func; // how to increment step from step to step
    enum aws_hal_tpb_pool_step_threshold_sel step_threshold_sel; // select threshold for step management. only relevant if step_mgm_func = *_CLEAR_WHEN_REACH_THRESHOLD
    uint8_t step_threshold_const; // potential threshold for step clearing
    uint8_t step_inc_delay; // delay to be applied prior to every step increment
    uint8_t step_clr_delay; // delay to be applied prior to every step clear

    // -- ALU control
    enum aws_hal_tpb_pool_alu_data_type_sel alu_data_type_sel; // source for ALU data-type (profile-table or instruction)
    struct aws_hal_tpb_inst_field_extract alu_data_type; // ALU data-type
    uint32_t alu_const[AWS_HAL_TPB_POOL_CHANNEL_NUM_ALUS][2];
    enum aws_hal_tpb_pool_alu_in_sel alu_in_sel[AWS_HAL_TPB_POOL_CHANNEL_NUM_ALUS][AWS_HAL_TPB_POOL_CHANNEL_NUM_ALU_INPUTS]; // ALUs input selection
    enum aws_hal_tpb_pool_alu_opcode alu_opcode[AWS_HAL_TPB_POOL_CHANNEL_NUM_ALUS]; // ALUs opcodoes
    enum aws_hal_tpb_pool_alu_res_cache_command alu_res_cache_command[AWS_HAL_TPB_POOL_CHANNEL_NUM_ALUS]; // ALU result caching command
    enum aws_hal_tpb_pool_out_sel out_sel; // pooling engine output selection
};


/*
 * Pooling init:
 * ============
 */
int aws_hal_tpb_pool_init (void* tpb_mem_handle);

#endif

