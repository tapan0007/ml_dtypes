# Copyright (C) 2018, Amazon.com. All Rights Reserved
#
# Kaena test for multi-head attention
#
# test naming
#   level-name
#   level is relative complexity and runtime cost, easiest 0  to most complex 9
#   Name can be composed of anything - 0 can be used as base/existence test
#   Columns: (nnStruct, nnConfig, nnLabel, nnArgs, rtArgs)

testConfigMap = {
    "2-multihead_attention_host": [
        "trivnet_mhatt",
        "tfloat32-wmin-1-wmax1-imin-1-imax1-"
        "batchsize4-inputlen16-outputlen32-headsize64-numhid512-neginf-100000000",
        "multihead_attention",
        " --images linspace1 linspace1 linspace1 "
        "--executors host 0 1 2 --scheduler wave2 "
        "--waive_wavegraph_checks "
        "--schedule_options ' --nname=generic --no_verify ' " # me options
        "--partition from_multi "
            "'multihead_attention/part1/b_q_heads_tiled',"
            "'multihead_attention/part1/b_k_heads_tiled',"
            "'multihead_attention/part1/b_bias_br_for_partition',"
            "'multihead_attention/part1/b_v_heads_t_tiled_for_partition' "
            "'multihead_attention/part3/output' ",
        "--input_files input_x_r:0=trivnet_input_x_r:0.npy "
                      "input_y_r:0=trivnet_input_y_r:0.npy "
                      "input_bias_br:0=trivnet_input_bias_br:0.npy "],
}


# Regression waiver mechanism
# If the testname matches the regexp then the FAIL status is replaced with
# with the string
testWaiver = [
]

noGpuTestWaiver = [
]

qemuTestWaiver = [

]

