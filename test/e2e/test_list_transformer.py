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
    "2-multihead_attention": [
        "trivnet_mhatt",
        "tfloat16-wmin-1-wmax1-imin-1-imax1-"
        "batchsize4-inputlen16-outputlen32-headsize64-numhid512-neginf-100000000",
        #"batchsize4-inputlen16-outputlen16-headsize4-numhid16-neginf-100000000",  # small input
        "multihead_attention",
        " --images linspace:0-0.1 linspace:0-0.1 linspace:0-0.1 linspace:0-0.1 linspace:0-0.1 linspace:0-0.1 linspace:0-0.1 "
        " --use_wc_2d_format --use_hwc_3d_format "
        "--executors host 1 wave 0 2 --scheduler wave2 "
        #"--executors host all --scheduler wave2 "
        "--waive_wavegraph_checks "
        "--schedule_options ' --nname=generic --no_verify ' " # me options
        "--partition from_multi "
            "'multihead_attention/part1/b_q_heads_tiled'," # sg00 - sg01 cut begins.
            "'multihead_attention/part1/b_k_heads_tiled',"
            "'multihead_attention/part1/b_bias_br_for_partition',"
            "'tr_part',"
            "'multihead_attention/part1/b_v_heads_t_tiled_for_partition' "  # sg00 - sg01 cut ends.
            "'multihead_attention/part3/pre_output'," # s01-s02 cut begins.
            "'tr_part2'", # s01 -sg02 cut ends
        " --diff_options '--tolerance 3.0 1e-5' "
        " --input_files input_x_r:0=trivnet_input_x_r:0.npy "
                      "input_y_r:0=trivnet_input_y_r:0.npy "
                      "input_bias_br:0=trivnet_input_bias_br:0.npy "
                      "input_q:0=trivnet_input_q:0.npy "
                      "input_k:0=trivnet_input_k:0.npy "
                      "input_v:0=trivnet_input_v:0.npy "
                      "input_tr:0=trivnet_input_tr:0.npy "
                      ],
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
