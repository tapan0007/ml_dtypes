# Copyright (C) 2018, Amazon.com. All Rights Reserved
#
# Kaena test for quantized neural nets
#
# test naming
#   level-name
#   level is relative complexity and runtime cost, easiest 0  to most complex 9
#   Name can be composed of anything - 0 can be used as base/existence test
#   Columns: (nnStruct, nnConfig, nnLabel, nnArgs, rtArgs)

tffe_options_v0 = ("--executors wave 1 host 0 2 --scheduler wave2  "
    "--schedule_options ' --nname=generic --save_layer_output ' " # me options
    "--partition from_multi 'conv_ba_relu_uint8/quantized_conv2d' 'conv_ba_relu_uint8/output' ")
tffe_options_v1 = ("--executors wave 0 host 1 --scheduler wave2  "
    "--schedule_options ' --nname=generic --save_layer_output --no_verify ' " # me options
    "--partition from_multi 'conv_ba_relu_uint8/output' ")
rt_options = "--diff_options '--tolerance 1 1'"

testConfigMap = {
    # Need rather high tolerance for now since TensorFlow's quantize/dequantize
    # operators are different than ours
    "0-1conv_uint8_wave": [
        "trivnet_conv_ba_relu_uint8_v0",
        "tuint8-b1-h10-r3-s1-c1-m1-wmin-1-wmax2-amin-1-amax1-imin-3-imax4-rqmin-30-rqmax30",
        "conv_ba_relu_uint8",
        tffe_options_v0,
        rt_options],
    "0-1conv1ba1relu_b1h1r1s1c1m1_uint8_wave": [
        "trivnet_conv_ba_relu_uint8_v0",
        "tuint8-b1-h1-r1-s1-c1-m1-wmin-1-wmax1-amin-1-amax1-imin-1-imax1-rqmin-30-rqmax30-hasba1-hasrelu1",
        "conv_ba_relu_uint8",
        tffe_options_v0,
        rt_options],
    "1-1conv1ba1relu_uint8_wave": [
        "trivnet_conv_ba_relu_uint8_v0",
        "tuint8-b8-h10-r3-s1-c3-m4-wmin-1-wmax1-amin-1-amax1-imin-1-imax1-rqmin-30-rqmax30-hasba1-hasrelu1",
        "conv_ba_relu_uint8",
        tffe_options_v0,
        rt_options],
    "1-1conv1relu_uint8_wave": [
        "trivnet_conv_ba_relu_uint8_v0",
        "tuint8-b8-h10-r3-s1-c3-m4-wmin-1-wmax1-amin-1-amax1-imin-1-imax1-rqmin-30-rqmax30-hasrelu1",
        "conv_ba_relu_uint8",
        tffe_options_v0,
        rt_options],
    "1-1conv1ba_uint8_wave": [
        "trivnet_conv_ba_relu_uint8_v0",
        "tuint8-b8-h10-r3-s1-c3-m4-wmin-1-wmax1-amin-1-amax1-imin-1-imax1-rqmin-30-rqmax30-hasba1",
        "conv_ba_relu_uint8",
        tffe_options_v0,
        rt_options],
    "2-rn50_block1_uint8": [
        "trivnet_rn50_block_uint8_v0",
        "tuint8-b1-h55-s1-cin64-cout256-wmin-0.03-wmax0.03-amin-0.1-amax0.1-imin0-imax1-rqamin0-rqamax1-rqbmin0-rqbmax1",
        "rn50_block1_uint8",
        "--executors wave 0 host 1 --scheduler wave2 --waive_wavegraph_checks "
        "--schedule_options ' --nname=generic --save_layer_output --no_verify ' " # me options
        "--partition from_multi 'rn50_block1_uint8/output' ",
        rt_options],
    "0-1conv_uint8_v1_wave": [
        "trivnet_conv_ba_relu_uint8_v1",
        "tuint8-b1-h10-r3-s1-c1-m1-wmin-1-wmax2-amin-1-amax1-imin-3-imax4-rqmin-30-rqmax30",
        "conv_ba_relu_uint8",
        tffe_options_v1,
        rt_options],
    "0-1conv1ba1relu_b1h1r1s1c1m1_uint8_v1_wave": [
        "trivnet_conv_ba_relu_uint8_v1",
        "tuint8-b1-h1-r1-s1-c1-m1-wmin-1-wmax1-amin-1-amax1-imin-1-imax1-rqmin-30-rqmax30-hasba1-hasrelu1",
        "conv_ba_relu_uint8",
        tffe_options_v1,
        "--diff_options '--tolerance 1 1'"], # rt options
    "1-1conv1ba1relu_uint8_v1_wave": [
        "trivnet_conv_ba_relu_uint8_v1",
        "tuint8-b8-h10-r3-s1-c3-m4-wmin-1-wmax1-amin-1-amax1-imin-1-imax1-rqmin-30-rqmax30-hasba1-hasrelu1",
        "conv_ba_relu_uint8",
        tffe_options_v1,
        rt_options],
    "1-1conv1relu_uint8_v1_wave": [
        "trivnet_conv_ba_relu_uint8_v1",
        "tuint8-b8-h10-r3-s1-c3-m4-wmin-1-wmax1-amin-1-amax1-imin-1-imax1-rqmin-30-rqmax30-hasrelu1",
        "conv_ba_relu_uint8",
        tffe_options_v1,
        rt_options],
    "1-1conv1ba_uint8_v1_wave": [
        "trivnet_conv_ba_relu_uint8_v1",
        "tuint8-b8-h10-r3-s1-c3-m4-wmin-1-wmax1-amin-1-amax1-imin-1-imax1-rqmin-30-rqmax30-hasba1",
        "conv_ba_relu_uint8",
        tffe_options_v1,
        rt_options],
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

