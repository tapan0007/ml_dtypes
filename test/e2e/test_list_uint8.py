# Copyright (C) 2018, Amazon.com. All Rights Reserved
#
# Kaena test for quantized neural nets
#
# test naming
#   level-name
#   level is relative complexity and runtime cost, easiest 0  to most complex 9
#   Name can be composed of anything - 0 can be used as base/existence test
#   Columns: (nnStruct, nnConfig, nnLabel, nnArgs, rtArgs)


testConfigMap = {
    # Need rather high tolerance for now since TensorFlow's quantize/dequantize
    # operators are different than ours
    "0-1conv_uint8_wave": [
        "trivnet_conv_uint8_v0",
        "tuint8-b1-h10-r3-s1-c1-m1-wmin-1-wmax2-imin-3-imax4-rqmin-30-rqmax30",
        "conv2d_uint8",
        "--executors host 0 2 wave 1 --scheduler wave2  "
        "--schedule_options ' --nname=generic --save_layer_output ' " # me options
        "--partition from_multi 'conv2d_uint8/quantized_conv2d' 'conv2d_uint8/output' ",
        "--diff_options '--tolerance 1 1'"], # rt options
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

