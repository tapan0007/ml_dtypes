# Copyright (C) 2018, Amazon.com. All Rights Reserved
#
# Kaena test for quantized neural nets
#
# test naming
#   level-name
#   level is relative complexity and runtime cost, easiest 0  to most complex 9
#   Name can be composed of anything - 0 can be used as base/existence test
#   Columns: (nnStruct, nnConfig, nnLabel, nnArgs, rtArgs)

from test_list import rnPreFp32, rnDogJpg

RT_STRICT_OPTION = "--diff_options '--tolerance 0 0'"
RT_LOOSE_OPTION = "--diff_options '--tolerance 1 1e-1'"

def options_uint8_strict(module, harness, net_name):
    tffe_options = ("--executors wave 1 host 0 2 --scheduler wave2 "
        "--schedule_options ' --nname=generic --no_verify ' " # me options
        "--partition from_multi '%s/quantized_conv2d' '%s/output' " % (net_name, net_name))
    return module, harness, net_name, tffe_options, RT_STRICT_OPTION

def options_uint8_loose(module, harness, net_name, rt_option=""):
    tffe_options = ("--executors wave 0 host 1 --scheduler wave2 "
        "--schedule_options ' --nname=generic --no_verify ' " # me options
        "--partition from_multi '%s/output' " % (net_name))
    return module, harness, net_name, tffe_options, rt_option

def options_uint8_perf_strict(module, harness, net_name):
    tffe_options = ("--executors wave 1 host 0 2 --scheduler wave2 "
        "--waive_wavegraph_check "
        "--schedule_options ' --nname=generic --no_verify --uint8_performance_mode ' " # me options
        "--partition from_multi '%s/quantized_conv2d' '%s/output' " % (net_name, net_name))
    return module, harness, net_name, tffe_options, RT_STRICT_OPTION

def options_uint8_perf_loose(module, harness, net_name, rt_option=""):
    tffe_options = ("--executors wave 0 host 1 --scheduler wave2 "
        "--waive_wavegraph_check "
        "--schedule_options ' --nname=generic --no_verify --uint8_performance_mode ' " # me options
        "--partition from_multi '%s/output' " % (net_name))
    return module, harness, net_name, tffe_options, rt_option

def options_uint8_repl_strict(module, harness, net_name):
    tffe_options = ("--executors wave 1 host 0 2 --scheduler wave2 "
        "--schedule_options ' --nname=generic --enable_replication --no_verify ' " # me options
        "--partition from_multi '%s/quantized_conv2d' '%s/output' " % (net_name, net_name))
    return module, harness, net_name, tffe_options, RT_STRICT_OPTION

testConfigMap = {
    # Need rather high tolerance for now since TensorFlow's quantize/dequantize
    # operators are different than ours
    "0-1quantize_uint8_wave": [
        "trivnet_quantize_dequantize_uint8",
        "tuint8-b1-h100-c1-imin-3-imax4-wmin-1-wmax2",
        "quantize_uint8",
        "--executors wave 0 host 1 --scheduler wave2 "
        "--schedule_options ' --nname=generic ' " # me options
        "--partition from_multi 'quantize_uint8/output' ",
        RT_LOOSE_OPTION,
        ],
    "0-1dequantize_uint8_wave": [
        "trivnet_quantize_dequantize_uint8",
        "tuint8-b1-h100-c1-imin-3-imax4-wmin-1-wmax2",
        "dequantize_uint8",
        "--executors wave 1 host 0 --scheduler wave2 "
        "--schedule_options ' --nname=generic ' " # me options
        "--partition from_multi 'dequantize_uint8/output' ",
        RT_LOOSE_OPTION,
        ],
    "0-1quantize1dequantize_uint8_wave": [
        "trivnet_quantize_dequantize_uint8",
        "tuint8-b1-h100-c1-imin-3-imax4-wmin-1-wmax2",
        "quantize_dequantize_uint8",
        "--executors wave all --scheduler wave2 "
        "--schedule_options ' --nname=generic ' " # me options
        "--partition none ",
        RT_LOOSE_OPTION,
        ],
    "0-1conv_uint8_wave":                       options_uint8_strict(
        "trivnet_conv_ba_relu_pool_uint8",
        "tuint8-b1-h10-r3-s1-c1-m1-imin-3-imax4-wmin-1-wmax2-same",
        "conv_uint8"),
    "0-1conv1ba1relu_b1h1r1s1c1m1_uint8_wave":  options_uint8_loose(
        "trivnet_conv_ba_relu_pool_uint8",
        "tuint8-b1-h1-r1-s1-c1-m1-imin-1-imax1-wmin-1-wmax1-amin-1-amax1-same-hasba1-hasrelu1",
        "conv_ba_relu_uint8"),
    "1-1conv1ba1relu_uint8_wave":               options_uint8_loose(
        "trivnet_conv_ba_relu_pool_uint8",
        "tuint8-b8-h10-r3-s1-c3-m4-imin-1-imax1-wmin-1-wmax1-amin-1-amax1-same-hasba1-hasrelu1",
        "conv_ba_relu_uint8", RT_LOOSE_OPTION),
    "1-1conv1relu_uint8_wave":                  options_uint8_loose(
        "trivnet_conv_ba_relu_pool_uint8",
        "tuint8-b8-h10-r3-s1-c3-m4-imin-1-imax1-wmin-1-wmax1-amin-1-amax1-same-hasrelu1",
        "conv_relu_uint8", RT_LOOSE_OPTION),
    "1-1conv1ba_uint8_wave":                    options_uint8_loose(
        "trivnet_conv_ba_relu_pool_uint8",
        "tuint8-b8-h10-r3-s1-c3-m4-imin-1-imax1-wmin-1-wmax1-amin-1-amax1-same-hasba1",
        "conv_ba_uint8", RT_LOOSE_OPTION),
    "1-1conv1ba1relu1maxpool_uint8_wave":       options_uint8_loose(
        "trivnet_conv_ba_relu_pool_uint8",
        "tuint8-b1-h10-r3-s1-c3-m4-imin-1-imax1-wmin-1-wmax1-amin-1-amax1-same-maxpool-k3-d2-hasba1-hasrelu1-haspool1",
        "conv_ba_relu_pool_uint8"),
    "1-1conv1ba1relu1avgpool_uint8_wave":       options_uint8_loose(
        "trivnet_conv_ba_relu_pool_uint8",
        "tuint8-b1-h10-r3-s1-c3-m4-imin-1-imax1-wmin-1-wmax1-amin-1-amax1-same-avgpool-k3-d2-hasba1-hasrelu1-haspool1",
        "conv_ba_relu_pool_uint8"),
    "1-1conv_h55r2same_uint8_wave":             options_uint8_strict(
        "trivnet_conv_ba_relu_pool_uint8",
        "tuint8-b1-h55-r2-s1-c1-m1-imin-1-imax1-wmin-1-wmax1-rqmin-2-rqmax2-same",
        "conv_uint8"),
    "1-1conv_h55r2valid_uint8_wave":            options_uint8_strict(
        "trivnet_conv_ba_relu_pool_uint8",
        "tuint8-b1-h55-r2-s1-c1-m1-imin-1-imax1-wmin-1-wmax1-rqmin-2-rqmax2-valid",
        "conv_uint8"),

    ## ResNet50 blocks
    # note for block 0: has to quantize on the host first as float32 input image is too big
    "2-rn50_block0_uint8_b1_wave":  options_uint8_strict(
        "trivnet_conv_ba_relu_pool_uint8",
        "tuint8-b1-h224-r7-s2-c3-m64-imin0-imax1-wmin-0.03-wmax0.03-amin-0.1-amax0.1-same-maxpool-k3-d2-hasba1-hasrelu1-haspool1-quantizeback1-rqmin0-rqmax1",
        "rn50_block0_uint8"),
    "2-rn50_block1_uint8_b1_wave":  options_uint8_loose(
        "trivnet_rn50_block_uint8",
        "tuint8-b1-h55-s1-cin64-chid64-cout256-imin0-imax1-wmin-0.03-wmax0.03-amin-0.1-amax0.1-rqmin0-rqmax1-convbranch1",
        "rn50_block1_uint8"),
    "2-rn50_block2_uint8_b1_wave":  options_uint8_loose(
        "trivnet_rn50_block_uint8",
        "tuint8-b1-h55-s1-cin256-chid64-cout256-imin0-imax1-wmin-0.03-wmax0.03-amin-0.1-amax0.1-rqmin0-rqmax1",
        "rn50_block2_uint8"),
    # block 3 is as same as block 2
    "2-rn50_block4_uint8_b1_wave":  options_uint8_loose(
        "trivnet_rn50_block_uint8",
        "tuint8-b1-h55-s2-cin256-chid128-cout512-imin0-imax1-wmin-0.01-wmax0.01-amin-0.1-amax0.1-rqmin0-rqmax1-convbranch1",
        "rn50_block4_uint8", RT_LOOSE_OPTION),
    "2-rn50_block5_uint8_b1_wave":  options_uint8_loose(
        "trivnet_rn50_block_uint8",
        "tuint8-b1-h28-s1-cin512-chid128-cout512-imin0-imax1-wmin-0.01-wmax0.01-amin-0.1-amax0.1-rqmin0-rqmax1",
        "rn50_block5_uint8"),
    # block 6, 7 are as same as block 5
    "2-rn50_block8_uint8_b1_wave":  options_uint8_loose(
        "trivnet_rn50_block_uint8",
        "tuint8-b1-h28-s2-cin512-chid256-cout1024-imin0-imax1-wmin-0.01-wmax0.01-amin-0.1-amax0.1-rqmin0-rqmax1-convbranch1",
        "rn50_block8_uint8", RT_LOOSE_OPTION),
    "2-rn50_block9_uint8_b1_wave":  options_uint8_loose(
        "trivnet_rn50_block_uint8",
        "tuint8-b1-h14-s1-cin1024-chid256-cout1024-imin0-imax1-wmin-0.01-wmax0.01-amin-0.1-amax0.1-rqmin0-rqmax1",
        "rn50_block9_uint8"),
    # block 10, 11, 12, 13 are as same as block 9
    "2-rn50_block14_uint8_b1_wave": options_uint8_loose(
        "trivnet_rn50_block_uint8",
        "tuint8-b1-h14-s2-cin1024-chid512-cout2048-imin0-imax1-wmin-0.01-wmax0.01-amin-0.1-amax0.1-rqmin0-rqmax1-convbranch1",
        "rn50_block14_uint14", RT_LOOSE_OPTION),
    "2-rn50_block15_uint8_b1_wave": options_uint8_loose(
        "trivnet_rn50_block_uint8",
        "tuint8-b1-h7-s1-cin2048-chid512-cout2048-imin0-imax1-wmin-0.01-wmax0.01-amin-0.1-amax0.1-rqmin0-rqmax1",
        "rn50_block15_uint8", RT_LOOSE_OPTION),
    "2-rn50_block16_uint8_b1_wave": options_uint8_loose(
        "trivnet_rn50_block_uint8",
        "tuint8-b1-h7-s1-cin2048-chid512-cout2048-imin0-imax1-wmin-0.01-wmax0.01-amin-0.1-amax0.1-rqmin0-rqmax1-hasrelu1-hasavgpool1",
        "rn50_block16_uint8", RT_LOOSE_OPTION),
    "7-rn50_full_uint8_b1_wave": [
        "trivnet_rn50_full_uint8",
        "tuint8-b1-imin0-imax1-wmin-0.03-wmax0.03-amin-0.1-amax0.1-rqmin0-rqmax1",
        "rn50_full_uint8",
        "--executors wave 1 host 0 2 --scheduler wave2 "
        "--schedule_options ' --nname=generic --no_verify ' " # me options
        "--partition from_multi 'rn50_full_uint8/block_0/quantized_conv2d' 'rn50_full_uint8/output' ",
        "",
        ],
    "7-rn50_uint8_hcv0_b1_wave": [
        "tf_s3", "s3://kaena-nn-models", "resnet50_uint8_hcv0.pb",
        "%s --executors wave 1 host 0 2 --scheduler wave2 "
        "--schedule_options ' --nname=generic --no_verify ' " # me options
        "--input_node input --images %s "
        "--partition from_multi 'conv1/quantized_conv2d' 'output' " % (rnPreFp32, rnDogJpg),
        "--input_files %s" % rnDogJpg,
        ],

    # performance mode tests
    "0-1conv_h5r1_uint8_perf_wave":             options_uint8_perf_strict(
        "trivnet_conv_ba_relu_pool_uint8",
        "tuint8-b1-h5-r1-s1-c1-m1-imin-3-imax4-wmin-1-wmax2",
        "conv_uint8"),
    "0-1conv_h10r1_uint8_perf_wave":            options_uint8_perf_strict(
        "trivnet_conv_ba_relu_pool_uint8",
        "tuint8-b1-h10-r1-s1-c1-m1-imin-3-imax4-wmin-1-wmax2",
        "conv_uint8"),
    "1-1conv_h5r1c512m256_uint8_perf_wave":     options_uint8_perf_strict(
        "trivnet_conv_ba_relu_pool_uint8",
        "tuint8-b1-h5-r1-s1-c512-m256-imin-3-imax4-wmin-1-wmax2",
        "conv_uint8"),
    "1-1conv_h10r1c512m256_uint8_perf_wave":    options_uint8_perf_strict(
        "trivnet_conv_ba_relu_pool_uint8",
        "tuint8-b1-h10-r1-s1-c512-m256-imin-3-imax4-wmin-1-wmax2",
        "conv_uint8"),
    "2-1conv_h55r1c512m256_uint8_perf_wave":    options_uint8_perf_strict(
        "trivnet_conv_ba_relu_pool_uint8",
        "tuint8-b1-h55-r1-s1-c512-m256-imin-3-imax4-wmin-1-wmax2",
        "conv_uint8"),
    "2-1conv_h60r1c512m256_uint8_perf_wave":    options_uint8_perf_strict(
        "trivnet_conv_ba_relu_pool_uint8",
        "tuint8-b1-h60-r1-s1-c512-m256-imin-3-imax4-wmin-1-wmax2",
        "conv_uint8"),
    "2-1conv_h71r1c512m256_uint8_perf_wave":    options_uint8_perf_strict(
        "trivnet_conv_ba_relu_pool_uint8",
        "tuint8-b1-h71-r1-s1-c512-m256-imin-3-imax4-wmin-1-wmax2",
        "conv_uint8"),

    "1-1conv_h55r1_uint8_perf_wave":            options_uint8_perf_strict(
        "trivnet_conv_ba_relu_pool_uint8",
        "tuint8-b1-h55-r1-s1-c1-m1-imin-3-imax4-wmin-1-wmax2",
        "conv_uint8"),
    "1-1conv_h60r1_uint8_perf_wave":            options_uint8_perf_strict(
        "trivnet_conv_ba_relu_pool_uint8",
        "tuint8-b1-h60-r1-s1-c1-m1-imin-3-imax4-wmin-1-wmax2",
        "conv_uint8"),
    "1-1conv_h3r2same_uint8_perf_wave":         options_uint8_perf_strict(
        "trivnet_conv_ba_relu_pool_uint8",
        "tuint8-b1-h3-r2-s1-c1-m1-imin-3-imax4-wmin-1-wmax2-same",
        "conv_uint8"),
    "1-1conv_h3r2valid_uint8_perf_wave":        options_uint8_perf_strict(
        "trivnet_conv_ba_relu_pool_uint8",
        "tuint8-b1-h3-r2-s1-c1-m1-imin-3-imax4-wmin-1-wmax2-valid",
        "conv_uint8"),
    "1-1conv_h4r2same_uint8_perf_wave":         options_uint8_perf_strict(
        "trivnet_conv_ba_relu_pool_uint8",
        "tuint8-b1-h4-r2-s1-c1-m1-imin-3-imax4-wmin-1-wmax2-same",
        "conv_uint8"),
    "1-1conv_h4r2valid_uint8_perf_wave":        options_uint8_perf_strict(
        "trivnet_conv_ba_relu_pool_uint8",
        "tuint8-b1-h4-r2-s1-c1-m1-imin-3-imax4-wmin-1-wmax2-valid",
        "conv_uint8"),
    "1-1conv_h57r2same_uint8_perf_wave":        options_uint8_perf_strict(
        "trivnet_conv_ba_relu_pool_uint8",
        "tuint8-b1-h57-r2-s1-c1-m1-imin-3-imax4-wmin-1-wmax2-same",
        "conv_uint8"),
    "1-1conv_h57r2valid_uint8_perf_wave":       options_uint8_perf_strict(
        "trivnet_conv_ba_relu_pool_uint8",
        "tuint8-b1-h57-r2-s1-c1-m1-imin-3-imax4-wmin-1-wmax2-valid",
        "conv_uint8"),
    "1-1conv_h58r2same_uint8_perf_wave":        options_uint8_perf_strict(
        "trivnet_conv_ba_relu_pool_uint8",
        "tuint8-b1-h58-r2-s1-c1-m1-imin-3-imax4-wmin-1-wmax2-same",
        "conv_uint8"),
    "1-1conv_h58r2valid_uint8_perf_wave":       options_uint8_perf_strict(
        "trivnet_conv_ba_relu_pool_uint8",
        "tuint8-b1-h58-r2-s1-c1-m1-imin-3-imax4-wmin-1-wmax2-valid",
        "conv_uint8"),

    "1-1conv_h4r3same_uint8_perf_wave":         options_uint8_perf_strict(
        "trivnet_conv_ba_relu_pool_uint8",
        "tuint8-b1-h4-r3-s1-c1-m1-imin-3-imax4-wmin-1-wmax2-same",
        "conv_uint8"),
    "1-1conv_h4r3valid_uint8_perf_wave":        options_uint8_perf_strict(
        "trivnet_conv_ba_relu_pool_uint8",
        "tuint8-b1-h4-r3-s1-c1-m1-imin-3-imax4-wmin-1-wmax2-valid",
        "conv_uint8"),
    "1-1conv_h5r3same_uint8_perf_wave":         options_uint8_perf_strict(
        "trivnet_conv_ba_relu_pool_uint8",
        "tuint8-b1-h5-r3-s1-c1-m1-imin-3-imax4-wmin-1-wmax2-same",
        "conv_uint8"),
    "1-1conv_h5r3valid_uint8_perf_wave":        options_uint8_perf_strict(
        "trivnet_conv_ba_relu_pool_uint8",
        "tuint8-b1-h5-r3-s1-c1-m1-imin-3-imax4-wmin-1-wmax2-valid",
        "conv_uint8"),
    "1-1conv_h57r3same_uint8_perf_wave":        options_uint8_perf_strict(
        "trivnet_conv_ba_relu_pool_uint8",
        "tuint8-b1-h57-r3-s1-c1-m1-imin-3-imax4-wmin-1-wmax2-same",
        "conv_uint8"),
    "1-1conv_h57r3valid_uint8_perf_wave":       options_uint8_perf_strict(
        "trivnet_conv_ba_relu_pool_uint8",
        "tuint8-b1-h57-r3-s1-c1-m1-imin-3-imax4-wmin-1-wmax2-valid",
        "conv_uint8"),
    "1-1conv_h58r3same_uint8_perf_wave":        options_uint8_perf_strict(
        "trivnet_conv_ba_relu_pool_uint8",
        "tuint8-b1-h58-r3-s1-c1-m1-imin-3-imax4-wmin-1-wmax2-same",
        "conv_uint8"),
    "1-1conv_h58r3valid_uint8_perf_wave":       options_uint8_perf_strict(
        "trivnet_conv_ba_relu_pool_uint8",
        "tuint8-b1-h58-r3-s1-c1-m1-imin-3-imax4-wmin-1-wmax2-valid",
        "conv_uint8"),

    "2-rn50_block1_uint8_perf_b1_wave":         options_uint8_perf_loose(
        "trivnet_rn50_block_uint8",
        "tuint8-b1-h55-s1-cin64-chid64-cout256-imin0-imax1-wmin-0.03-wmax0.03-amin-0.1-amax0.1-rqmin0-rqmax1-convbranch1",
        "rn50_block1_uint8"),
    "2-rn50_block4_uint8_perf_b1_wave":         options_uint8_perf_loose(
        "trivnet_rn50_block_uint8",
        "tuint8-b1-h55-s2-cin256-chid128-cout512-imin0-imax1-wmin-0.01-wmax0.01-amin-0.1-amax0.1-rqmin0-rqmax1-convbranch1",
        "rn50_block4_uint8", RT_LOOSE_OPTION),
    "2-rn50_block16_uint8_perf_b1_wave":        options_uint8_perf_loose(
        "trivnet_rn50_block_uint8",
        "tuint8-b1-h7-s1-cin2048-chid512-cout2048-imin0-imax1-wmin-0.01-wmax0.01-amin-0.1-amax0.1-rqmin0-rqmax1-hasrelu1-hasavgpool1",
        "rn50_block16_uint8", RT_LOOSE_OPTION),
    "7-rn50_uint8_perf_hcv0_b1_wave": [
        "tf_s3", "s3://kaena-nn-models", "resnet50_uint8_hcv0.pb",
        "%s --executors wave 1 host 0 2 --scheduler wave2 "
        "--waive_wavegraph_check "
        "--schedule_options ' --nname=generic --no_verify --uint8_performance_mode ' " # me options
        "--input_node input --images %s "
        "--partition from_multi 'conv1/quantized_conv2d' 'output' " % (rnPreFp32, rnDogJpg),
        "--input_files %s" % rnDogJpg,
        ],

    # replication tests
    "3-1conv_h10r7c3m64valid_wave_repl":        options_uint8_repl_strict(
        "trivnet_conv_ba_relu_pool_uint8",
        "tuint8-b1-h10-r7-s2-c3-m64-valid-imin-3-imax4-wmin-1-wmax2",
        "conv_uint8"),
    "3-1conv_h224r7c3m64valid_wave_repl":       options_uint8_repl_strict(
        "trivnet_conv_ba_relu_pool_uint8",
        "tuint8-b1-h224-r7-s2-c3-m64-valid-imin-3-imax4-wmin-1-wmax2",
        "conv_uint8"),
    "3-rn50_block0_uint8_b1_wave_repl":         options_uint8_repl_strict(
        "trivnet_conv_ba_relu_pool_uint8",
        "tuint8-b1-h224-r7-s2-c3-m64-imin0-imax1-wmin-0.03-wmax0.03-amin-0.1-amax0.1-same-maxpool-k3-d2-hasba1-hasrelu1-haspool1-quantizeback1-rqmin0-rqmax1",
        "rn50_block0_uint8"),
    "7-rn50_uint8_hcv0_b1_wave_repl": [
        "tf_s3", "s3://kaena-nn-models", "resnet50_uint8_hcv0.pb",
        "%s --executors wave 1 host 0 2 --scheduler wave2 "
        "--schedule_options ' --nname=generic --enable_replication --no_verify ' " # me options
        "--input_node input --images %s "
        "--partition from_multi 'conv1/quantized_conv2d' 'output' " % (rnPreFp32, rnDogJpg),
        "--input_files %s" % rnDogJpg,
        ],
    "7-rn50_uint8_perf_hcv0_b1_wave_repl": [
        "tf_s3", "s3://kaena-nn-models", "resnet50_uint8_hcv0.pb",
        "%s --executors wave 1 host 0 2 --scheduler wave2 "
        "--waive_wavegraph_check "
        "--schedule_options ' --nname=generic --enable_replication --uint8_performance_mode --no_verify ' " # me options
        "--input_node input --images %s "
        "--partition from_multi 'conv1/quantized_conv2d' 'output' " % (rnPreFp32, rnDogJpg),
        "--input_files %s" % rnDogJpg,
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

