# Copyright (C) 2018, Amazon.com. All Rights Reserved
#
# Kaena test configuration
#
# test naming
#   level-name
#   level is relative complexity and runtime cost, easiest 0  to most complex 9
#   Name can be composed of anything - 0 can be used as base/existence test
#   Columns: (nnStruct, nnConfig, nnLabel, nnArgs, rtArgs)

import os

kPath = os.environ.get('KAENA_PATH')
kePath = os.environ.get('KAENA_EXT_PATH')

rnDogJpg = "%s/%s" % (kePath, "images/dog.jpg")
rnCatJpg = "%s/%s" % (kePath, "images/cat.jpeg")
#rnPandaJpg = "%s/%s" % (kePath, "images/cropped_panda.jpg")    # Fails comparison on C5 for some intermediate FP16 ResNet50 layers
rnKoalaJpg = "%s/%s" % (kePath, "images/koala_bear.jpeg")
fp16AccJpg = "%s/%s" % (kePath, "images/3404.jpg")

rnDogCatB2Fp32 = "%s/%s" % (kePath, "images/res50_dog_cat_fp32.npy")

rnBfloat16Npy = "%s/%s" % (kePath, "images/res50_bfp16infp16.npy")

transformerInX = '{0}/images/transformer_x.npy'.format(kePath)
transformerRtInX = 'infer_x:0={0}/images/transformer_x.npy'.format(kePath)

transformerIn = '{0}/images/transformer_x.npy {0}/images/transformer_y.npy'.format(kePath)
transformerRtIn = 'infer_x:0={0}/images/transformer_x.npy infer_y:0={0}/images/transformer_y.npy'.format(kePath)

transformerEncoderIn = '{0}/images/transformer_x.npy'.format(kePath)
transformerEncoderRtIn = 'infer_x:0={0}/images/transformer_x.npy infer_y:0={0}/images/transformer_y.npy'.format(kePath)

def getBatchedJpgs(batchLevel):
    listExtraJpgs = [rnDogJpg, rnCatJpg, rnKoalaJpg] * ((batchLevel+3)//3)
    return ' '.join(tuple(listExtraJpgs[0:batchLevel]))

rnPre = os.path.join(kPath, "compiler/util/res50_preprocessor.py")
incPre = os.path.join(kPath, "compiler/util/inceptv3_preprocessor.py")
rnPost = os.path.join(kPath, "compiler/util/res50_classify")
rnPreFp16 = "--preprocessor {} --preprocessor-args '--data-type fp16' --postprocessor {}".format(rnPre, rnPost)
rnPreFp32 = "--preprocessor {} --preprocessor-args '--data-type fp32' --postprocessor {}".format(rnPre, rnPost)
incPreFp16 = "--preprocessor {} --preprocessor-args '--data-type fp16 --input_height 299 --input_width 299' --postprocessor {}".format(incPre, rnPost)
incPreFp32 = "--preprocessor {} --preprocessor-args '--data-type fp32 --input_height 299 --input_width 299' --postprocessor {}".format(incPre, rnPost)

lstmD0T4 = "%s/%s" % (kePath, "apps/tf/ptb_word_lm/keras_unrolled/data-t4-0.npy")
lstmD0T4B64 = "%s/%s" % (kePath, "apps/tf/ptb_word_lm/keras_unrolled/sigmoid_b64/data-t4-b64-0.npy")
lstmD0T32 = "%s/%s" % (kePath, "apps/tf/ptb_word_lm/keras_unrolled/data-t32-0.npy")

melSpectra = "%s/%s" % (kePath, "apps/tf/parallel_wavenet/example1/melspec_input_fp16.npy")

# ME recipes
def MEv2(optstr):
    optsdec = {"cleanwg": "enable_cleanup", "rn50": "nname=resnet50", "generic": "nname=generic",
        "repl": "enable_replication", "saveall": "save_layer_output", "noverify": "no_verify", "waivewc": "waive_wavegraph_checks",
        "relaxdep" : "relax_dependencies", "fulldep" : "full_dependencies",
    }
    opts = optstr.lower().split("-")
    sched_options, tffe_options = [], []
    for i in opts:
        test_options = tffe_options if i == "waivewc" else sched_options
        if i in optsdec.keys(): i = optsdec[i]
        test_options.append("--" + i)
    return " --scheduler wave2 --schedule_options \' %s \' %s "%(" ".join(sched_options), " ".join(tffe_options))

TFFE_OPTION_IDX = 3
NNE_OPTION_IDX = 4

testConfigMap = {


#  Activation
  "0-rtl-2conv3_relu_h1c1m1r3_wave"      : [ "trivnet_lin", "tfloat16-l2-b1-h1-r3-s1-c1-m1-relu-wmin-0.2-wmax0.24-imin-10000-imax10100", "1conv3", MEv2("Generic")],
  "0-rtl-2conv3_relu_h16c64m64r3_wave"  : [ "trivnet_lin", "tfloat16-l2-b1-h16-r3-s1-c64-m64-relu-wmin0.23-wmax0.24-imin-0.1-imax0.2",   "1conv3", MEv2("Generic")],
  "0-rtl-2conv3_relu_h32c128m128r3_wave" : [ "trivnet_lin", "tfloat16-l2-b1-h32-r3-s1-c128-m128-relu-wmin0.23-wmax0.24-imin-0.1-imax0.2","1conv3", MEv2("Generic")],

# Activation with bias
  "0-rtl-rn50_ba_relu_h1c1m1_fp16_wave"     : [ "trivnet_conv_ba","tfloat16-b1-h1-r1-s1-c1-m1-SAME-relu-wmin-1-wmax1.1-imin-3-imax3.2-amin-3-amax3", "2conv32b", MEv2("Generic")],
  "0-rtl-rn50_ba_relu_h16c128m64_fp16_wave" : [ "trivnet_conv_ba","tfloat16-b1-h16-r1-s1-c128-m64-SAME-relu-wmin-0.01-wmax0.01-imin-0.3-imax0.2-amin-1-amax0.1", "2conv32b", MEv2("Generic")],
  "0-rtl-rn50_ba_relu_h32c256m128_fp16_wave": [ "trivnet_conv_ba","tfloat16-b1-h32-r1-s1-c256-m128-SAME-relu-wmin-0.1-wmax0.2-imin-0.1-imax0.2-amin-0.1-amax0.2", "2conv32b", MEv2("Generic") ],

# MaxPool
  "0-rtl-1conv1maxpool_h1c1m1_wave"        : [ "trivnet_conv_pool","tfloat16-b1-h4-r1-s1-c1-m1-VALID-MaxPool-k2-d2-wmin0.2-wmax2.2-imin-1-imax2", "1conv1pool", MEv2("Generic")],
  "0-rtl-1conv1maxpool_h16c128m64_wave"    : [ "trivnet_conv_pool","tfloat16-b1-h16-r1-s1-c128-m64-VALID-MaxPool-k2-d2-wmin0.1-wmax0.2-imin-1-imax2", "1conv1pool", MEv2("Generic")],
  "0-rtl-1conv1maxpool_h32c256m128_wave"   : [ "trivnet_conv_pool","tfloat16-b1-h32-r1-s1-c256-m128-VALID-MaxPool-k2-d2-wmin0.3-wmax0.4-imin-0.1-imax0.1", "1conv1pool", MEv2("Generic")],

# AvgPool
  "0-rtl-1conv1avgpool_h4c1m1_same_wave"     : [ "trivnet_conv_pool","tfloat16-b1-h4-r1-s1-c1-m1-SAME-AvgPool-k2-d2-wmin2-wmax2.2-imin1-imax16", "1conv1pool", MEv2("Generic")],
  "0-rtl-1conv1avgpool_h16c128m64_same_wave" : [ "trivnet_conv_pool","tfloat16-b1-h16-r1-s1-c128-m64-SAME-AvgPool-k2-d2-wmin2.1-wmax2.2-imin-0.1-imax2", "1conv1pool", MEv2("Generic")],
  "0-rtl-1conv1avgpool_h32c256m128_same_wave": [ "trivnet_conv_pool","tfloat16-b1-h32-r1-s1-c256-m128-SAME-AvgPool-k2-d2-wmin0.1-wmax0.2-imin-0.1-imax0.2", "1conv1pool", MEv2("Generic")],

# Bias -> ResAdd
  "0-rtl-resadd_h1c1_wave"         : [ "trivnet_add",    "tfloat16-b1-h1-c1-wmin2-wmax2.2-imin3-imax6", "add", MEv2("Generic")],
  "0-rtl-resadd_h16c128_wave"      : [ "trivnet_add",    "tfloat16-b1-h16-c128-wmin2.19-wmax2.2-imin-0.1-imax0.2", "add", MEv2("Generic")],
  "0-rtl-resadd_h32c256_wave"      : [ "trivnet_add",    "tfloat16-b1-h32-c256-wmin2.19-wmax2.2-imin-0.1-imax0.2", "add", MEv2("Generic")],

# Batching
  "0-1conv1maxpool_b4h1c128m1_wave"        : [ "trivnet_conv_pool","tfloat16-b4-h1-r1-s1-c128-m1-VALID-MaxPool-k1-d1-wmin0.2-wmax2.2-imin-1-imax2", "1conv1pool", MEv2("Generic")],

# Multiple convolves
  "0-4conv_multiout_wave"      : [
    "trivnet_lin",
    "tfloat16-l4-b1-h4-r3-s1-c1-m1-relu-wmin-0.2-wmax0.4-imin-1000-imax1010",
    "10cr",
    ( " --scheduler qemu_wave2  "
     + " --wavegraph_checks structure "   ## With extra Saves, data races are present.
     + " --schedule_options ' --save_layer_output ' "
    )
  ],

  "0-rtl-10conv_h4c1m1_relu_wave"      : [
    "trivnet_lin",
    "tfloat16-l10-b1-h4-r3-s1-c1-m1-relu-wmin-0.2-wmax0.4-imin-1000-imax1010",
    "10cr",
    MEv2("Generic")
  ],

  "0-rtl-10conv_h16c128m64_relu_wave"  : [ "trivnet_lin",    "tfloat16-l10-b1-h4-r3-s1-c1-m1-relu-wmin-0.39-wmax0.4-imin-0.1-imax0.2", "10cr", MEv2("Generic")],
  "0-rtl-10conv_h32c256m128_relu_wave" : [ "trivnet_lin",    "tfloat16-l10-b1-h4-r3-s1-c1-m1-relu-wmin-0.02-wmax0.4-imin-0.1-imax0.2", "10cr", MEv2("Generic")],
 ########### ########### ########### ########### ########### ########### ########### ###########
  "0-rtl-1conv_wave"                   : [ "trivnet_conv1",    "tfloat16-b1-h1-r1-s1-c1-m1-wmin2-wmax2.2-imin3-imax3.2",         "1conv" , MEv2("Generic")],
  "0-rtl-1conv_m8_wave"                : [ "trivnet_conv1",    "tfloat16-b1-h1-r1-s1-c1-m8-wmin2-wmax2.2-imin3-imax3.2",         "1conv" , MEv2("Generic")],
  "0-rtl-1conv_h2c16_wave"             : [ "trivnet_conv1",    "tfloat16-b1-h2-r1-s1-c16-m1-wmin2-wmax2.2-imin1-imax7",          "1conv" , MEv2("Generic")],
  "0-rtl-1conv_h4r2c8m4_wave"          : [ "trivnet_conv1",    "tfloat16-b1-h4-r2-s1-c8-m4-wmin-0.1-wmax0.12-imin-0.2-imax0.25", "1conv" , MEv2("Generic")],
  "0-rtl-1conv_h4r3c4m2_wave"          : [ "trivnet_conv1",    "tfloat16-b1-h4-r3-s1-c4-m2-wmin0-wmax9-imin0-imax15",            "1conv" , MEv2("Generic")],
  "0-rtl-1conv_relu_wave"              : [ "trivnet_lin",      "tfloat16-l2-b1-h4-r3-s1-c1-m1-relu-wmin-0.2-wmax0.4-imin-1000-imax1010", "1cr", MEv2("Generic")],
  "0-rtl-1conv_h8r4c64m16_wave"        : [ "trivnet_conv1",    "tfloat16-b1-h8-r4-s1-c64-m16-wmin0-wmax9-imin0-imax15",          "1conv" , MEv2("Generic")],
  "0-rtl-1conv3_h4r3_relu_wave"        : [ "trivnet_lin",      "tfloat16-l2-b1-h4-r3-s1-c1-m1-relu-wmin-0.2-wmax0.24-imin-10000-imax10100", "1conv3", MEv2("Generic")],
  "0-rtl-act_h2c16_wave"               : [ "trivnet_act",      "tfloat16-b1-h2-c16-tanh-wmin2-wmax2.2-imin-1-imax2",             "act", MEv2("Generic")],

  # TODO: https://sim.amazon.com/issues/kaena-773
  "0-rtl-act_tanh_sweep_wave"          : [ "trivnet_act",      "tfloat32-b1-h128-c128-tanh-wmin2-wmax2.2-imin-10000000-imax10000000",          "act", MEv2("Generic")],
  "0-rtl-act_sigmoid_sweep_wave"       : [ "trivnet_act",      "tfloat32-b1-h128-c64-sigmoid-wmin2-wmax2.2-imin-10000000-imax10000000",          "act", MEv2("Generic")],
  "0-rtl-act_relu_sweep_wave"          : [ "trivnet_act",      "tfloat32-b1-h128-c64-relu-wmin2-wmax2.2-imin-10000000-imax10000000",          "act", MEv2("Generic")],
  "0-rtl-act_identity_sweep_wave"      : [ "trivnet_biasadd",      "tfloat32-b1-h128-c64-wmin2-wmax2.2-imin-10000000-imax10000000",          "act", MEv2("Generic")],

  "0-act_exp_sweep_wave"           : [ "trivnet_act", "tfloat32-b1-h128-c64-exp-wmin2-wmax2.2-imin-5-imax5",                         "act", MEv2("Generic")],
  "0-act_lrelu_sweep_wave"         : [ "trivnet_act", "tfloat32-b1-h80-c64-lrelu-wmin2-wmax2.2-imin-10000000-imax10000000",          "act", MEv2("Generic")],
  "0-act_fused_lrelu_sweep_wave"         : [ "trivnet_act",      "tfloat32-b1-h80-c64-lrelu-wmin2-wmax2.2-imin-10000000-imax10000000",  "act", "--scheduler wave2  --schedule_options ' --nname=generic --fuse_lrelu ' "],

  "0-rtl-act_tanh_minisweep_wave"          : [ "trivnet_act",      "tfloat32-b1-h128-c64-tanh-wmin2-wmax2.2-imin-1-imax1",          "act", MEv2("Generic")],
  "0-rtl-act_sigmoid_minisweep_wave"       : [ "trivnet_act",      "tfloat32-b1-h128-c64-sigmoid-wmin2-wmax2.2-imin-1-imax1",          "act", MEv2("Generic")],
  "0-rtl-act_relu_minisweep_wave"          : [ "trivnet_act",      "tfloat32-b1-h128-c64-relu-wmin2-wmax2.2-imin-1-imax1",          "act", MEv2("Generic")],
  "0-rtl-act_identity_minisweep_wave"      : [ "trivnet_biasadd",      "tfloat32-b1-h128-c64-wmin2-wmax2.2-imin-1-imax1",          "act", MEv2("Generic")],

  "0-act_exp_minisweep_wave"           : [ "trivnet_act",      "tfloat32-b1-h128-c64-exp-wmin2-wmax2.2-imin-1-imin-1-imax1",          "act", MEv2("Generic")],
  "0-act_lrelu_minisweep_wave"         : [ "trivnet_act", "tfloat32-b1-h128-c64-lrelu-wmin2-wmax2.2-imin-1-imax1", "act", MEv2("Generic")],
  "0-act_fused_lrelu_minisweep_wave"         : [ "trivnet_act",      "tfloat32-b1-h128-c64-lrelu-wmin2-wmax2.2-imin-1-imax1",          "act", "--scheduler wave2  --schedule_options ' --nname=generic --fuse_lrelu '"],

  "0-1clipbyvalue_wave" : [ "trivnet_clipbyvalue",
    "tfloat16-b1-h4-r1-s1-c1-m1-wmin2-wmax2.2-imin0-imax3.2-xmin1-xmax3", "1conv",
    ("--scheduler wave2 "
    + " --partition from 1conv/output"
    )
  ],
  "0-1slice_w_wave" : [ "trivnet_slice_w",  "tfloat16-b1-h100-r1-s1-c1-m1-wmin2-wmax2.2-imin0-imax3.2-sbegin16-ssize50", "1conv", MEv2("Generic")],
  "0-1slice_h_wave" : [ "trivnet_slice_h",  "tfloat16-b1-h100-r1-s1-c1-m1-wmin2-wmax2.2-imin0-imax3.2-sbegin16-ssize50", "1conv", MEv2("Generic")],
  "0-1conv_dilated_wave" : [ "trivnet_conv_dilated",  "tfloat16-b1-h8-r3-s1-c1-m1-d2-wmin-0.1-wmax0.2-imin-0.1-imax0.2", "1conv", MEv2("Generic")],
  "0-1conv_dilated_h32_wave" : [ "trivnet_conv_dilated",  "tfloat16-b1-h32-r3-s1-c1-m1-d2-wmin-0.1-wmax0.2-imin-0.1-imax0.2", "1conv", MEv2("Generic")],
  "0-1conv_dilated_1d_h32_wave" : [ "trivnet_conv_dilated_1d",  "tfloat16-b1-h32-r3-s1-c1-m1-d2-wmin-0.1-wmax0.2-imin-0.1-imax0.2", "1conv", MEv2("Generic")],
  "0-1conv_dilated_1d_h32r3c64m64d2_wave" : [ "trivnet_conv_dilated_1d",  "tfloat16-b1-h32-r3-s1-c64-m64-d2-wmin-0.1-wmax0.2-imin-0.1-imax0.2", "1conv", MEv2("Generic")],
  "0-1conv_dilated_1d_h1536r3c64m64d2_wave" : [ "trivnet_conv_dilated_1d",  "tfloat16-b1-h1536-r3-s1-c64-m64-d2-wmin-0.1-wmax0.2-imin-0.1-imax0.2", "1conv", MEv2("Generic")],
  "0-1conv_evict_host" : [ "trivnet_evict",  "tfloat16-b1-h340-r3-s2-c1-m1-d1-wmin-0.1-wmax0.2-imin-0.1-imax0.2", "1conv", "--executors host all --scheduler wave2 --schedule_options ' --nname=generic --enable_eviction' "],
  "0-1conv_evict_wave" : [ "trivnet_evict",  "tfloat16-b1-h340-r3-s2-c1-m1-d1-wmin-0.1-wmax0.2-imin-0.1-imax0.2", "1conv", "--scheduler wave2 --schedule_options ' --nname=generic --enable_eviction' "],
  "0-1reshape_wave" : [ "trivnet_reshape",  "tfloat16-b1-h32-r1-s1-c1-m1-wmin2-wmax2.2-imin0-imax3.2-xmin1-xmax3", "1conv", MEv2("Generic")],
  "0-1squeeze_wave" : [ "trivnet_squeeze",  "tfloat16-b1-h32-r1-s1-c1-m1-wmin2-wmax2.2-imin0-imax3.2-xmin1-xmax3", "1conv", MEv2("Generic")],
  "0-1expanddims_wave" : [ "trivnet_expanddims",  "tfloat16-b1-h32-r1-s1-c1-m1-wmin2-wmax2.2-imin0-imax3.2-xmin1-xmax3", "1conv", MEv2("Generic")],
  "0-1transpose_wave" : [ "trivnet_transpose",  "tfloat16-b1-h4-r1-s1-c1-m1-wmin2-wmax2.2-imin0-imax3.2-xmin1-xmax3", "1conv", MEv2("Generic")],
  "0-1stridedslice_tanh_sigmoid_wave" : [ "trivnet_stridedslice_tanh_sigmoid",  "tfloat16-b1-h4-r1-s1-c2-m1-wmin2-wmax2.2-imin0-imax3.2-xmin1-xmax3", "1conv", MEv2("Generic")],
  "0-1stridedslice_wave" : [ "trivnet_stridedslice",  "tfloat16-b1-h4-r1-s1-c2-m1-wmin2-wmax2.2-imin0-imax3.2-xmin1-xmax3", "1conv", MEv2("Generic")],

  "0-1conv0_wave" : [ "trivnet_conv1",  "tfloat16-b1-h1-r1-s1-c1-m1-wmin2-wmax2.2-imin3-imax3.2", "1conv", MEv2("Generic")],
  "0-1conv0_fp32_wave" : [ "trivnet_conv1",  "tfloat32-b1-h1-r1-s1-c1-m1-wmin2-wmax2.2-imin3-imax3.2", "1conv", MEv2("Generic")],
  "0-1conv0_subnormal_wave" : [ "trivnet_conv1",
      "tfloat16-b1-h1-r1-s1-c1-m1-wmin0.0000022-wmax0.0000022-imin33.3-imax33.3",
      "1conv", MEv2("Generic")],

  "0-1conv0_wave_h35c288m64" : [ "trivnet_conv1",  "tfloat16-b1-h35-r1-s1-c288-m64-wmin0.1-wmax0.2-imin0.2-imax0.3", "1conv", MEv2("Generic")],
  "0-1conv0_ckpt_wave" : [ "ckpt_conv1",  "tfloat16-b1-h1-r1-s1-c1-m1-wmin2-wmax2.2-imin3-imax3.2", "1conv", "--scheduler wave2 --wavegraph_checks structure data-race --show_op_name_in_kgraph --exclude_ops_from_capture 'save|Save|restore' --debug 1"],
  "0-1conv0_qemu_wave" : [ "trivnet_conv1",  "tfloat16-b1-h1-r1-s1-c1-m1-wmin2-wmax2.2-imin3-imax3.2", "1conv", "--scheduler qemu_wave --wavegraph_checks structure data-race"],
  "0-1conv0_qemu_sem_wave" : [ "trivnet_conv1",  "tfloat16-b1-h1-r1-s1-c1-m1-wmin2-wmax2.2-imin3-imax3.2", "1conv",
      ( "--scheduler qemu_wave2 "
      + " --wavegraph_checks structure data-race "
      + " --be_options sync-with-semaphores "
      )
  ],
  "0-1conv0_b16_wave" : [ "trivnet_conv1",  "tfloat16-b16-h1-r1-s1-c1-m1-wmin2-wmax2.2-imin3-imax3.2", "1conv", MEv2("RN50")],
  "0-1conv0m4_wave" : [ "trivnet_conv1",  "tfloat16-b1-h1-r1-s1-c1-m4-wmin2-wmax2.2-imin3-imax3.2", "1conv", MEv2("Generic")],
  "0-1conv0m8_wave" : [ "trivnet_conv1",  "tfloat16-b1-h1-r1-s1-c1-m8-wmin2-wmax2.2-imin3-imax3.2", "1conv", MEv2("Generic")],
  "0-1conv0m16_wave" : [ "trivnet_conv1",  "tfloat16-b1-h1-r1-s1-c1-m16-wmin2-wmax2.2-imin3-imax3.2", "1conv", MEv2("Generic")],
  "0-1conv0_padvalid_wave" : [ "trivnet_conv1_padvalid",  "tfloat16-b1-h229-r7-s2-c1-m1-wmin2-wmax2.2-imin3-imax3.2", "1conv", MEv2("Generic")],
  "0-1conv0_h16r2s2_wave"      : [ "trivnet_conv1",  "tfloat16-b1-h16-r2-s2-c1-m1-wmin-2-wmax2.2-imin-3-imax3.3", "1conv", MEv2("Generic")],
  "0-1conv0_h16r3s2_wave"      : [ "trivnet_conv1",  "tfloat16-b1-h16-r3-s2-c1-m1-wmin-2-wmax2.2-imin-3-imax3.3", "1conv", MEv2("Generic")],
  "0-1conv0_c1h2_wave" : [ "trivnet_conv1",  "tfloat16-b1-h2-r1-s1-c1-m1-wmin2-wmax2.2-imin1-imax7", "1conv", MEv2("Generic")],
  "0-1conv0_c1h16_wave" : [ "trivnet_conv1",  "tfloat16-b1-h16-r1-s1-c1-m1-wmin2-wmax2.2-imin3-imax3.2", "1conv", MEv2("Generic")],
  "0-neg"         : [ "trivnet_conv2",  "b1-Zh1-r1-s1-c1-m1-wmin2-wmax3-imin5-imax5.5", "I_ALWAYS_FAIL"],
  "0-1conv_tile_wave"  : [ "trivnet_conv1",  "tfloat16-b1-h35-r3-s1-c1-m1-wmin-0.1-wmax0.12-imin-0.2-imax0.25", "1conv", MEv2("Generic")],

  # Sealife existence tests
  # Float16
  "0-1conv_h16r1c128m64_wave"    : [ "trivnet_conv1","tfloat16-b1-h16-r1-s1-c128-m64-VALID-wmin0.1-wmax0.2-imin-1-imax2", "1conv", MEv2("Generic")],
  "0-1conv_h16r3c128m64_wave"    : [ "trivnet_conv1","tfloat16-b1-h16-r3-s1-c128-m64-VALID-wmin0.1-wmax0.2-imin-1-imax2", "1conv", MEv2("Generic")],
  "0-1avgpool_h16c128m64k1d1_valid_wave"  : [ "trivnet_pool", "tfloat16-b1-h16-r1-s1-c128-m64-VALID-AvgPool-k1-d1-wmin-0.1-wmax0.2-imin-0.2-imax0.3", "1pool", MEv2("Generic")],
  "0-1avgpool_h16c128m64k3d1_valid_wave"  : [ "trivnet_pool", "tfloat16-b1-h16-r1-s1-c128-m64-VALID-AvgPool-k3-d1-wmin-0.1-wmax0.2-imin-0.2-imax0.3", "1pool", MEv2("Generic")],
  # Float32
  "0-1conv_h16r1c128m64_fp32_wave"    : [ "trivnet_conv1","tfloat32-b1-h16-r1-s1-c128-m64-VALID-wmin0.1-wmax0.2-imin-1-imax2", "1conv", MEv2("Generic")],
  "0-1conv_h16r3c128m64_fp32_wave"    : [ "trivnet_conv1","tfloat32-b1-h16-r3-s1-c128-m64-VALID-wmin0.1-wmax0.2-imin-1-imax2", "1conv", MEv2("Generic")],
  "0-1avgpool_h16c128m64k1d1_valid_fp32_wave"  : [ "trivnet_pool", "tfloat32-b1-h16-r1-s1-c128-m64-VALID-AvgPool-k1-d1-wmin-0.1-wmax0.2-imin-0.2-imax0.3", "1pool", MEv2("Generic")],
  "0-1avgpool_h16c128m64k3d1_valid_fp32_wave"  : [ "trivnet_pool", "tfloat32-b1-h16-r1-s1-c128-m64-VALID-AvgPool-k3-d1-wmin-0.1-wmax0.2-imin-0.2-imax0.3", "1pool", MEv2("Generic")],

  "0-1conv_h4r1_wave"  : [ "trivnet_conv1",  "tfloat16-b1-h4-r1-s1-c1-m1-wmin-0.1-wmax0.12-imin-0.2-imax0.25", "1conv", MEv2("Generic")],
  "0-1conv_h4r1_b2_wave"  : [ "trivnet_conv1",  "tfloat16-b2-h4-r1-s1-c1-m1-wmin-0.1-wmax0.12-imin-0.2-imax0.25", "1conv", MEv2("RN50")],
  "0-1conv_b1h1r1s1c2m2_tile_wave"  : [ "trivnet_conv1",  "tfloat16-b1-h1-r1-s1-c2-m2-wmin-0.1-wmax0.12-imin-0.2-imax0.25", "1conv", MEv2("Generic")],
  "0-1conv_h4r2s2_wave"  : [ "trivnet_conv1",  "tfloat16-b1-h4-r2-s2-c1-m1-wmin-0.1-wmax0.12-imin-0.2-imax0.25", "1conv", MEv2("Generic")],
  "0-1conv_h6r2s3_wave"  : [ "trivnet_conv1",  "tfloat16-b1-h6-r2-s3-c1-m1-wmin-0.1-wmax0.12-imin-0.2-imax0.25", "1conv", MEv2("Generic")],
  "0-1conv_h6r3s2_wave"  : [ "trivnet_conv1",  "tfloat16-b1-h6-r3-s2-c1-m1-wmin-0.1-wmax0.12-imin-0.2-imax0.25", "1conv", MEv2("Generic")],
  "0-1conv_h6r3s2_b2_wave"  : [ "trivnet_conv1",  "tfloat16-b2-h6-r3-s2-c1-m1-wmin-0.1-wmax0.12-imin-0.2-imax0.25", "1conv", MEv2("RN50")],
  "0-1conv_h4r3s1_wave"  : [ "trivnet_conv1",  "tfloat16-b1-h4-r3-s1-c1-m1-wmin0-wmax9-imin0-imax15", "1conv", MEv2("Generic")],
  "0-1conv_h4r3s1_b2_wave"  : [ "trivnet_conv1",  "tfloat16-b2-h4-r3-s1-c1-m1-wmin0-wmax9-imin0-imax15", "1conv", MEv2("RN50")],

  "0-1conv_tile_r1_wave"  : [ "trivnet_conv1",  "tfloat16-b1-h35-r1-s1-c1-m1-wmin-0.1-wmax0.12-imin-0.2-imax0.25", "1conv", MEv2("Generic")],
  "0-1conv_tile_r1h32_wave"  : [ "trivnet_conv1",  "tfloat16-b1-h32-r1-s1-c1-m1-wmin-0.1-wmax0.12-imin-0.2-imax0.25", "1conv", MEv2("Generic")],
  "0-1conv_tile_r1_e1_wave"  : [ "trivnet_conv1",  "tfloat16-b1-h35-r1-s1-c1-m1-F_31_31=3-wmin2-wmax2-imin-0-imax0", "1conv", MEv2("Generic")],
  #"0-2conv3_relu" : [ "trivnet_lin",    "tfloat16-l2-b1-h4-r3-s1-c1-m1-relu-wmin-0.2-wmax0.24-imin-10000-imax10100", "1conv3"],
  "0-3conv_1concat_wave" : [ "trivnet_concat2",  "tfloat16-b1-h1-r1-s1-c1-m1-wmin2-wmax2.2-imin3-imax3.2", "1concat", " --partition from 1concat/i3 " + MEv2("Generic")],
  "0-3conv_1concat" : [ "trivnet_concat2",  "tfloat16-b1-h1-r1-s1-c1-m1-wmin2-wmax2.2-imin3-imax3.2", "1concat", MEv2("Generic")],
  "0-3conv_1concat_c32m32" : [ "trivnet_concat2",  "tfloat16-b1-h1-r1-s1-c32-m32-wmin2-wmax2.2-imin3-imax3.2", "1concat", MEv2("Generic")],
  "0-3conv_1concat_h16c32m32" : [ "trivnet_concat2",  "tfloat16-b1-h16-r1-s1-c32-m32-wmin2-wmax2.2-imin3-imax3.2", "1concat", MEv2("Generic")],
  "0-3conv_1concat_h16c32m63" : [ "trivnet_concat2",  "tfloat16-b1-h16-r1-s1-c32-m63-wmin2-wmax2.2-imin3-imax3.2", "1concat", MEv2("Generic")],
  "0-1concat_h1c1m1ni2" : [ "trivnet_concat_variable_inputs",  "tfloat16-b1-h1-r1-s1-c1-m1-ni2-wmin2-wmax2.2-imin3-imax3.2", "1concat", MEv2("Generic")],
  "0-1concat_h17c1m1ni2" : [ "trivnet_concat_variable_inputs",  "tfloat16-b1-h17-r1-s1-c1-m1-ni2-wmin2-wmax2.2-imin3-imax3.2", "1concat", MEv2("Generic")],
  "0-1concat_h1c1m10ni5" : [ "trivnet_concat_variable_inputs",  "tfloat16-b1-h1-r1-s1-c1-m10-ni5-wmin2-wmax2.2-imin3-imax3.2", "1concat", MEv2("Generic")],
  "0-1concat_h16c63m127ni5" : [ "trivnet_concat_variable_inputs",  "tfloat16-b1-h16-r1-s1-c63-m127-ni5-wmin2-wmax2.2-imin3-imax3.2", "1concat", MEv2("Generic")],
  "0-1concat_h35c1m1ni2" : [ "trivnet_concat_variable_inputs",  "tfloat16-b1-h35-r1-s1-c1-m1-ni2-wmin2-wmax2.2-imin3-imax3.2", "1concat", MEv2("Generic")],
  "0-1concat_h35c64m64ni4" : [ "trivnet_concat_variable_inputs",  "tfloat16-b1-h35-r1-s1-c64-m64-ni4-wmin2-wmax2.2-imin3-imax3.2", "1concat", MEv2("Generic")],
  "0-1concat_h35c63m127ni5" : [ "trivnet_concat_variable_inputs",  "tfloat16-b1-h35-r1-s1-c63-m127-ni5-wmin2-wmax2.2-imin3-imax3.2", "1concat", MEv2("Generic")],

  "0-1concat_h16c63m127ni5_sem_qemu" : [
    "trivnet_concat_variable_inputs",
    "tfloat16-b1-h16-r1-s1-c63-m127-ni5-wmin2-wmax2.2-imin3-imax3.2",
    "1concat",
    ("--scheduler wave2 "
    + " --schedule_options ' --nname=generic --save_layer_output' "
    + " --waive_wavegraph_checks"
    + " --be_options sync-with-semaphores "
    )
  ],

  "0-2conv3_relu_wave" : [ "trivnet_lin",    "tfloat16-l2-b1-h4-r3-s1-c1-m1-relu-wmin-0.2-wmax0.24-imin-10000-imax10100", "1conv3", MEv2("Generic")],
  "0-2conv3_relu_b16_wave" : [ "trivnet_lin",    "tfloat16-l2-b16-h4-r3-s1-c1-m1-relu-wmin-0.2-wmax0.24-imin-10000-imax10100", "1conv3", MEv2("RN50")],
  "3-rn50_relu_fp16_wave"  : [ "trivnet_lin","tfloat16-l2-b1-h224-r7-s2-c3-m3-relu-wmin-1-wmax1.1-imin-3-imax3.2", "2conv32b", MEv2("Generic")],
  "3-rn50_ba_relu_fp16_wave"  : [ "trivnet_conv_ba","tfloat16-b1-h224-r7-s2-c3-m64-SAME-relu-wmin-1-wmax1.1-imin-3-imax3.2-amin-3-amax3", "2conv32b", MEv2("Generic")],
  "0-ba_relu_fp32_wave"  : [ "trivnet_conv_ba","tfloat32-b1-h1-r1-s1-c1-m1-SAME-relu-wmin-2-wmax2-imin3-imax10-amin-7-amax7", "2conv32b", MEv2("Generic")],
  "0-3conv_relu_wave" : [ "trivnet_lin",    "tfloat16-l3-b1-h1-r1-s1-c1-m1-relu-wmin-0.2-wmax0.4-imin-1000-imax1010", "10cr", MEv2("Generic")],
  #"0-116conv_tanh" : [ "trivnet_lin",   "tfloat16-l116-b1-h4-r3-s1-c1-m1-tanh-wmin-0.2-wmax0.8-imin-4-imax8", "116ct"],
  "0-1conv_h4_softplus_wave" : [ "trivnet_lin",    "tfloat16-l2-b1-h4-r1-s1-c1-m1-softplus-wmin-0.2-wmax0.4-imin-1-imax1.2", "10cr", MEv2("Generic")],
  "0-1conv_h4_sigmoid_wave" : [ "trivnet_lin",    "tfloat16-l2-b1-h4-r1-s1-c1-m1-sigmoid-wmin-0.2-wmax0.4-imin-1000-imax1010", "10cr", MEv2("Generic")],
  "0-116conv_tanh_wave" : [ "trivnet_lin",   "tfloat16-l116-b1-h4-r3-s1-c1-m1-tanh-wmin-0.2-wmax0.8-imin-4-imax8", "116ct", MEv2("Generic")],

  "0-116conv_tanh_sem_qemu_wave" : [ "trivnet_lin",   "tfloat16-l116-b1-h4-r3-s1-c1-m1-tanh-wmin-0.2-wmax0.8-imin-4-imax8", "116ct",
      ( "--scheduler qemu_wave2 "
      + " --schedule_options ' --nname=generic ' "
      + " --be_options sync-with-semaphores "
      )
  ],

  "0-300conv_tanh_wave-all-layers" : [ "trivnet_lin", "tfloat16-l300-b1-h4-r3-s1-c1-m1-tanh-wmin-0.2-wmax0.8-imin-4-imax8", "300ct", MEv2("Generic-SaveAll")+" --wavegraph_transform cleaner"],

  "0-1conv_s8_wave"    : [ "trivnet_conv1",  "tfloat16-b1-h16-r1-s8-c1-m1-wmin2-wmax22-imin1-imax256", "1conv", MEv2("Generic")],
  "0-1mp_r3s2_16_wave"  : [ "trivnet_mp1", "b1-h16-r3-s2-c1-m1-wmin0-wmax0.1-imin1-imax12544", "1mp", MEv2("Generic")],
  "0-1conv1pool_wave"  : [ "trivnet_conv_pool", "tfloat16-b1-h4-r1-s1-c1-m128-SAME-MaxPool-k2-d2-wmin1-wmax1-imin0-imax127", "1conv1pool", MEv2("Generic")],
  "0-1conv1avgpool_wave"  : [ "trivnet_conv_pool", "tfloat16-b1-h4-r1-s1-c1-m1-SAME-AvgPool-k2-d2-wmin2-wmax2.2-imin1-imax16", "1conv1pool", MEv2("Generic")],
  "0-1conv1pool_b5_wave" : [ "trivnet_conv_pool", "tfloat16-b5-h4-r1-s1-c1-m1-SAME-AvgPool-k2-d2-wmin2-wmax2.2-imin1-imax80", "1conv1pool", MEv2("Generic")],
  "0-1conv1pool_b5m3_wave" : [ "trivnet_conv_pool", "tfloat16-b5-h4-r1-s1-c1-m3-SAME-AvgPool-k2-d2-wmin2-wmax2.2-imin1-imax240", "1conv1pool", MEv2("Generic")],
  "0-1conv1maxpool_k3d2_wave"  : [ "trivnet_conv_pool", "tfloat16-b16-h1-r3-s2-c1-m1-SAME-MaxPool-k3-d2-wmin-0.2-wmax0.3-imin-0.2-imax0.3", "1conv1pool", MEv2("Generic")],

  # Multi-threading on Multi-TPBs
  "0-1conv1avgpool_wave_2tpbs"  : [ "trivnet_conv_pool", "tfloat16-b1-h4-r1-s1-c1-m1-SAME-AvgPool-k2-d2-wmin2-wmax2.2-imin1-imax16", "1conv1pool", "--scheduler wave2 --partition from 1conv1pool/i2 1conv1pool/output --executor wave 0 1"],

  # Conv, BiasAdd

  "0-1conv1ba1_h1c1m1_wave"  : [ "trivnet_conv_ba", "tfloat16-b1-h1-r1-s1-c1-m1-SAME-wmin2-wmax2.2-imin1-imax1.6-amin3-amax3.2", "1conv1ba", MEv2("Generic")],

  "0-1conv1ba1_h4c1m256_wave"  : [ "trivnet_conv_ba", "tfloat16-b1-h4-r1-s1-c1-m256-SAME-wmin2-wmax2.2-imin1-imax1.6-amin3-amax3.2", "1conv1ba", MEv2("Generic")],


  "0-1conv1ba1_wave"  : [ "trivnet_conv_ba", "tfloat16-b1-h4-r1-s1-c1-m256-SAME-wmin2-wmax2.2-imin1-imax16-amin3-amax3.2", "1conv1ba", MEv2("Generic")],
  "0-1conv1ba1_h4c1m1_fp32_wave"  : [ "trivnet_conv_ba", "tfloat32-b1-h4-r1-s1-c1-m1-SAME-wmin2-wmax2.2-imin1-imax16-amin3-amax3.2", "1conv1ba", MEv2("Generic")],
  "0-1conv1ba1_h4c2m2_fp32_wave"  : [ "trivnet_conv_ba", "tfloat32-b1-h4-r1-s1-c2-m2-SAME-wmin2-wmax2.2-imin1-imax16-amin3-amax3.2", "1conv1ba", MEv2("Generic")],
  "0-1conv1ba1_h55c1m1_fp32_wave"  : [ "trivnet_conv_ba", "tfloat32-b1-h55-r1-s1-c1-m1-SAME-wmin2-wmax2.2-imin1-imax16-amin3-amax3.2", "1conv1ba", MEv2("Generic")],
  "0-1conv1ba1_h1c64m64_fp32_wave"  : [ "trivnet_conv_ba", "tfloat32-b1-h1-r1-s1-c64-m64-SAME-wmin2-wmax2.2-imin1-imax16-amin3-amax3.2", "1conv1ba", MEv2("Generic")],
  "0-1conv1ba1_h4c2m2_wave"  : [ "trivnet_conv_ba", "tfloat16-b1-h4-r1-s1-c2-m2-SAME-wmin2-wmax2.2-imin1-imax16-amin3-amax3.2", "1conv1ba", MEv2("Generic")],
  "0-1conv1ba1_h55c1m1_wave"  : [ "trivnet_conv_ba", "tfloat16-b1-h55-r1-s1-c1-m1-SAME-wmin2-wmax2.2-imin1-imax16-amin3-amax3.2", "1conv1ba", MEv2("Generic")],
  "0-1conv1ba1_h55c1m1_b2_wave"  : [ "trivnet_conv_ba", "tfloat16-b2-h55-r1-s1-c1-m1-SAME-wmin2-wmax2.2-imin1-imax16-amin3-amax3.2", "1conv1ba", MEv2("RN50")],
  "0-1conv1ba1_h1c64m64_wave"  : [ "trivnet_conv_ba", "tfloat16-b1-h1-r1-s1-c64-m64-SAME-wmin2-wmax2.2-imin1-imax16-amin3-amax3.2", "1conv1ba", MEv2("Generic")],
  "0-1conv1ba1_fp32_wave"  : [ "trivnet_conv_ba", "tfloat32-b1-h224-r7-s2-c3-m64-SAME-wmin1-wmax1-imin0-imax50175-amin-20000-amax-20000", "1conv1ba", MEv2("Generic")],

  "0-act_wave"     : [ "trivnet_act", "tfloat16-b1-h2-c128-tanh-wmin2-wmax2.2-imin-1-imax2", "act", MEv2("Generic")],
  "0-act_softplus_wave"     : [ "trivnet_act", "tfloat16-b1-h2-c128-softplus-wmin2-wmax2.2-imin-1-imax2", "act", MEv2("Generic")],
  "0-scaleadd_wave"       : [ "trivnet_scaleadd", "tfloat16-b1-h1-c16-wmin2-wmax2.2-amin3-amax3.5-imin3-imax6-xmin2-xmax5", "scaleadd", MEv2("Generic")],
  "0-resadd_wave"         : [ "trivnet_add",    "tfloat16-b1-h2-c1-wmin2-wmax2.2-imin3-imax6", "add", MEv2("Generic")],
  "0-resadd_fp32_wave"    : [ "trivnet_add",    "tfloat32-b1-h17-c4-wmin-0.1-wmax0.11-imin1-imax5", "add", MEv2("Generic")],
  "0-resadd_uint8_wave"   : [ "trivnet_add",    "tuint8-b1-h4-c3-wmin1-wmax4-imin5-imax53", "add", MEv2("Generic")],
  "0-resadd_2in_wave"    : [ "trivnet_add_2in",    "tfloat16-b1-h2-c1-wmin2-wmax2.2-imin3-imax6", "add", MEv2("Generic")],
  "0-subtract_psum_sb_wave"    : [ "trivnet_add",    "tfloat16-b1-h2-c1-SUB-wmin2-wmax2.2-imin3-imax6", "add", MEv2("Generic")],

  "0-3resadd_fp16_wave"  : [ "trivnet_conv_ba_add",
    "tfloat16-b1-h4-r1-s1-c1-m1-SAME-wmin-1-wmax2-imin-0.1-imax0.3-amin-0.01-amax-0.03",
    "add", "--partition from add/i3 add/output --executor host 0 wave 1 " + MEv2("Generic")
    ],

  "0-3conv_ba_resadd_fp32_wave"  : [ "trivnet_conv_ba_add", "tfloat32-b1-h55-r3-s1-c1-m1-SAME-wmin-1-wmax2-imin-0.1-imax0.3-amin-0.01-amax-0.03", "add", MEv2("Generic")],
  "0-3conv_ba_resadd_fp16_wave"  : [ "trivnet_conv_ba_add", "tfloat16-b1-h55-r3-s1-c1-m1-SAME-wmin-1-wmax2-imin-0.1-imax0.3-amin-0.01-amax-0.03", "add", MEv2("Generic")],
  "0-3conv_ba_resadd_h1_fp16_wave"  : [ "trivnet_conv_ba_add", "tfloat16-b1-h1-r1-s1-c1-m1-SAME-wmin-1-wmax2-imin-0.1-imax0.3-amin-0.01-amax-0.03", "add", MEv2("Generic")],
  "0-3conv_ba_mult_fp32_wave"  : [ "trivnet_conv_ba_mult", "tfloat32-b1-h55-r3-s2-c1-m1-SAME-wmin-1-wmax2-imin-0.1-imax0.3-amin-0.01-amax-0.03", "mult", MEv2("Generic")],
  "0-3conv_ba_mult_fp16_wave"  : [ "trivnet_conv_ba_mult", "tfloat16-b1-h55-r3-s2-c1-m1-SAME-wmin-1-wmax2-imin-0.1-imax0.3-amin-0.01-amax-0.03", "mult", MEv2("Generic")],
  "0-2matmult_add_fp32_wave"  : [ "trivnet_matmul_add", "tfloat32-b1-h1-r1-s1-c512-m2048-SAME-wmin-1-wmax2-imin-0.1-imax0.3-amin-0.01-amax-0.03", "matmult", MEv2("Generic")],
  "0-2matmult_add_fp16_wave"  : [ "trivnet_matmul_add", "tfloat16-b1-h1-r1-s1-c512-m2048-SAME-wmin-1-wmax2-imin-0.1-imax0.3-amin-0.01-amax-0.03", "matmult", MEv2("Generic")],

  "0-1conv_s8_32b_wave": [ "trivnet_lin",    "tfloat32-l2-b1-h16-r1-s8-c1-m1-wmin-0.1-wmax0.11-imin-0.2-imax0.21", "1conv32", MEv2("Generic")],
  "0-1conv_exp_pad_wave" : [ "trivnet_conv1",  "tfloat16-b1-h16-r7-s2-c1-m1-wmin2-wmax2.2-imin3-imax3.2-padw2-pade3", "1conv", MEv2("Generic")],
  "1-1conv7_64_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h16-r7-s1-c64-m64-wmin-0.1-wmax0.11-imin-0.2-imax0.21", "1conv", MEv2("Generic")],
  "1-1conv9_64_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h16-r9-s1-c64-m64-wmin-0.1-wmax0.11-imin-0.2-imax0.21", "1conv", MEv2("Generic")],

  # Wave graph development tcc reference and tests
  "1-1conv0_wave"      : [ "trivnet_conv1",  "tfloat16-b1-h1-r1-s1-c33-m1-wmin-0.01-wmax0.011-imin-0.02-imax0.022", "1conv", MEv2("Generic")],
  "1-1conv0_c128_wave"      : [ "trivnet_conv1",  "tfloat16-b1-h1-r1-s1-c128-m1-wmin-0.01-wmax0.011-imin-0.022-imax0.023", "1conv", MEv2("Generic")],
  "1-1conv0_c256_wave"      : [ "trivnet_conv1",  "tfloat16-b1-h1-r1-s1-c256-m1-wmin-0.01-wmax0.011-imin-0.022-imax0.023", "1conv", MEv2("Generic")],
  "1-1conv0_m64_wave"      : [ "trivnet_conv1",  "tfloat16-b1-h1-r1-s1-c1-m64-wmin1-wmax1.1-imin2-imax2.2", "1conv", MEv2("Generic")],
  "1-1conv0_m128_wave"      : [ "trivnet_conv1",  "tfloat16-b1-h1-r1-s1-c1-m128-wmin-0.01-wmax0.011-imin-0.02-imax0.022", "1conv", MEv2("Generic")],
  "1-1conv0_m2_wave"      : [ "trivnet_conv1",  "tfloat16-b1-h1-r1-s1-c1-m2-wmin-0.01-wmax0.011-imin-0.02-imax0.022", "1conv", MEv2("Generic")],
  "1-1conv0_h16c128_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h16-r1-s1-c128-m1-wmin2-wmax2.2-imin3-imax3.2", "1conv", MEv2("Generic")],
  "1-1conv0_h16c256_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h16-r1-s1-c256-m1-wmin2-wmax2.2-imin3-imax3.2", "1conv", MEv2("Generic")],
  "1-1conv0_h16c1m1_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h16-r1-s1-c2-m1-wmin-0.2-wmax0.3-imin-0.1-imax0.2", "1conv", MEv2("Generic")],
  "1-1conv0_h16c2m1_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h16-r1-s1-c2-m1-wmin-0.2-wmax0.3-imin-0.1-imax0.2", "1conv", MEv2("Generic")],

  "1-1conv0_h16c256m128_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h16-r1-s1-c256-m128-wmin-1-wmax2-imin-1-imax3", "1conv", MEv2("Generic")],
  "1-1conv0_h16c256m128_fp32_wave"   : [ "trivnet_conv1",  "tfloat32-b1-h16-r1-s1-c256-m128-wmin-1-wmax2-imin-1-imax3", "1conv", MEv2("Generic")],

  "1-1conv0_h40c1m1_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h40-r1-s1-c1-m1-wmin2-wmax2.2-imin3-imax3.2", "1conv", MEv2("Generic")],
  #"1-1conv0_h40c128m64_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h40-r1-s1-c128-m64-wmin2-wmax2.2-imin3-imax3.2", "1conv", MEv2("Generic")],
  #"1-1conv0_h40c256m128_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h40-r1-s1-c256-m128-wmin2-wmax2.2-imin3-imax3.2", "1conv", MEv2("Generic")],
  #"1-1conv0_h40c128m128_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h40-r1-s1-c128-m128-wmin2-wmax2.2-imin3-imax3.2", "1conv", MEv2("Generic")],
  #"1-1conv0_h40c256m64_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h40-r1-s1-c256-m64-wmin2-wmax2.2-imin3-imax3.2", "1conv", MEv2("Generic")],

  #"1-1conv0_h32c128m64_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h32-r1-s1-c128-m64-wmin2-wmax2.2-imin3-imax3.2", "1conv", MEv2("Generic")],
  #"1-1conv0_h32c256m128_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h32-r1-s1-c256-m128-wmin2-wmax2.2-imin3-imax3.2", "1conv", MEv2("Generic")],
  #"1-1conv0_h32c128m128_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h32-r1-s1-c128-m128-wmin2-wmax2.2-imin3-imax3.2", "1conv", MEv2("Generic")],
  #"1-1conv0_h32c256m64_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h32-r1-s1-c256-m64-wmin2-wmax2.2-imin3-imax3.2", "1conv", MEv2("Generic")],

  "1-1conv0_h64c2m1_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h64-r1-s1-c2-m1-wmin2-wmax2.2-imin3-imax3.2", "1conv", MEv2("Generic")],
  ##"1-1conv0_h256c2m1_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h256-r1-s1-c2-m1-wmin2-wmax2.2-imin3-imax3.2", "1conv", MEv2("Generic")],

  "1-1conv0_h32c1m1_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h32-r1-s1-c1-m1-wmin0-wmax1-imin0-imax1023", "1conv", MEv2("Generic")],
  "1-1conv0_h32c2m1_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h32-r1-s1-c2-m1-wmin0-wmax1-imin0-imax1023", "1conv", MEv2("Generic")],
  "1-1conv0_h32c4m1_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h32-r1-s1-c4-m1-wmin0-wmax1-imin0-imax1023", "1conv", MEv2("Generic")],
  "1-1conv0_h32c8m1_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h32-r1-s1-c8-m1-wmin0-wmax1-imin0-imax1023", "1conv", MEv2("Generic")],
  "1-1conv0_h32c64m1_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h32-r1-s1-c64-m1-wmin2-wmax2.2-imin3-imax3.2", "1conv", MEv2("Generic")],
  "1-1conv0_h32c128m1_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h32-r1-s1-c128-m1-wmin2-wmax2.2-imin3-imax3.2", "1conv", MEv2("Generic")],
  "1-1conv0_h32c1m2_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h32-r1-s1-c1-m2-wmin2-wmax2.2-imin3-imax3.2", "1conv", MEv2("Generic")],

  "1-1conv0_h28c1m1_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h28-r1-s1-c1-m1-wmin2-wmax2.2-imin3-imax3.2", "1conv", MEv2("Generic")],
  "1-1conv0_h56c1m1_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h56-r1-s1-c1-m1-wmin-1-wmax2-imin-2-imax3", "1conv", MEv2("Generic")],
  "1-1conv0_h112c1m1_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h112-r1-s1-c1-m1-wmin2-wmax2.2-imin3-imax3.2", "1conv", MEv2("Generic")],
  "1-1conv0_h224c1m1_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h224-r1-s2-c1-m1-wmin2-wmax2.2-imin3-imax3.2", "1conv", MEv2("Generic")],

  "1-1conv0_h55c256_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h55-r1-s1-c256-m1-wmin-0.1-wmax0.11-imin-0.2-imax0.22", "1conv", MEv2("Generic")],
  #"1-1conv0_h55c64m256_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h55-r1-s1-c64-m256-wmin-0.1-wmax0.11-imin-0.2-imax0.22", "1conv", MEv2("Generic")],
  "0-1conv0_h55c256m1_wave" : [ "trivnet_conv1",  "tfloat16-b1-h55-r1-s1-c256-m1-wmin2-wmax2.2-imin3-imax3.2", "1conv", MEv2("Generic")],
  "1-1conv0_h55m64_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h55-r1-s1-c1-m64-wmin-0.1-wmax0.11-imin-0.2-imax0.22", "1conv", MEv2("Generic")],
  "1-1conv0_r3h16c128_wave" : [ "trivnet_conv1",  "tfloat16-b1-h16-r3-s1-c128-m1-wmin2-wmax2.2-imin3-imax3.2", "1conv", MEv2("Generic")],
  "1-1conv0_r3h55c256_wave" : [ "trivnet_conv1",  "tfloat16-b1-h55-r3-s1-c256-m1-wmin-0.1-wmax0.11-imin-0.2-imax0.22", "1conv", MEv2("Generic")],

  "1-1conv_transpose_wave" : [ "trivnet_conv_transpose",  "tfloat16-l1-b1-h4-r1-s1-c1-m1-wmin-0.1-wmax0.2-imin0-imax0.2", "1conv", MEv2("Generic")],
  "1-1conv_transpose_r3_wave" : [ "trivnet_conv_transpose",  "tfloat16-l1-b1-h2-r3-s1-c1-m1-wmin-0.1-wmax0.2-imin0-imax0.2", "1conv", MEv2("Generic")],
  "1-1conv_transpose_s2_wave" : [ "trivnet_conv_transpose",  "tfloat16-l1-b1-h4-r1-s2-c1-m1-wmin-0.1-wmax0.2-imin0-imax0.2", "1conv", MEv2("Generic")],
  "1-1conv_transpose_s4_wave" : [ "trivnet_conv_transpose",  "tfloat16-l1-b1-h4-r1-s4-c1-m1-wmin-0.1-wmax0.2-imin0-imax0.2", "1conv", MEv2("Generic")],
  "1-1conv_transpose_1d_h8r4s1_wave" : [ "trivnet_conv_transpose_1d",  "tfloat16-l1-b1-h8-r4-s1-c1-m1-wmin-0.1-wmax0.2-imin0-imax0.2", "1conv", MEv2("Generic")],
  "1-1conv_transpose_1d_h8r4s2_wave" : [ "trivnet_conv_transpose_1d",  "tfloat16-l1-b1-h8-r4-s2-c1-m1-wmin-0.1-wmax0.2-imin0-imax0.2", "1conv", MEv2("Generic")],
  "1-1conv_transpose_1d_h32r4s2_wave" : [ "trivnet_conv_transpose_1d",  "tfloat16-l1-b1-h32-r4-s2-c1-m1-wmin-0.1-wmax0.2-imin0-imax0.2", "1conv", MEv2("Generic")],
  "1-1conv_transpose_1d_h32r4s10_wave" : [ "trivnet_conv_transpose_1d",  "tfloat16-l1-b1-h32-r4-s10-c1-m1-wmin-0.1-wmax0.2-imin0-imax0.2", "1conv", MEv2("Generic")],

  "2-1conv_transpose_1d_h128r4s8_wave" : [ "trivnet_conv_transpose_1d",  "tfloat16-l1-b1-h128-r4-s8-c1-m1-wmin-0.1-wmax0.2-imin-0.1-imax0.2", "1conv", MEv2("Generic")],
  "2-1conv_transpose_1d_h128r4s10_wave" : [ "trivnet_conv_transpose_1d",  "tfloat16-l1-b1-h128-r4-s8-c1-m1-wmin-0.1-wmax0.2-imin-0.1-imax0.2", "1conv", MEv2("Generic")],
  "2-1conv_transpose_1d_h128r40s10_wave" : [ "trivnet_conv_transpose_1d",  "tfloat16-l1-b1-h128-r4-s8-c1-m1-wmin-0.1-wmax0.2-imin-0.1-imax0.2", "1conv", MEv2("Generic")],
  "2-1conv_transpose_1d_h30r20s10_wave" : [ "trivnet_conv_transpose_1d",  "tfloat16-l1-b1-h30-r20-s10-c1-m1-wmin-0.1-wmax0.2-imin-0.1-imax0.2", "1conv", MEv2("Generic")],
  "2-1conv_transpose_1d_h30r40s10_wave" : [ "trivnet_conv_transpose_1d",  "tfloat16-l1-b1-h30-r40-s10-c1-m1-wmin-0.1-wmax0.2-imin-0.1-imax0.2", "1conv", MEv2("Generic")],

  "3-1conv_transpose_1d_h10r40s10_wave" : [ "trivnet_conv_transpose_1d",  "tfloat16-l1-b1-h10-r40-s10-c80-m256-wmin-0.1-wmax0.2-imin0-imax0.2", "1conv", MEv2("Generic")],
  "3-1conv_transpose_1d_h89r40s10_wave" : [ "trivnet_conv_transpose_1d",  "tfloat16-l1-b1-h89-r40-s10-c32-m32-wmin-0.1-wmax0.2-imin-0.1-imax0.2", "1conv", MEv2("Generic")],
  "3-1conv_transpose_1d_h100r80s20_wave" : [ "trivnet_conv_transpose_1d",  "tfloat16-l1-b1-h100-r80-s20-c256-m256-wmin-0.1-wmax0.2-imin-0.1-imax0.2", "1conv", MEv2("Generic-NoVerify")],
  "3-1conv_transpose_1d_h10r40s10_wave" : [ "trivnet_conv_transpose_1d",  "tfloat16-l1-b1-h10-r40-s10-c256-m256-wmin-0.1-wmax0.12-imin-0.1-imax0.12", "1conv", MEv2("Generic-NoVerify")],

  "2-1conv3_64s8_wave" : [ "trivnet_conv1",  "tfloat16-b1-h16-r3-s8-c64-m64-wmin-0.1-wmax0.11-imin-0.2-imax0.22", "1conv", MEv2("Generic")],
  "2-1conv9_64s8_wave" : [ "trivnet_conv1",  "tfloat16-b1-h16-r9-s8-c64-m64-wmin-0.1-wmax0.11-imin-0.2-imax0.22", "1conv", MEv2("Generic")],

  "2-padasym_strd_h112r7s2_wave" : [ "trivnet_conv1", "tfloat16-b1-h112-r7-s2-c1-m1-wmin-0.1-wmax0.11-imin-0.2-imax0.22", "1conv", MEv2("Generic")],
  "2-padasym_strd_h224r7s2_wave" : [ "trivnet_conv1", "tfloat16-b1-h224-r7-s2-c3-m64-wmin-0.1-wmax0.2-imin-0.2-imax0.3", "1conv", MEv2("Generic")],
  "2-padasym_strd_h224r7s2_fp32_wave" : [ "trivnet_conv1", "tfloat32-b1-h224-r7-s2-c3-m64-wmin-0.1-wmax0.2-imin-0.2-imax0.3", "1conv", MEv2("Generic")],

  # Full c, m in resnet50 are 512, 2048
  "3-rn50_pool2_wave"  : [ "trivnet_conv_pool", "tfloat16-b1-h7-r1-s1-c128-m64-SAME-AvgPool-k7-d7-PERM-wmin-0.1-wmax0.1-imin-1-imax2", "1conv1pool", MEv2("Generic")],
  "3-1conv1maxpool_k3d2_wave"  : [ "trivnet_conv_pool_conv", "tfloat16-b1-h224-r3-s2-c128-m64-VALID-MaxPool-k3-d2-wmin-0.2-wmax0.3-imin-0.2-imax0.3", "1conv1pool", MEv2("Generic")],

  "3-1conv1relupoolconv_k3d2_wave"  : [ "trivnet_conv_relu_pool_conv", "tfloat16-b1-h4-r1-s1-c1-m1-VALID-MaxPool-k3-d2-wmin-0.2-wmax0.3-imin-0.2-imax0.3", "convrelupool", MEv2("Generic")],

  "3-1conv0_padvalid_wave" : [ "trivnet_conv1_padvalid",  "tfloat16-b1-h230-r7-s2-c3-m64-wmin-2-wmax2.2-imin-3-imax3.2", "1conv", MEv2("Generic")],
  "3-1conv0_h298_wave" : [ "trivnet_conv1",  "tfloat16-b1-h298-r3-s2-c1-m1-wmin-0.1-wmax0.11-imin-0.2-imax0.22", "1conv", MEv2("Generic")],

  # Sprint9 Story 1 milestone - all resnet50 float32 Conv2D layers as unit test
  # The 00 is just for testing the regression harness
  "3-rn50-t00_fp32_wave" : [ "trivnet_conv1",  "tfloat32-b1-h1-r1-s1-c1-m1-wmin-1-wmax1.1-imin-3-imax3.2",       "1conv", MEv2("Generic")],
  "3-rn50-01_fp32_wave" : [ "trivnet_conv1",  "tfloat32-b1-h14-r3-s1-c256-m256-wmin-1-wmax1.1-imin-3-imax3.2",  "1conv", MEv2("Generic")],
  "3-rn50-02_fp32_wave" : [ "trivnet_conv1",  "tfloat32-b1-h14-r1-s1-c256-m1024-wmin-1-wmax1.1-imin-3-imax3.2", "1conv", MEv2("Generic")],
  "3-rn50-03_fp32_wave" : [ "trivnet_conv1",  "tfloat32-b1-h14-r1-s1-c1024-m256-wmin-1-wmax1.1-imin-3-imax3.2", "1conv", MEv2("Generic")],
  "3-rn50-04_fp32_wave" : [ "trivnet_conv1",  "tfloat32-b1-h55-r1-s1-c64-m256-wmin-1-wmax1.1-imin-3-imax3.2",   "1conv", MEv2("Generic")],
  "3-rn50-05_fp32_wave" : [ "trivnet_conv1",  "tfloat32-b1-h28-r3-s1-c128-m128-wmin-1-wmax1.1-imin-3-imax3.2",  "1conv", MEv2("Generic")],
  "3-rn50-06_fp32_wave" : [ "trivnet_conv1",  "tfloat32-b1-h28-r1-s1-c128-m512-wmin-1-wmax1.1-imin-3-imax3.2",  "1conv", MEv2("Generic")],
  "3-rn50-07_fp32_wave" : [ "trivnet_conv1",  "tfloat32-b1-h55-r3-s1-c64-m64-wmin-1-wmax1.1-imin-3-imax3.2",    "1conv", MEv2("Generic")],
  "3-rn50-08_fp32_wave" : [ "trivnet_conv1",  "tfloat32-b1-h7-r3-s1-c512-m512-wmin-1-wmax1.1-imin-3-imax3.2",   "1conv", MEv2("Generic")],
  "3-rn50-09_fp32_wave" : [ "trivnet_conv1",  "tfloat32-b1-h7-r1-s1-c512-m2048-wmin-1-wmax1.1-imin-3-imax3.2",  "1conv", MEv2("Generic")],
  "3-rn50-10_fp32_wave" : [ "trivnet_conv1",  "tfloat32-b1-h28-r1-s1-c512-m128-wmin-1-wmax1.1-imin-3-imax3.2",  "1conv", MEv2("Generic")],
  "3-rn50-11_fp32_wave" : [ "trivnet_conv1",  "tfloat32-b1-h55-r1-s1-c256-m64-wmin-1-wmax1.1-imin-3-imax3.2",   "1conv", MEv2("Generic")],
  "3-rn50-12_fp32_wave" : [ "trivnet_conv1",  "tfloat32-b1-h7-r1-s1-c2048-m512-wmin-1-wmax1.1-imin-3-imax3.2",  "1conv", MEv2("Generic")],
  "3-rn50-13_fp32_wave" : [ "trivnet_conv1",  "tfloat32-b1-h55-r1-s1-c64-m64-wmin-1-wmax1.1-imin-3-imax3.2",    "1conv", MEv2("Generic")],
  "3-rn50-14_fp32_wave" : [ "trivnet_conv1",  "tfloat32-b1-h28-r1-s2-c512-m256-wmin-1-wmax1.1-imin-3-imax3.2",  "1conv", MEv2("Generic")],
  "3-rn50-15_fp32_wave" : [ "trivnet_conv1",  "tfloat32-b1-h28-r1-s2-c512-m1024-wmin-1-wmax1.1-imin-3-imax3.2", "1conv", MEv2("Generic")],

  "3-rn50-16_fp32_wave" : [ "trivnet_conv1", "tfloat32-b1-h224-r7-s2-c3-m64-wmin-1-wmax1.1-imin-3-imax3.2", "1conv", MEv2("Generic") ],

  "3-rn50-16_fp32_wave-fast_dram" : [ "trivnet_conv1", "tfloat32-b1-h224-r7-s2-c3-m64-wmin-1-wmax1.1-imin-3-imax3.2", "1conv", MEv2("Generic"),
    "--env SIM_ADD_FLAGS=' --dram_frequency 0 --dram_latency 1'" ],

  "3-rn50-17_fp32_wave" : [ "trivnet_conv1",  "tfloat32-b1-h55-r1-s2-c256-m512-wmin-1-wmax1.1-imin-3-imax3.2",  "1conv", MEv2("Generic")],
  "3-rn50-18_fp32_wave" : [ "trivnet_conv1",  "tfloat32-b1-h55-r1-s2-c256-m128-wmin-1-wmax1.1-imin-3-imax3.2",  "1conv", MEv2("Generic")],
  "3-rn50-19_fp32_wave" : [ "trivnet_conv1",  "tfloat32-b1-h14-r1-s2-c1024-m512-wmin-1-wmax1.1-imin-3-imax3.2",  "1conv", MEv2("Generic")],
  "3-rn50-20_fp32_wave" : [ "trivnet_conv1",  "tfloat32-b1-h14-r1-s2-c1024-m2048-wmin-1-wmax1.1-imin-3-imax3.2", "1conv", MEv2("Generic")],

  "3-rn50-t00_wave" : [ "trivnet_conv1",  "tfloat16-b1-h1-r1-s1-c1-m1-wmin-1-wmax1.1-imin-3-imax3.2",       "1conv", MEv2("Generic")],
  "3-rn50-01_wave" : [ "trivnet_conv1",  "tfloat16-b1-h14-r3-s1-c256-m256-wmin-1-wmax1.1-imin-3-imax3.2",  "1conv", MEv2("Generic")],
  "3-rn50-02_wave" : [ "trivnet_conv1",  "tfloat16-b1-h14-r1-s1-c256-m1024-wmin-1-wmax1.1-imin-3-imax3.2", "1conv", MEv2("Generic")],
  "3-rn50-03_wave" : [ "trivnet_conv1",  "tfloat16-b1-h14-r1-s1-c1024-m256-wmin-1-wmax1.1-imin-3-imax3.2", "1conv", MEv2("Generic")],
  "3-rn50-04_wave" : [ "trivnet_conv1",  "tfloat16-b1-h55-r1-s1-c64-m256-wmin-1-wmax1.1-imin-3-imax3.2",   "1conv", MEv2("Generic")],
  "3-rn50-05_wave" : [ "trivnet_conv1",  "tfloat16-b1-h28-r3-s1-c128-m128-wmin-1-wmax1.1-imin-3-imax3.2",  "1conv", MEv2("Generic")],
  "3-rn50-06_wave" : [ "trivnet_conv1",  "tfloat16-b1-h28-r1-s1-c128-m512-wmin-1-wmax1.1-imin-3-imax3.2",  "1conv", MEv2("Generic")],
  "3-rn50-07_wave" : [ "trivnet_conv1",  "tfloat16-b1-h55-r3-s1-c64-m64-wmin-1-wmax1.1-imin-3-imax3.2",    "1conv", MEv2("Generic")],
  "3-rn50-08_wave" : [ "trivnet_conv1",  "tfloat16-b1-h7-r3-s1-c512-m512-wmin-1-wmax1.1-imin-3-imax3.2",   "1conv", MEv2("Generic")],
  #"3-rn50-08_b2_wave" : [ "trivnet_conv1",  "tfloat16-b2-h7-r3-s1-c512-m512-wmin-1-wmax1.1-imin-3-imax3.2",   "1conv", MEv2("Generic")],
  "3-rn50-09_wave" : [ "trivnet_conv1",  "tfloat16-b1-h7-r1-s1-c512-m2048-wmin-1-wmax1.1-imin-3-imax3.2",  "1conv", MEv2("Generic")],
  "3-rn50-10_wave" : [ "trivnet_conv1",  "tfloat16-b1-h28-r1-s1-c512-m128-wmin-1-wmax1.1-imin-3-imax3.2",  "1conv", MEv2("Generic")],
  "3-rn50-11_wave" : [ "trivnet_conv1",  "tfloat16-b1-h55-r1-s1-c256-m64-wmin-1-wmax1.1-imin-3-imax3.2",   "1conv", MEv2("Generic")],
  "3-rn50-12_wave" : [ "trivnet_conv1",  "tfloat16-b1-h7-r1-s1-c2048-m512-wmin-1-wmax1.1-imin-3-imax3.2",  "1conv", MEv2("Generic")],
  #"3-rn50-12_b2_wave" : [ "trivnet_conv1",  "tfloat16-b2-h7-r1-s1-c2048-m512-wmin-1-wmax1.1-imin-3-imax3.2",  "1conv", MEv2("Generic")],
  "3-rn50-13_wave" : [ "trivnet_conv1",  "tfloat16-b1-h55-r1-s1-c64-m64-wmin-1-wmax1.1-imin-3-imax3.2",    "1conv", MEv2("Generic")],
  "3-rn50-14_wave" : [ "trivnet_conv1",  "tfloat16-b1-h28-r1-s2-c512-m256-wmin-1-wmax1.1-imin-3-imax3.2",  "1conv", MEv2("Generic")],
  "3-rn50-15_wave" : [ "trivnet_conv1",  "tfloat16-b1-h28-r1-s2-c512-m1024-wmin-1-wmax1.1-imin-3-imax3.2", "1conv", MEv2("Generic")],
  "3-rn50-16_wave" : [ "trivnet_conv1",  "tfloat16-b1-h224-r7-s2-c3-m64-wmin-1-wmax1.1-imin-3-imax3.2",    "1conv", MEv2("Generic")],
  "3-rn50-17_wave" : [ "trivnet_conv1",  "tfloat16-b1-h55-r1-s2-c256-m512-wmin-1-wmax1.1-imin-3-imax3.2",  "1conv", MEv2("Generic")],
  "3-rn50-18_wave" : [ "trivnet_conv1",  "tfloat16-b1-h55-r1-s2-c256-m128-wmin0-wmax1-imin0-imax3",  "1conv", MEv2("Generic")],
  "3-rn50-19_wave" : [ "trivnet_conv1",  "tfloat16-b1-h14-r1-s2-c1024-m512-wmin-1-wmax1.1-imin-3-imax3.2",  "1conv", MEv2("Generic")],
  "3-rn50-20_wave" : [ "trivnet_conv1",  "tfloat16-b1-h14-r1-s2-c1024-m2048-wmin-1-wmax1.1-imin-3-imax3.2", "1conv", MEv2("Generic")],

  "3-rn50-16_wave_repl" : [ "trivnet_conv1", "tfloat16-b1-h224-r7-s2-c3-m64-wmin-1-wmax1.1-imin-3-imax3.2", "1conv", MEv2("Generic-Repl")],

  ## db
  "3-rn50-16_wave_repl-fast_dram" : [ "trivnet_conv1", "tfloat16-b1-h224-r7-s2-c3-m64-wmin-1-wmax1.1-imin-3-imax3.2", "1conv", MEv2("RN50-Repl"),
    "--env SIM_ADD_FLAGS=' --dram_frequency 0 --dram_latency 1 '" ],

  "3-rn50-16_b2_wave_repl" : [ "trivnet_conv1",  "tfloat16-b2-h224-r7-s2-c3-m64-wmin1-wmax1-imin0-imax223", "1conv", MEv2("RN50-Repl")],

  "3-incep_amoeba_h299r3s2c3m32_wave_repl" : [ "trivnet_conv1", "tfloat16-b1-h299-r3-s2-c3-m32-VALID-wmin-1-wmax1.1-imin-3-imax3.2", "1conv", MEv2("Generic-Repl")],
  "3-h400r3s2c3m32_wave_repl" : [ "trivnet_conv1", "tfloat16-b1-h400-r3-s2-c3-m32-VALID-wmin-1-wmax1.1-imin-3-imax3.2", "1conv", MEv2("Generic-Repl")],
  "3-h256r3s2c3m32_wave_repl" : [ "trivnet_conv1", "tfloat16-b1-h256-r3-s2-c3-m32-VALID-wmin-1-wmax1.1-imin-3-imax3.2", "1conv", MEv2("Generic-Repl")],
  "3-h128r3s2c3m32_wave_repl" : [ "trivnet_conv1", "tfloat16-b1-h128-r3-s2-c3-m32-VALID-wmin-1-wmax1.1-imin-3-imax3.2", "1conv", MEv2("Generic-Repl")],
  "3-h128r3s2c2m32_wave_repl" : [ "trivnet_conv1", "tfloat16-b1-h128-r3-s2-c2-m32-VALID-wmin-1-wmax1.1-imin-3-imax3.2", "1conv", MEv2("Generic-Repl")],
  "3-h128r3s2c1m32_wave_repl" : [ "trivnet_conv1", "tfloat16-b1-h128-r3-s2-c1-m32-VALID-wmin-1-wmax1.1-imin-3-imax3.2", "1conv", MEv2("Generic-Repl")],
  "3-h64r3s2c1m32_wave_repl" : [ "trivnet_conv1", "tfloat16-b1-h64-r3-s2-c1-m32-VALID-wmin-1-wmax1.1-imin-3-imax3.2", "1conv", MEv2("Generic-Repl")],
  "3-h16r3s2c1m32_wave_repl" : [ "trivnet_conv1", "tfloat16-b1-h16-r3-s2-c1-m32-VALID-wmin-1-wmax1.1-imin-3-imax3.2", "1conv", MEv2("Generic-Repl")],
  "3-h4r3s2c1m32_wave_repl" : [ "trivnet_conv1", "tfloat16-b1-h16-r3-s2-c1-m32-VALID-wmin-1-wmax1.1-imin-3-imax3.2", "1conv", MEv2("Generic-Repl")],
  # Replication with stride 1 not working yet
  "3-h128r3s1c6m32_wave_repl" : [ "trivnet_conv1", "tfloat16-b1-h128-r2-s1-c6-m32-VALID-wmin-1-wmax1.1-imin-3-imax3.2", "1conv", MEv2("Generic-Repl")],
  "3-h128r2s1c6m32_wave_repl" : [ "trivnet_conv1", "tfloat16-b1-h128-r2-s1-c6-m32-VALID-wmin-1-wmax1.1-imin-3-imax3.2", "1conv", MEv2("Generic-Repl")],
  "3-h128r2s1c4m32_wave_repl" : [ "trivnet_conv1", "tfloat16-b1-h128-r2-s1-c4-m32-VALID-wmin-1-wmax1.1-imin-3-imax3.2", "1conv", MEv2("Generic-Repl")],
  "3-h128r2s1c2m32_wave_repl" : [ "trivnet_conv1", "tfloat16-b1-h128-r2-s1-c2-m32-VALID-wmin-1-wmax1.1-imin-3-imax3.2", "1conv", MEv2("Generic-Repl")],
  "3-h128r2s1c1m32_wave_repl" : [ "trivnet_conv1", "tfloat16-b1-h128-r2-s1-c1-m32-VALID-wmin-1-wmax1.1-imin-3-imax3.2", "1conv", MEv2("Generic-Repl")],

  #"5-lstm_ptb"     : [ "tf_pb",          "lstm_ptb_word_lm/ptb_word_lm.pb",  "lstm_ptb", "--input_node Valid/ValidInput/StridedSlice ", "linspace1"],
  "6-alexnet"     : [ "tf_pb",          "alexnet_v100/alexnet_fp32.pb",  "alexnet", "--input_node Variable ", "linspace1"],
  "8-resnet50"                : [ "tf_pb",   "resnet50/resnet_v1_50_fp32_opt.pb",        "resnet50", " --depth 2", "linspace1"],
  "8-resnet50_fp32_keras"     : [ "tf_pb",   "resnet50_keras/resnet50_fp32_keras.pb",    "resnet50", "--input_node input_1  --depth 2", "linspace1"],
  "8-resnet50_fp32_keras_opt" : [ "tf_pb",   "resnet50_keras/resnet50_fp32_keras_opt.pb","resnet50", "--input_node input_1  --depth 2", "linspace1"],
  "8-resnet50_fp16_keras"     : [ "tf_pb",   "resnet50_keras/resnet50_fp16_keras.pb",    "resnet50", "--input_node input_1  --depth 2", "linspace1"],
  "8-resnet50_fp16_keras_opt_b16" : [ "tf_pb",   "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2 --batch 16", "linspace1"],
  "8-resnet50_fp16_keras_opt" : [ "tf_pb",   "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2", "linspace1"],
  "8-resnet50_fp16_wave"      : [ "tf_pb",   "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2 --scheduler wave2 --wavegraph_checks structure data-race", "linspace1"],
  "8-resnet50_fp32_wave"      : [ "tf_pb",   "resnet50_keras/resnet50_fp32_keras_opt.pb","resnet50", "--input_node input_1  --depth 2 --scheduler wave2 --wavegraph_checks structure data-race", "linspace1"],
  "8-resnet50_fp16_wave_b2"   : [ "tf_pb",   "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2 --scheduler wave2 --batch 2 --wavegraph_checks structure data-race", "linspace1"],
  "9-resnet152"               : [ "tf_pb",   "resnet_v2_152/pb/resnet_v2_152_fp32.pb",   "resnet152", " --depth 2", "linspace1"],
  "9-resnet152_waveopt"       : [ "tf_pb",   "resnet_v2_152/pb/resnet_v2_152_fp32.pb",   "resnet152", "--partition from resnet_v2_152/conv1/convolution resnet_v2_152/postnorm/batchnorm/mul_1 --executors host all waveopt 1  --depth 2 --scheduler wave2 --images %s --wavegraph_checks structure data-race" % rnDogJpg, "--input_files %s" % rnDogJpg],

  # Parallel WaveNet (ME only)
  "3-parwavenet_10_fp16_waveopt" : [ "tf_pb",   "parallel_wavenet/example1/parwavenet_10_frozen_fp16.pb", "parallel_wavenet", " --focus_to truediv --show_op_name_in_kgraph --input_node sub_1 --depth -1 --partition from truediv --executors host all waveopt 1 --images linspace1", "--input_files trivnet_sub_1:0.npy"],
  "3-parwavenet_10_10_fp16_waveopt" : [ "tf_pb",   "parallel_wavenet/example1/parwavenet_10_10_frozen_fp16.pb", "parallel_wavenet", " --input_node Placeholder --depth 2 --partition from truediv --executors waveopt 0 host 1 --images %s"%melSpectra, "--input_files %s" % melSpectra],

  "3-parwavenet_10_fp16_ba15_ba16_reshape67_dbg" :
    [ "tf_pb", "parallel_wavenet/example1/parwavenet_10_frozen_fp16.pb", "parallel_wavenet",
          "--level_order_seed 3 --input_node Placeholder Log Log_1 --focus_to %s --show_op_name_in_kgraph --depth -1 "%("Reshape_67")
          + "--sg_input_format Log NW Log_1 NW sub_1 NW mul_13 NW add_24 NW Squeeze_36 NW Squeeze_35 NW "
          + "--executors host all wave 2 --partition from_multi BiasAdd_15 BiasAdd_16 "
          + "%s --images %s linspace1 linspace1" % (MEv2("Generic-NoVerify-SaveAll"), melSpectra),
          "--input_files Placeholder:0=%s Log:0=trivnet_Log:0.npy Log_1:0=trivnet_Log_1:0.npy " % melSpectra],

  # Subgraph partioned flow using neural network executor
  "0-4conv_relu_nne" : [ "trivnet_lin",    "tfloat16-l3-b1-h4-r3-s1-c1-m1-relu-wmin-0.2-wmax0.4-imin-1-imax2", "4conv_nne", "--partition conv --executors wave 1 3 host 0 2 4 --debug 1 " + MEv2("Generic")],

  # Resnet
  "8-rn50_nne_auto"             : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition auto --executors wave all  --scheduler wave2 --images %s --wavegraph_checks structure data-race" %(rnPreFp16, rnDogJpg), "--input_files %s" % rnDogJpg ],
  #"8-rn50_nne_fp32_meauto"      : [ "tf_pb", "resnet50_keras/resnet50_fp32_keras_opt.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition meauto --executors wave all host 17  --scheduler wave2 --images %s" %(rnPreFp32, rnDogJpg), "--input_files %s" % rnDogJpg ],
  "8-rn50_nne_fp16_meauto"      : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition meauto --executors wave all host 17  --scheduler wave2 --schedule_options ' --nname=generic' --images %s --waive_wavegraph_checks" %(rnPreFp16, rnDogJpg), "--input_files %s" % rnDogJpg ],
  #"8-rn50_nne_conv"            : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition conv --executors tcc 2 6 8 13 15 20 22 host 0 --images %s" %(rnPreFp16, rnDogJpg), "linspace1"],
  "4-rn50_nne_fc"               : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from avg_pool/AvgPool --executors host 0 host 1 --images %s" %(rnPreFp16, rnDogJpg), "--input_files %s" % rnDogJpg ],
  #"4-rn50_nne_from1_fp32_wave"  : [ "tf_pb", "resnet50_keras/resnet50_fp32_keras_opt.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from bn_conv1/batchnorm_1/add_1   --executors wave 0 host 1  --scheduler wave2 --images %s --wavegraph_checks structure data-race" %(rnPreFp32, rnDogJpg), "--input_files %s" % rnDogJpg ],
  #"4-rn50_nne_from2_fp32_wave"  : [ "tf_pb", "resnet50_keras/resnet50_fp32_keras_opt.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from activation_1/Relu   --executors wave 0 host 1  --scheduler wave2 --images %s --wavegraph_checks structure data-race" %(rnPreFp32, rnDogJpg), "--input_files %s" % rnDogJpg ],
  "4-rn50_nne_from3_fp32_wave"  : [ "tf_pb", "resnet50_keras/resnet50_fp32_keras_opt.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from max_pooling2d_1/MaxPool   --executors wave 0 host 1  --scheduler wave2 --images %s --waive_wavegraph_checks" %(rnPreFp32, rnDogJpg), "--input_files %s" % rnDogJpg ],
  #"4-rn50_nne_from1_wave"       : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from bn_conv1/batchnorm_1/add_1   --executors wave 0 host 1  --scheduler wave2 --images %s" %(rnPreFp16, rnDogJpg), "--input_files %s" % rnDogJpg ],
  #"4-rn50_nne_from2_wave"       : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from activation_1/Relu   --executors wave 0 host 1  --scheduler wave2 --images %s" %(rnPreFp16, rnDogJpg), "--input_files %s" % rnDogJpg ],
  "4-rn50_nne_from3_wave"       : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from max_pooling2d_1/MaxPool   --executors wave 0 host 1  --scheduler wave2 --images %s" %(rnPreFp16, rnDogJpg), "--input_files %s" % rnDogJpg ],
  #"5-rn50_nne_to_act4_wave"     : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from activation_4/Relu --executors wave 0 host 1  --scheduler wave2 --images %s" %(rnPreFp16, rnDogJpg), "--input_files %s" % rnDogJpg ],

  "5-rn50_nne_to_act4_wave-no_repl"     : ["tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50",
    "--input_node input_1  --depth 2  --debug 1 %s --partition from activation_4/Relu --executors wave 0 host 1 %s --images %s " %(rnPreFp16, MEv2("RN50"), rnDogJpg),
    "--input_files %s" % rnDogJpg],

  "5-rn50_nne_to_act4_wave-no_repl-all-layers" : ["tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50",
    "--input_node input_1  --depth 2  --debug 1 %s --partition from activation_4/Relu --executors wave 0 host 1 %s --images %s " %(rnPreFp16, MEv2("RN50-SaveAll"), rnDogJpg),
    "--input_files %s --check_against_ref all_available" % rnDogJpg],

  "5-rn50_nne_to_act4_wave-repl" : ["tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50",
    "--input_node input_1  --depth 2  --debug 1 %s --partition from activation_4/Relu --executors wave 0 host 1 %s --images %s " %(rnPreFp16, MEv2("RN50-Repl"), rnDogJpg),
    "--input_files %s" % rnDogJpg],

  "5-rn50_nne_to_act4_b8_wave-repl"  : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50",
    "--input_node input_1  --depth 2  --debug 1 %s --partition from activation_4/Relu --executors wave 0 host 1 %s --batch 8 --images %s "%(rnPreFp16, MEv2("RN50-Repl"), getBatchedJpgs(8)),
    "--input_files %s" % (getBatchedJpgs(8))],

  "5-rn50_nne_to_act4_b16_wave-repl"  : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50",
    "--input_node input_1  --depth 2  --debug 1 %s --partition from activation_4/Relu --executors wave 0 host 1 %s --batch 16 --images %s "%(rnPreFp16, MEv2("RN50-Repl"), getBatchedJpgs(16)),
    "--input_files %s" % (getBatchedJpgs(16))],

  "5-rn50_nne_to_act13_wave-repl"     : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb", "resnet50",
    "--input_node input_1  --depth 2  --debug 1 %s --partition from activation_13/Relu --executors wave 0 host 1  %s --batch 1 --images %s "%(rnPreFp16, MEv2("RN50-Repl"), getBatchedJpgs(1)),
    "--input_files %s" % (getBatchedJpgs(1))],

  "5-rn50_nne_to_act13_wave-no_repl"  : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb", "resnet50",
    "--input_node input_1  --depth 2  --debug 1 %s --partition from activation_13/Relu --executors wave 0 host 1 %s --images %s " %(rnPreFp16, MEv2("RN50"), rnDogJpg),
    "--input_files %s" % rnDogJpg],

  "5-rn50_nne_to_act13_wave-no_repl-all-layers"     : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb", "resnet50",
    "--input_node input_1  --depth 2  --debug 1 %s --partition from activation_13/Relu --executors wave 0 host 1 %s --images %s " %(rnPreFp16, MEv2("RN50-SaveAll"), rnDogJpg),
    "--input_files %s  --check_against_ref all_available" % rnDogJpg],

  # Kaena-820 Control over order of layers in compiler.json
  "6-rn50_nne_to_act46_wave-repl-order0"     : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50",
    "--level_order_seed 0 --input_node input_1  --depth 2  --debug 1 %s --partition from activation_46/Relu --executors wave 0 host 1 %s --images %s "%(rnPreFp16, MEv2("RN50-Repl"), getBatchedJpgs(1)),
    "--input_files %s" % (getBatchedJpgs(1))],

  "6-rn50_nne_to_act46_wave-repl-order1"     : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50",
    "--level_order_seed 1 --input_node input_1  --depth 2  --debug 1 %s --partition from activation_46/Relu --executors wave 0 host 1 %s --images %s "%(rnPreFp16, MEv2("RN50-Repl"), getBatchedJpgs(1)),
    "--input_files %s" % (getBatchedJpgs(1))],

  "7-rn50_nne_fp16_wave-no_repl-save-last-n-layers"        : [
      "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50",
    ( "--input_node input_1  --depth 2  --debug 1 %s --partition from fc1000/Softmax "
      + " --executors wave 0 host 1  --scheduler wave2 --images %s"
      + " --schedule_options ' --save_layer_output -13 --enable_cleanup ' "
    ) %(rnPreFp16, rnDogJpg),
    "--input_files %s --check_against_ref all_available" % rnDogJpg ],

  "7-rn50_nne_fp16_wave-no_repl-save-first-n-layers"        : [
      "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50",
    ( "--input_node input_1  --depth 2  --debug 1 %s --partition from fc1000/Softmax "
      + " --executors wave 0 host 1  --scheduler wave2 --images %s"
      + " --schedule_options ' --save_layer_output 13 --enable_cleanup ' "
    ) %(rnPreFp16, rnDogJpg),
    "--input_files %s --check_against_ref all_available" % rnDogJpg ],

  "7-rn50_nne_fp16_wave-no_repl"        : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from fc1000/Softmax --executors wave 0 host 1  %s --images %s " %(rnPreFp16, MEv2("RN50"), rnDogJpg), "--input_files %s" % rnDogJpg ],
  #"7-rn50_nne_fp16_ap_wave-no_repl"        : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from_multi flatten_1/Shape,flatten_1/Reshape --executors wave 0 host 1  --scheduler wave2 --images %s" %(rnPreFp16, rnDogJpg), "--input_files %s" % rnDogJpg ],
  "7-rn50_nne_fp16_wave"        : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from fc1000/Softmax --executors wave 0 host 1  %s --images %s " %(rnPreFp16, MEv2("RN50-Repl"), rnDogJpg), "--input_files %s" % rnDogJpg ],
  "7-rn50_nne_fp16_ap_wave"        : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from_multi flatten_1/Shape,flatten_1/Reshape --executors wave 0 host 1 %s --images %s " %(rnPreFp16, MEv2("RN50-Repl"), rnDogJpg), "--input_files %s" % rnDogJpg ],

  "7-rn50_nne_fp16_wave-two_banks" : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from fc1000/Softmax --executors wave 0 host 1 %s --images %s " %(rnPreFp16, MEv2("RN50-Repl"), rnDogJpg), "--env SIM_ADD_FLAGS=' --dram_frequency 6400'  --input_files %s" % rnDogJpg ],
  "7-rn50_nne_fp16_wave-fast_dram" : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from fc1000/Softmax --executors wave 0 host 1 %s --images %s " %(rnPreFp16, MEv2("RN50-Repl"), rnDogJpg), "--env SIM_ADD_FLAGS=' --dram_frequency 0 --dram_latency 1' --input_files %s" % rnDogJpg ],
  #"7-rn50_nne_fp16_b2_wave"        : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from fc1000/Softmax --executors wave 0 host 1  --scheduler wave2 --batch 2 --images %s"%(rnPreFp16, getBatchedJpgs(2)), "--input_files %s" % (getBatchedJpgs(2))],
  "7-rn50_nne_fp16_b4_wave-no_repl"        : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from fc1000/Softmax --executors wave 0 host 1  %s --batch 4 --images %s "%(rnPreFp16, MEv2("RN50"), getBatchedJpgs(4)), "--input_files %s" % (getBatchedJpgs(4))],
  "7-rn50_nne_fp16_b4_wave"        : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from fc1000/Softmax --executors wave 0 host 1 %s --batch 4 --images %s "%(rnPreFp16, MEv2("RN50-Repl"), getBatchedJpgs(4)), "--input_files %s" % (getBatchedJpgs(4))],
  "7-rn50_nne_fp16_b4_wave-two_banks"        : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from fc1000/Softmax --executors wave 0 host 1  %s --batch 4 --images %s "%(rnPreFp16, MEv2("RN50-Repl"), getBatchedJpgs(4)), "--env SIM_ADD_FLAGS=' --dram_frequency 6400' --input_files %s" % (getBatchedJpgs(4))],
  "7-rn50_nne_fp16_b4_wave-fast_dram"        : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from fc1000/Softmax --executors wave 0 host 1  %s --batch 4 --images %s "%(rnPreFp16, MEv2("RN50-Repl"), getBatchedJpgs(4)), "--env SIM_ADD_FLAGS=' --dram_frequency 0 --dram_latency 1' --input_files %s" % (getBatchedJpgs(4))],
  "8-rn50_nne_fp16_b8_wave"        : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from fc1000/Softmax --executors wave 0 host 1 %s --batch 8 --images %s "%(rnPreFp16, MEv2("RN50-Repl"), getBatchedJpgs(8)), "--input_files %s" % (getBatchedJpgs(8))],
  "8-rn50_nne_fp16_b16_wave-no_repl"        : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from fc1000/Softmax --executors wave 0 host 1 %s --batch 16 --images %s "%(rnPreFp16, MEv2("RN50"), getBatchedJpgs(16)), "--input_files %s" % (getBatchedJpgs(16))],
  "8-rn50_nne_fp16_b16_wave"        : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from fc1000/Softmax --executors wave 0 host 1 %s --batch 16 --images %s "%(rnPreFp16, MEv2("RN50-Repl"), getBatchedJpgs(16)), "--input_files %s" % (getBatchedJpgs(16))],
  "8-rn50_nne_fp16_b16_wave-two_banks"        : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from fc1000/Softmax --executors wave 0 host 1 %s --batch 16 --images %s "%(rnPreFp16, MEv2("RN50-Repl"), getBatchedJpgs(16)), "--env SIM_ADD_FLAGS=' --dram_frequency 6400' --input_files %s" % (getBatchedJpgs(16))],
  "8-rn50_nne_fp16_b16_wave-fast_dram"        : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from fc1000/Softmax --executors wave 0 host 1 %s --batch 16 --images %s "%(rnPreFp16, MEv2("RN50-Repl"), getBatchedJpgs(16)), "--env SIM_ADD_FLAGS=' --dram_frequency 0 --dram_latency 1' --input_files %s" % (getBatchedJpgs(16))],
  #"7-rn50_nne_fp16_wave-two_banks"        : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from fc1000/Softmax --executors wave 0 host 1  --scheduler wave2 --images %s" %(rnPreFp16, rnDogJpg), "--input_files %s" % rnDogJpg,  ],
  #"7-rn50_nne_fp16_wave-fast_dram"        : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from fc1000/Softmax --executors wave 0 host 1  --scheduler wave2 --images %s" %(rnPreFp16, rnDogJpg), "--input_files %s" % rnDogJpg,  ],
  "7-rn50_nne_fp16_host"        : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from fc1000/Softmax --executors host all --batch 2 --images %s %s" % (rnPreFp16, rnDogJpg, rnCatJpg), "--input_files %s %s" % (rnDogJpg, rnCatJpg)],
  "7-rn50_nne_fc_wave"          : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from avg_pool/AvgPool --executors wave 0 host 1 %s --images %s " %(rnPreFp16, MEv2("RN50-Repl"), rnDogJpg), "--input_files %s" % rnDogJpg ],
  #"7-rn50_nne_fc_b16_wave"          : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from avg_pool/AvgPool --executors wave 0 host 1  --scheduler wave2 --batch 16 --images %s"%(rnPreFp16, getBatchedJpgs(16)), "--input_files %s" % (getBatchedJpgs(16))],
  "8-rn50_nne_conv_wave"        : [ "tf_pb", "resnet50_keras/resnet50_fp32_keras_opt.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition conv --executors host all wave 01 03 04 07 09 12 14 16 19 21 23 26 27 30 32 35 37 39 42 44 46 49 51 53 56 57 60 62 65 67 69 72 74 76 79 81 83 86 88 90 93 95 97 100 101 104 106 109 111 113 116 118 120  --scheduler wave2 --images %s " %(rnPreFp32, rnDogJpg), "--input_files %s" % rnDogJpg ],
  "7-rn50_nne_fp32_host"        : [ "tf_pb", "resnet50_keras/resnet50_fp32_keras_opt.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from fc1000/Softmax --executors host all --batch 2 --images %s" %(rnPreFp32, rnDogCatB2Fp32), "--input_files %s %s" % (rnDogJpg, rnCatJpg)],

  # Matmult
  "4-rn50_matmul_plus_softmax_fp32_wave"   : [ "tf_pb",   "resnet50_keras/resnet50_fp32_keras_opt2.pb","resnet50", " --input_node input_1  --depth 2  --debug 1 %s --partition from avg_pool/AvgPool --executors host 0 wave 1 --scheduler wave2 --images %s --wavegraph_checks structure data-race"% (rnPreFp32, rnDogJpg),"--input_files %s" % rnDogJpg ],
  "4-rn50_matmul_plus_softmax_fp16_wave"   : [ "tf_pb",   "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", " --input_node input_1  --depth 2  --debug 1 %s --partition from avg_pool/AvgPool --executors host 0 wave 1 --scheduler wave2 --images %s --wavegraph_checks structure data-race"% (rnPreFp32, rnDogJpg),"--input_files %s" % rnDogJpg ],
  #"4-rn50_matmul_fp32_waveopt"   : [ "tf_pb",   "resnet50_keras/resnet50_fp32_keras_opt.pb","resnet50", " --input_node input_1  --depth 2  --debug 1 %s --partition from avg_pool/AvgPool --executors host 0 waveopt 1 --scheduler wave2 --images %s"% (rnPreFp32, rnDogJpg),"--input_files %s" % rnDogJpg ],
  #"4-rn50_matmul_nosm_fp32_wave"   : [ "tf_pb",   "resnet50_keras/resnet50_fp32_keras_opt.pb","resnet50", " --input_node input_1  --depth 2  --debug 1 %s  --partition from avg_pool/AvgPool fc1000/Softmax --executors host 0 2 wave 1 --scheduler wave2 --images %s" %(rnPreFp32, rnDogJpg),"--input_files %s" % rnDogJpg ],
  "4-rn50_matmul_nosm_wave"   : [ "tf_pb",   "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", " --input_node input_1  --depth 2  --debug 1 %s  --partition from avg_pool/AvgPool fc1000/Softmax --executors host 0 2 wave 1 %s --images %s" %(rnPreFp16, MEv2("Generic"), rnDogJpg),"--input_files %s" % rnDogJpg ],
  #"4-rn50_matmul_nosm_b4_wave"   : [ "tf_pb",   "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", " --input_node input_1  --depth 2  --debug 1 %s  --partition from avg_pool/AvgPool fc1000/Softmax --executors host 0 2 wave 1 --scheduler wave2 --schedule_options ' '  --batch 4 --images %s"%(rnPreFp16, getBatchedJpgs(4)),"--input_files %s" % (getBatchedJpgs(4))],

  # Resnet50 batching
  #"7-rn50_nne_fp16_waveopt"   : [ "tf_pb",   "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 --partition from fc1000/Softmax --executors waveopt 0 host 1  --scheduler wave2 --batch 1 --images linspace1", ],
  #"7-rn50_nne_fp16_b2_waveopt"   : [ "tf_pb",   "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from fc1000/Softmax --executors waveopt 0 host 1  --scheduler wave2 --batch 2 --images %s %s"%(rnPreFp16, rnDogJpg, rnCatJpg), "--input_files %s %s" % (rnDogJpg, rnCatJpg)],
  #"7-rn50_nne_fp16_b16_waveopt"   : [ "tf_pb",   "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from fc1000/Softmax --executors waveopt 0 host 1  --scheduler wave2 --batch 16 --images %s"%(rnPreFp16, getBatchedJpgs(16)), "--input_files %s" % (getBatchedJpgs(16))],

  # Resnet50 bfloat16
  "7-rn50_nne_bfloat16_wave": [ "tf_s3", "s3://kaena-nn-models", "resnet50_bfp16infp16_keras_opt.pb", "--input_node input_1  --depth 2  --debug 1 %s --partition from fc1000/Softmax --executors wave 0 host 1 --wavegraph_transform 'shell=perl -i -p -e s/float16/bfloat16/g wavegraph.json' cleaner euler cleaner --images %s --euler_options '--max_events 230'" % (MEv2("RN50-no_verify"), rnBfloat16Npy), "--input_files %s --check_against_ref none" % rnBfloat16Npy ],


##################################################################
# act4 initial subgraphs

  "5-rn50_nne_to_act4_wave-no_repl-t1"     : ["tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50",
    "--focus 'conv1/kernel|input_1|conv1/BiasAdd|bn_conv1/batchnorm_1/sub/_104__cf__104|bn_conv1/batchnorm_1/add_1|activation_1/Relu|max_pooling2d_1/MaxPool|res2a_branch2a/kernel|res2a_branch2a/BiasAdd|bn2a_branch2a/batchnorm_1/sub/_50__cf__50|bn2a_branch2a/batchnorm_1/add_1' --input_node input_1  --depth 2 --show_op_name_in_kgraph  --debug 1 --preprocessor $KAENA_PATH/compiler/util/res50_preprocessor.py  --preprocessor-args '--data-type fp16' %s --images %s " %(MEv2("Generic"), rnDogJpg),
    "--input_files %s" % rnDogJpg],

  "5-rn50_nne_to_act4_wave-no_repl-t1_focus_to"     : ["tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50",
    ( "--focus_to bn2a_branch2a/batchnorm_1/add_1  --input_node input_1  --depth 2 "
    + " --show_op_name_in_kgraph  --debug 1 "
    + "--preprocessor $KAENA_PATH/compiler/util/res50_preprocessor.py  --preprocessor-args '--data-type fp16' "
    + "%s --images %s "
    )%(MEv2("Generic"), rnDogJpg),
    "--input_files %s" % rnDogJpg],

  ## This test has outputs on multiple output queues, Act and Pool.
  "5-rn50_nne_to_act4_wave-no_repl-t1_focus_to-qemu-all_layers"     : ["tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50",
    ( "--focus_to bn2a_branch2a/batchnorm_1/add_1  --input_node input_1  --depth 2 "
    + " --show_op_name_in_kgraph  --debug 1 "
    + "--preprocessor $KAENA_PATH/compiler/util/res50_preprocessor.py  --preprocessor-args '--data-type fp16' "
    + " --scheduler qemu_wave2 "
    + "%s --images %s "
    )%(MEv2("Generic-SaveAll"), rnDogJpg),
    "--input_files %s" % rnDogJpg],

  "5-rn50_nne_to_act4_wave-no_repl-t2"     : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50",
    "--focus 'conv1/kernel|input_1|conv1/BiasAdd|bn_conv1/batchnorm_1/sub/_104__cf__104|bn_conv1/batchnorm_1/add_1|activation_1/Relu|max_pooling2d_1/MaxPool|res2a_branch2a/kernel|res2a_branch2a/BiasAdd|bn2a_branch2a/batchnorm_1/sub/_50__cf__50|bn2a_branch2a/batchnorm_1/add_1|activation_2/Relu|res2a_branch2b/kernel|res2a_branch2b/BiasAdd|bn2a_branch2b/batchnorm_1/sub/_48__cf__48|bn2a_branch2b/batchnorm_1/add_1' --input_node input_1  --depth 2 --show_op_name_in_kgraph  --debug 1 --preprocessor $KAENA_PATH/compiler/util/res50_preprocessor.py  --preprocessor-args '--data-type fp16' %s --images %s" %(MEv2("Generic"), rnDogJpg),
    "--input_files %s" % rnDogJpg],

  "5-rn50_nne_to_act4_wave-no_repl-t3"     : ["tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50",
    "--focus 'conv1/kernel|input_1|conv1/BiasAdd|bn_conv1/batchnorm_1/sub/_104__cf__104|bn_conv1/batchnorm_1/add_1|activation_1/Relu|max_pooling2d_1/MaxPool|res2a_branch2a/kernel|res2a_branch2a/BiasAdd|bn2a_branch2a/batchnorm_1/sub/_50__cf__50|bn2a_branch2a/batchnorm_1/add_1|activation_2/Relu|res2a_branch2b/kernel|res2a_branch2b/BiasAdd|bn2a_branch2b/batchnorm_1/sub/_48__cf__48|bn2a_branch2b/batchnorm_1/add_1|activation_3/Relu|res2a_branch2c/kernel|res2a_branch2c/BiasAdd' --input_node input_1  --depth 2 --show_op_name_in_kgraph  --debug 1 --preprocessor $KAENA_PATH/compiler/util/res50_preprocessor.py  --preprocessor-args '--data-type fp16'  %s --images %s " %(MEv2("Generic"), rnDogJpg),
    "--input_files %s" % rnDogJpg],

  "5-rn50_nne_to_act4_wave-no_repl-t4"     : ["tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50",
    "--focus 'conv1/kernel|input_1|conv1/BiasAdd|bn_conv1/batchnorm_1/sub/_104__cf__104|bn_conv1/batchnorm_1/add_1|activation_1/Relu|max_pooling2d_1/MaxPool|res2a_branch1/kernel|res2a_branch1/BiasAdd|bn2a_branch1/batchnorm_1/sub/_102__cf__102|bn2a_branch1/batchnorm_1/add_1' --input_node input_1  --depth 2 --show_op_name_in_kgraph  --debug 1 --preprocessor $KAENA_PATH/compiler/util/res50_preprocessor.py  --preprocessor-args '--data-type fp16'  %s --images %s " %(MEv2("Generic"), rnDogJpg),
    "--input_files %s" % rnDogJpg],

  "5-rn50_nne_to_act4_wave-no_repl-t5"     : ["tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50",
    "--focus 'conv1/kernel|input_1|conv1/BiasAdd|bn_conv1/batchnorm_1/sub/_104__cf__104|bn_conv1/batchnorm_1/add_1|activation_1/Relu|max_pooling2d_1/MaxPool|res2a_branch2a/kernel|res2a_branch2a/BiasAdd|bn2a_branch2a/batchnorm_1/sub/_50__cf__50|bn2a_branch2a/batchnorm_1/add_1|activation_2/Relu|res2a_branch2b/kernel|res2a_branch2b/BiasAdd|bn2a_branch2b/batchnorm_1/sub/_48__cf__48|bn2a_branch2b/batchnorm_1/add_1|activation_3/Relu|res2a_branch2c/kernel|res2a_branch2c/BiasAdd|res2a_branch1/kernel|res2a_branch1/BiasAdd|bn2a_branch1/batchnorm_1/sub/_102__cf__102|bn2a_branch1/batchnorm_1/add_1|bn2a_branch2c/batchnorm_1/add_1' --input_node input_1  --depth 2 --show_op_name_in_kgraph  --debug 1 --preprocessor $KAENA_PATH/compiler/util/res50_preprocessor.py  --preprocessor-args '--data-type fp16'  %s --images %s " %(MEv2("Generic"),  rnDogJpg),
    "--input_files %s" % rnDogJpg],

  "5-rn50_nne_to_act4_wave-no_repl-t8"     : ["tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50",
    "--focus 'conv1/kernel|input_1|conv1/BiasAdd|bn_conv1/batchnorm_1/sub/_104__cf__104|bn_conv1/batchnorm_1/add_1' --input_node input_1  --depth 2 --show_op_name_in_kgraph  --debug 1 --preprocessor $KAENA_PATH/compiler/util/res50_preprocessor.py  --preprocessor-args '--data-type fp16'  %s --images %s " %(MEv2("Generic"),  rnDogJpg),
    "--input_files %s" % rnDogJpg],

  "5-rn50_nne_to_act4_wave-no_repl-t9"     : ["tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50",
    "--focus 'conv1/kernel|input_1|conv1/BiasAdd' --input_node input_1  --depth 2 --show_op_name_in_kgraph  --debug 1 --preprocessor $KAENA_PATH/compiler/util/res50_preprocessor.py  --preprocessor-args '--data-type fp16'  %s --images %s " %(MEv2("Generic"), rnDogJpg),
    "--input_files %s" % rnDogJpg],

  "5-rn50_nne_to_act4_wave-no_repl-t9_focus_to"     : ["tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50",
    "--focus_to conv1/BiasAdd --input_node input_1  --depth 2 --show_op_name_in_kgraph  --debug 1 --preprocessor $KAENA_PATH/compiler/util/res50_preprocessor.py  --preprocessor-args '--data-type fp16' %s --images %s " %(MEv2("Generic"), rnDogJpg),
    "--input_files %s" % rnDogJpg],

##################################################################


  # Multi-tpb
  "7-rn50_fp16_multi_tpb_o_host"        : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--show_op_name_in_kgraph --input_node input_1  --depth 2  --debug 1 %s --partition multi_tpb ops 6.7 --executors host all host 7  --scheduler wave2 --images %s --wavegraph_checks structure data-race" %(rnPreFp16, rnDogJpg), "--input_files %s" % rnDogJpg ],
  "7-rn50_fp16_multi_tpb_w_host"        : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--show_op_name_in_kgraph --input_node input_1  --depth 2  --debug 1 %s --partition multi_tpb weights 4 --executors host all host 7  --scheduler wave2 --images %s --wavegraph_checks structure data-race" %(rnPreFp16, rnDogJpg), "--input_files %s" % rnDogJpg ],

  # LSTM
  "4-ptb_word_lm1_host"      : [ "tf_pb",   "ptb_word_lm/keras_unrolled/model-b16s32h512.pb","lm", " --input_node embedding_1_input_1  --depth 3  --debug 0   --partition from  lstm_2_1/transpose_1  --executors host all --scheduler wave2 --schedule_options ' --nname=lm' --input_constants dropout_1/keras_learning_phase:0 False --exclude_ops_from_capture ^dropout_1_1/cond/ --images %s --wavegraph_checks structure data-race" % lstmD0T32, "--input_files %s" % lstmD0T32],
  "4-ptb_word_lm1"   : [ "tf_pb",   "ptb_word_lm/keras_unrolled/model-b16s32h512.pb", "lm", " --input_node input_1  --depth 3  --debug 0 %s --partition from avg_pool/AvgPool --executors host 0 wave 1 --scheduler wave2 --schedule_options ' --nname=lm' --images %s --wavegraph_checks structure data-race"% (rnPreFp32, rnDogJpg), "--input_files %s" % rnDogJpg ],
  "4-ptb_word_small1_host"   : [ "tf_pb",   "ptb_word_lm/keras_unrolled/model-b32s4h512.pb","lm", " --input_node embedding_1_input_1  --depth 3  --debug 0   --partition from  lstm_2_1/transpose_1  --executors host all --scheduler wave2 --schedule_options ' --nname=lm' --input_constants dropout_1/keras_learning_phase:0 False --exclude_ops_from_capture ^dropout_1_1/cond/ --images %s --wavegraph_checks structure data-race" % lstmD0T4, "--input_files %s" % lstmD0T4],

  "4-ptb_word_small1_wave"   : [ "tf_pb",   "ptb_word_lm/keras_unrolled/model-b32s4h512.pb","lm",
    (" --input_node embedding_1_input_1 --sg_input_format lstm_1_1/transpose HNC lstm_2_1/transpose_1 HNC --depth 3  --debug 0    "
    + " --partition from_multi  lstm_1_1/unstack,lstm_1_1/Tile_1,lstm_1_1/Tile,lstm_1_1/Tile_1  lstm_1_1/stack  lstm_2_1/transpose_1  "
    + " --adjust_node_color lstm_1_1/Tile 0 lstm_1_1/Tile_1 0  "
    + " --show_op_name_in_kgraph --input_node embedding_1_input_1 "
    + " --executors host 0 2 wave 1  "
    + " --input_constants dropout_1/keras_learning_phase:0 False  "
    + " --exclude_ops_from_capture ^dropout_1_1/cond/  "
    + " %s --images %s  "
    )% (MEv2("Generic-SaveAll"), lstmD0T4), "--input_files %s" % lstmD0T4],

  ##########################################
  ## fp32,b32 on Tonga
  "4-ptb_word_small_sigmoid_wave"   : [ "tf_pb",   "ptb_word_lm/keras_unrolled/sigmoid/model-b32s4h512.pb","lm",
    (" --input_node embedding_1_input_1 --sg_input_format lstm_1_1/transpose HNC lstm_2_1/transpose_1 HNC --depth 3  "
    + " --partition from_multi  lstm_1_1/unstack,lstm_1_1/Tile_1,lstm_1_1/Tile,lstm_1_1/Tile_1  lstm_1_1/stack  lstm_2_1/transpose_1  "
    + " --adjust_node_color lstm_1_1/Tile 0 lstm_1_1/Tile_1 0  "
    + " --executors host 0 2 wave 1 "
    + " --input_constants dropout_1/keras_learning_phase:0 False  "
    + " --exclude_ops_from_capture ^dropout_1_1/cond/ "
    + " %s --images %s "
    ) % (
    " --scheduler wave2 "
    + " --schedule_options ' --nname=generic --enable_cleanup --save_layer_regex  lstm_1_1/transpose\|lstm_1_1/Tile\|lstm_1_1/Tile_1\|lstm_1_1/mul_2\|lstm_1_1/mul_5\|lstm_1_1/mul_8\|lstm_1_1/mul_11\|lstm_2_1/stack ' " ,
    lstmD0T4), "--input_files %s" % lstmD0T4],

  ## fp16,b32 on Tonga, model on s3
  "4-ptb_word_small_sigmoid_fp16_b32_wave"   : [ "tf_s3", "s3://kaena-nn-models/lstm_fp16", "model-b32s4h512_fp16.pb",
    (" --input_node embedding_1_input_1 --sg_input_format lstm_1_1/transpose HNC lstm_2_1/transpose_1 HNC --depth 3  "
    + " --partition from_multi  lstm_1_1/unstack,lstm_1_1/Tile_1,lstm_1_1/Tile,lstm_1_1/Tile_1  lstm_1_1/stack  lstm_2_1/transpose_1  "
    + " --adjust_node_color lstm_1_1/Tile 0 lstm_1_1/Tile_1 0  "
    + " --executors host 0 2 wave 1 "
    + " --input_constants dropout_1/keras_learning_phase:0 False  "
    + " --exclude_ops_from_capture ^dropout_1_1/cond/ "
    + " %s --images %s "
    ) % (
    " --scheduler wave2 "
    + " --schedule_options ' --nname=generic --enable_cleanup --save_layer_regex  lstm_1_1/transpose\|lstm_1_1/Tile\|lstm_1_1/Tile_1\|lstm_1_1/mul_2\|lstm_1_1/mul_5\|lstm_1_1/mul_8\|lstm_1_1/mul_11\|lstm_2_1/stack ' " ,
    lstmD0T4), "--input_files %s" % lstmD0T4
    ],

  ## fp16,b32 on host
  "4-ptb_word_small_sigmoid_fp16_b32_host"   : [ "tf_s3", "s3://kaena-nn-models/lstm_fp16", "model-b32s4h512_fp16.pb",
    (" --input_node embedding_1_input_1 --sg_input_format lstm_1_1/transpose HNC lstm_2_1/transpose_1 HNC --depth 3  "
    + " --partition from_multi  lstm_1_1/unstack,lstm_1_1/Tile_1,lstm_1_1/Tile,lstm_1_1/Tile_1  lstm_1_1/stack  lstm_2_1/transpose_1  "
    + " --adjust_node_color lstm_1_1/Tile 0 lstm_1_1/Tile_1 0  "
    + " --executors host 0 1 2 "
    + " --input_constants dropout_1/keras_learning_phase:0 False  "
    + " --exclude_ops_from_capture ^dropout_1_1/cond/ "
    + " %s --images %s "
    ) % (
    " --scheduler wave2 "
    + " --schedule_options ' --nname=generic --enable_cleanup --save_layer_regex  lstm_1_1/transpose\|lstm_1_1/Tile\|lstm_1_1/Tile_1\|lstm_1_1/mul_2\|lstm_1_1/mul_5\|lstm_1_1/mul_8\|lstm_1_1/mul_11\|lstm_2_1/stack ' " ,
    lstmD0T4), "--input_files %s" % lstmD0T4
    ],



  # LSTM small: 5 color 2-layer small host-tpb-host-tpb-host - waveopt and wave versions
  "4-ptb_word_small_sigmoid_2l_waveopt"  : [
    "tf_pb", "ptb_word_lm/keras_unrolled/sigmoid/model-b32s4h512.pb", "lm",
    (" --show_op_name_in_kgraph --input_node embedding_1_input_1 "
    + " --sg_input_format lstm_1_1/transpose HNC lstm_2_1/transpose HNC  --depth 3  --debug 1   "
    + " --partition from_multi  lstm_1_1/unstack,lstm_1_1/Tile_1,lstm_1_1/Tile,lstm_1_1/Tile_1  lstm_1_1/stack   lstm_2_1/unstack,lstm_2_1/Tile_1,lstm_2_1/Tile,lstm_2_1/Tile_1  lstm_2_1/stack     "
    + " --adjust_node_color  lstm_1_1/Tile 0 lstm_1_1/Tile_1 0 lstm_2_1/Tile 2 lstm_2_1/Tile_1 2   "
    + " --executors  waveopt 1 3  "
    + " --input_constants dropout_1/keras_learning_phase:0 False  "
    + " --exclude_ops_from_capture ^dropout_1_1/cond/ "
    + " %s --images %s "
    ) % (MEv2("Generic-SaveAll"), lstmD0T4), "--input_files %s" % lstmD0T4],

  ##########################################
  ## fp32,b32,2l on Tonga
  "4-ptb_word_small_sigmoid_2l_wave"  : [ "tf_pb",   "ptb_word_lm/keras_unrolled/sigmoid/model-b32s4h512.pb","lm",
  (" --show_op_name_in_kgraph --input_node embedding_1_input_1  "
  + " --sg_input_format lstm_1_1/transpose HNC lstm_2_1/transpose HNC  --depth 3  --debug 1    "
  + " --partition from_multi  lstm_1_1/unstack,lstm_1_1/Tile_1,lstm_1_1/Tile,lstm_1_1/Tile_1  lstm_1_1/stack   lstm_2_1/unstack,lstm_2_1/Tile_1,lstm_2_1/Tile,lstm_2_1/Tile_1  lstm_2_1/stack      "
  + " --adjust_node_color  lstm_1_1/Tile 0 lstm_1_1/Tile_1 0 lstm_2_1/Tile 2 lstm_2_1/Tile_1 2    "
  + " --executors  wave 1 3   "
  + " --input_constants dropout_1/keras_learning_phase:0 False   "
  + " --exclude_ops_from_capture ^dropout_1_1/cond/ "
  + " %s --images %s "
  ) % (
  " --scheduler wave2 "
  + " --schedule_options ' --nname=generic --enable_cleanup --save_layer_regex  lstm_1_1/transpose\|lstm_1_1/Tile_1\|lstm_1_1/Tile\|lstm_1_1/mul_2\|lstm_1_1/mul_5\|lstm_1_1/mul_8\|lstm_1_1/mul_11\|lstm_2_1/transpose\|lstm_2_1/Tile\|lstm_2_1/Tile_1\|lstm_2_1/mul_2\|lstm_2_1/mul_5\|lstm_2_1/mul_8\|lstm_2_1/mul_11 ' " ,
    lstmD0T4), "--input_files %s" % lstmD0T4],

  ## fp16,b32,2l on Tonga
  "4-ptb_word_small_sigmoid_2l_fp16_b32_wave"  : [ "tf_s3", "s3://kaena-nn-models/lstm_fp16", "model-b32s4h512_fp16.pb",
  (" --show_op_name_in_kgraph --input_node embedding_1_input_1  "
  + " --sg_input_format lstm_1_1/transpose HNC lstm_2_1/transpose HNC  --depth 3  --debug 1    "
  + " --partition from_multi  lstm_1_1/unstack,lstm_1_1/Tile_1,lstm_1_1/Tile,lstm_1_1/Tile_1  lstm_1_1/stack   lstm_2_1/unstack,lstm_2_1/Tile_1,lstm_2_1/Tile,lstm_2_1/Tile_1  lstm_2_1/stack      "
  + " --adjust_node_color  lstm_1_1/Tile 0 lstm_1_1/Tile_1 0 lstm_2_1/Tile 2 lstm_2_1/Tile_1 2    "
  + " --executors  wave 1 3   "
  + " --input_constants dropout_1/keras_learning_phase:0 False   "
  + " --exclude_ops_from_capture ^dropout_1_1/cond/ "
  + " %s --images %s "
  ) % (
  " --scheduler wave2 "
  + " --schedule_options ' --nname=generic --enable_cleanup --save_layer_regex  lstm_1_1/transpose\|lstm_1_1/Tile_1\|lstm_1_1/Tile\|lstm_1_1/mul_2\|lstm_1_1/mul_5\|lstm_1_1/mul_8\|lstm_1_1/mul_11\|lstm_2_1/transpose\|lstm_2_1/Tile\|lstm_2_1/Tile_1\|lstm_2_1/mul_2\|lstm_2_1/mul_5\|lstm_2_1/mul_8\|lstm_2_1/mul_11 ' " ,
  lstmD0T4), "--input_files %s" % lstmD0T4],

  ## fp16,b32,2l on host
  #"4-ptb_word_small_sigmoid_2l_fp16_b32_host"  : [ "tf_pb",
  #"ptb_word_lm/keras_unrolled/sigmoid/fp16/model-b32s4h512_fp16.pb","lm",
  #(" --show_op_name_in_kgraph --input_node embedding_1_input_1  "
  #+ " --sg_input_format lstm_1_1/transpose HNC lstm_2_1/transpose HNC  --depth 3  --debug 1    "
  #+ " --partition from_multi  lstm_1_1/unstack,lstm_1_1/Tile_1,lstm_1_1/Tile,lstm_1_1/Tile_1  lstm_1_1/stack   lstm_2_1/unstack,lstm_2_1/Tile_1,lstm_2_1/Tile,lstm_2_1/Tile_1  lstm_2_1/stack      "
  #+ " --adjust_node_color  lstm_1_1/Tile 0 lstm_1_1/Tile_1 0 lstm_2_1/Tile 2 lstm_2_1/Tile_1 2    "
  #+ " --executors  host 1 3   "
  #+ " --input_constants dropout_1/keras_learning_phase:0 False   "
  #+ " --exclude_ops_from_capture ^dropout_1_1/cond/ "
  #+ " %s --images %s "
  #) % (
  #" --scheduler wave2 "
  #+ " --schedule_options ' --nname=generic --enable_cleanup --save_layer_regex  lstm_1_1/transpose\|lstm_1_1/Tile_1\|lstm_1_1/Tile\|lstm_1_1/mul_2\|lstm_1_1/mul_5\|lstm_1_1/mul_8\|lstm_1_1/mul_11\|lstm_2_1/transpose\|lstm_2_1/Tile\|lstm_2_1/Tile_1\|lstm_2_1/mul_2\|lstm_2_1/mul_5\|lstm_2_1/mul_8\|lstm_2_1/mul_11 ' " ,
  #lstmD0T4), "--input_files %s" % lstmD0T4],

  ##########################################

  # Batched small LSTM
  "4-ptb_word_small_sigmoid_2l_b64_wave"  : [ "tf_pb",   "ptb_word_lm/keras_unrolled/sigmoid_b64/model-b64s4h512.pb","lm",
    (" --show_op_name_in_kgraph   --depth 3  --debug 1 --sg_input_format lstm_1_1/transpose HNC lstm_2_1/transpose_1 HNC    "
    + " --partition from_multi  lstm_1_1/unstack,lstm_1_1/Tile_1,lstm_1_1/Tile,lstm_1_1/Tile_1  lstm_1_1/stack   lstm_2_1/unstack,lstm_2_1/Tile_1,lstm_2_1/Tile,lstm_2_1/Tile_1  lstm_2_1/stack      "
    + " --adjust_node_color  lstm_1_1/Tile 0 lstm_1_1/Tile_1 0 lstm_2_1/Tile 2 lstm_2_1/Tile_1 2    "
    + " --executors  wave 1 3   "
    + " --input_constants dropout_1/keras_learning_phase:0 False   "
    + " --exclude_ops_from_capture ^dropout_1_1/cond/  "
    + " --batch 64  "
    + " %s --images %s "
    )% (MEv2("Generic-SaveAll"), lstmD0T4B64), "--input_files %s" % lstmD0T4B64],

  # LSTM small: levelauto partitioning
  "4-ptb_word_small_sigmoid_2l_auto_waveopt"  : [ "tf_pb",   "ptb_word_lm/keras_unrolled/sigmoid/model-b32s4h512.pb","lm", " --show_op_name_in_kgraph --input_node embedding_1_input_1 --sg_input_format lstm_2_1/transpose_1 HNC  --depth 3  --debug 1   --partition levelauto   --executors  waveopt 0 2 4 %s --input_constants dropout_1/keras_learning_phase:0 False  --exclude_ops_from_capture ^dropout_1_1/cond/ --images %s " %(MEv2("Generic-SaveAll"), lstmD0T4), "--input_files %s" % lstmD0T4],

   # LSTM debug of matmult, only sg00 is usable
  "2-ptb_word_unstack_matmul4"  : [ "tf_pb",   "ptb_word_lm/keras_unrolled/sigmoid/model-b32s4h512.pb","lm", " --show_op_name_in_kgraph --input_node embedding_1_input_1 --sg_input_format lstm_2_1/transpose_1 HNC  --depth 3  --debug 1   --partition from_multi  lstm_1_1/unstack,lstm_1_1/Tile_1,lstm_1_1/Tile,lstm_1_1/Tile_1   lstm_1_1/add_6,lstm_1_1/add_4,lstm_1_1/add_2,lstm_1_1/MatMul,lstm_1_1/MatMul_1,lstm_1_1/mul,lstm_1_1/mul_1  --adjust_node_color lstm_1_1/Tile 0 lstm_1_1/Tile_1 0  --executors wave 0 %s --input_constants dropout_1/keras_learning_phase:0 False  --exclude_ops_from_capture ^dropout_1_1/cond/ --images %s" %(MEv2("Generic-SaveAll"), lstmD0T4), "--input_files %s" % lstmD0T4],

   # LSTM debug of matmult, only sg00 is usable

  "2-ptb_word_unstack_matmul1"  : [
    "tf_pb",   "ptb_word_lm/keras_unrolled/sigmoid/model-b32s4h512.pb","lm",
    (" --show_op_name_in_kgraph "
    + " --input_node embedding_1_input_1 "
    + " --sg_input_format lstm_2_1/transpose_1 HNC "
    + " --depth 3  --debug 1   "
    + " --partition from_multi  lstm_1_1/unstack,lstm_1_1/Tile_1,lstm_1_1/Tile,lstm_1_1/Tile_1   lstm_1_1/add_6,lstm_1_1/add_4,lstm_1_1/add_2,lstm_1_1/MatMul,lstm_1_1/MatMul_1,lstm_1_1/mul,lstm_1_1/mul_1  "
    + " --adjust_node_color lstm_1_1/Tile 0 lstm_1_1/Tile_1 0 lstm_1_1/MatMul_2 0  lstm_1_1/MatMul_4 0  "
    + " --executors wave 0 "
    + MEv2("Generic")
    + " --input_constants dropout_1/keras_learning_phase:0 False  "
    + " --exclude_ops_from_capture ^dropout_1_1/cond/ "
    + " --images %s "
    )% lstmD0T4,
    "--input_files %s" % lstmD0T4
  ],

  #InceptionV3
#"8-inceptionv3_wave_dog_sg00_tpb" : ["tf_pb", "inceptionv3/inceptionv3_fp16_keras_opt.pb", "inceptionv3", "%s --partition from mixed0/concat --executors host 1 wave 0 --scheduler wave2 --schedule_options '--nname=generic --enable_cleanup ' --input_node input_1 --images %s" %(incPreFp16, rnDogJpg), "--input_files %s" % rnDogJpg],
"5-inceptionv3_wave_dog_sg00_tpb_upto_concat1" : ["tf_pb", "inceptionv3/inceptionv3_fp16_keras_opt.pb", "inceptionv3", "%s --partition from_multi conv2d_14/convolution,conv2d_16/convolution,conv2d_13/convolution,average_pooling2d_2/AvgPool --executors host 1 wave 0 %s --input_node input_1 --images %s" %(incPreFp16, MEv2("Generic"), rnDogJpg), "--input_files %s" % rnDogJpg],
"5-inceptionv3_wave_dog_sg00_tpb_upto_concat2" : ["tf_pb", "inceptionv3/inceptionv3_fp16_keras_opt.pb", "inceptionv3", "%s --partition from_multi conv2d_20/convolution,conv2d_21/convolution,conv2d_23/convolution,average_pooling2d_3/AvgPool --executors host 1 wave 0 %s --input_node input_1 --images %s" %(incPreFp16, MEv2("Generic"), rnDogJpg), "--input_files %s" % rnDogJpg],
"5-inceptionv3_wave_dog_sg00_tpb_upto_concat3" : ["tf_pb", "inceptionv3/inceptionv3_fp16_keras_opt.pb", "inceptionv3", "%s --partition from_multi conv2d_28/convolution,conv2d_27/convolution,max_pooling2d_3/MaxPool --executors host 1 wave 0 %s --input_node input_1 --images %s" %(incPreFp16, MEv2("Generic-NoVerify"), rnDogJpg), "--input_files %s" % rnDogJpg],
"5-inceptionv3_wave_dog_sg00_tpb_concat1_concat3" : ["tf_pb", "inceptionv3/inceptionv3_fp16_keras_opt.pb", "inceptionv3", "%s --partition from_multi conv2d_14/convolution,conv2d_16/convolution,conv2d_13/convolution,average_pooling2d_2/AvgPool conv2d_28/convolution,conv2d_27/convolution,max_pooling2d_3/MaxPool --executors host 0 2 wave 1 %s --input_node input_1 --images %s" %(incPreFp16, MEv2("Generic-NoVerify"), rnDogJpg), "--input_files %s" % rnDogJpg],
"5-inceptionv3_wave_dog_sg00_tpb_concat2_concat3" : ["tf_pb", "inceptionv3/inceptionv3_fp16_keras_opt.pb", "inceptionv3", "%s --partition from_multi conv2d_20/convolution,conv2d_21/convolution,conv2d_23/convolution,average_pooling2d_3/AvgPool conv2d_28/convolution,conv2d_27/convolution,max_pooling2d_3/MaxPool --executors host 0 2 wave 1 %s --input_node input_1 --images %s" %(incPreFp16, MEv2("Generic"), rnDogJpg), "--input_files %s" % rnDogJpg],
"5-inceptionv3_wave_dog_sg00_tpb_concat3_concat4" : ["tf_pb", "inceptionv3/inceptionv3_fp16_keras_opt.pb", "inceptionv3", "%s --partition from_multi conv2d_28/convolution,conv2d_27/convolution,max_pooling2d_3/MaxPool conv2d_31/convolution,conv2d_32/convolution,conv2d_35/convolution,average_pooling2d_4/AvgPool --executors host 0 2 wave 1 %s --input_node input_1 --images %s" %(incPreFp16, MEv2("Generic"), rnDogJpg), "--input_files %s" % rnDogJpg],
"5-inceptionv3_wave_dog_sg00_tpb_concat4_concat5" : ["tf_pb", "inceptionv3/inceptionv3_fp16_keras_opt.pb", "inceptionv3", "%s --partition from_multi conv2d_31/convolution,conv2d_32/convolution,conv2d_35/convolution,average_pooling2d_4/AvgPool conv2d_41/convolution,conv2d_42/convolution,conv2d_45/convolution,average_pooling2d_5/AvgPool --executors host 0 2 wave 1 %s --input_node input_1 --images %s" %(incPreFp16, MEv2("Generic"), rnDogJpg), "--input_files %s" % rnDogJpg],
"5-inceptionv3_wave_dog_sg00_tpb_concat8_concat9" : ["tf_pb", "inceptionv3/inceptionv3_fp16_keras_opt.pb", "inceptionv3", "%s --partition from_multi conv2d_73/convolution,conv2d_71/convolution,max_pooling2d_4/MaxPool conv2d_77/convolution,conv2d_78/convolution,conv2d_81/convolution,average_pooling2d_8/AvgPool --executors host 0 2 wave 1 %s --input_node input_1 --images %s" %(incPreFp16, MEv2("Generic"), rnDogJpg), "--input_files %s" % rnDogJpg],
"5-inceptionv3_wave_dog_sg00_tpb_concat9_concat10" : ["tf_pb", "inceptionv3/inceptionv3_fp16_keras_opt.pb", "inceptionv3", "%s --partition from_multi conv2d_77/convolution,conv2d_78/convolution,conv2d_81/convolution,average_pooling2d_8/AvgPool conv2d_86/convolution,conv2d_87/convolution,conv2d_90/convolution,average_pooling2d_9/AvgPool --executors host 0 2 wave 1 %s --input_node input_1 --images %s" %(incPreFp16, MEv2("Generic"), rnDogJpg), "--input_files %s" % rnDogJpg],
"5-inceptionv3_wave_dog_sg00_tpb_concat10_concat11" : ["tf_pb", "inceptionv3/inceptionv3_fp16_keras_opt.pb", "inceptionv3", "%s --partition from_multi conv2d_86/convolution,conv2d_87/convolution,conv2d_90/convolution,average_pooling2d_9/AvgPool avg_pool/Mean --executors host 0 2 wave 1 %s --input_node input_1 --images %s" %(incPreFp16, MEv2("Generic"), rnDogJpg), "--input_files %s" % rnDogJpg],
"6-inceptionv3_wave_dog_sg00_tpb_upto_concat4" : ["tf_pb", "inceptionv3/inceptionv3_fp16_keras_opt.pb", "inceptionv3", "%s --partition from_multi conv2d_31/convolution,conv2d_32/convolution,conv2d_35/convolution,average_pooling2d_4/AvgPool --executors host 1 wave 0 %s --input_node input_1 --images %s" %(incPreFp16, MEv2("Generic"), rnDogJpg), "--input_files %s" % rnDogJpg],
"6-inceptionv3_wave_dog_sg00_tpb_upto_concat5" : ["tf_pb", "inceptionv3/inceptionv3_fp16_keras_opt.pb", "inceptionv3", "%s --partition from_multi conv2d_41/convolution,conv2d_42/convolution,conv2d_45/convolution,average_pooling2d_5/AvgPool --executors host 1 wave 0 %s --input_node input_1 --images %s" %(incPreFp16, MEv2("Generic"), rnDogJpg), "--input_files %s" % rnDogJpg],
"6-inceptionv3_wave_dog_sg00_tpb_upto_concat8" : ["tf_pb", "inceptionv3/inceptionv3_fp16_keras_opt.pb", "inceptionv3", "%s --partition from_multi conv2d_71/convolution,conv2d_73/convolution,max_pooling2d_4/MaxPool --executors host 1 wave 0 %s --input_node input_1 --images %s " %(incPreFp16, MEv2("Generic-NoVerify"), rnDogJpg), "--input_files %s" % rnDogJpg],
"6-inceptionv3_wave_dog_sg00_tpb_upto_concat9" : ["tf_pb", "inceptionv3/inceptionv3_fp16_keras_opt.pb", "inceptionv3", "%s --partition from_multi conv2d_77/convolution,conv2d_78/convolution,conv2d_81/convolution,average_pooling2d_8/AvgPool --executors host 1 wave 0 --scheduler wave2 %s --input_node input_1 --images %s " %(incPreFp16, MEv2("Generic-NoVerify"), rnDogJpg), "--input_files %s" % rnDogJpg],
"7-inceptionv3_wave_dog_sg00_tpb_upto_concat10" : ["tf_pb", "inceptionv3/inceptionv3_fp16_keras_opt.pb", "inceptionv3", "%s --partition from_multi conv2d_86/convolution,conv2d_87/convolution,conv2d_90/convolution,average_pooling2d_9/AvgPool --executors host 1 wave 0 --scheduler wave2 %s --input_node input_1 --images %s " %(incPreFp16, MEv2("Generic-NoVerify"), rnDogJpg), "--input_files %s" % rnDogJpg],
"7-inceptionv3_wave_dog_sg00_tpb_upto_concat11" : ["tf_pb", "inceptionv3/inceptionv3_fp16_keras_opt.pb", "inceptionv3", "%s --partition from avg_pool/Mean --executors host 1 wave 0 --input_node input_1 %s --images %s --euler_options '--max_events 220'" %(incPreFp16, MEv2("Generic-NoVerify"), rnDogJpg), "--input_files %s" % rnDogJpg],

  "0-1conv1maxpool_wave_k3d1"  : [ "trivnet_conv_pool", "tfloat16-b1-h4-r1-s1-c1-m1-VALID-MaxPool-k3-d1-wmin2-wmax2.2-imin1-imax16", "1conv1pool", MEv2("Generic")],
  "0-1conv1maxpool_wave_h17k3d1"  : [ "trivnet_conv_pool", "tfloat16-b1-h17-r1-s1-c1-m1-VALID-MaxPool-k3-d1-wmin2-wmax2.2-imin1-imax16", "1conv1pool", MEv2("Generic")],
  "0-1conv1maxpool_wave_h17c128k3d1"  : [ "trivnet_conv_pool", "tfloat16-b1-h17-r1-s1-c128-m1-VALID-MaxPool-k3-d1-wmin2-wmax2.2-imin1-imax16", "1conv1pool", MEv2("Generic")],
  "0-1conv1maxpool_wave_h17c196k3d1"  : [ "trivnet_conv_pool", "tfloat16-b1-h17-r1-s1-c196-m1-VALID-MaxPool-k3-d1-wmin2-wmax2.2-imin1-imax16", "1conv1pool", MEv2("Generic")],
  "0-1conv1maxpool_wave_h71k3d1"  : [ "trivnet_conv_pool", "tfloat16-b1-h71-r1-s1-c1-m1-VALID-MaxPool-k3-d1-wmin2-wmax2.2-imin1-imax16", "1conv1pool", MEv2("Generic")],
  #"0-1conv1maxpool_wave_h71c192k3d1"  : [ "trivnet_conv_pool", "tfloat16-b1-h71-r1-s1-c192-m1-VALID-MaxPool-k3-d1-wmin2-wmax2.2-imin1-imax16", "1conv1pool", MEv2("Generic")],
  "0-1maxpool_wave_h71c192m192k3d1_valid"  : [ "trivnet_pool", "tfloat16-b1-h71-r1-s1-c192-m192-VALID-MaxPool-k3-d1-wmin-0.1-wmax0.2-imin-0.2-imax0.3", "1pool", MEv2("Generic")],
  "0-1maxpool_wave_h71c192m192k3d1_same"  : [ "trivnet_pool", "tfloat16-b1-h71-r1-s1-c192-m192-SAME-MaxPool-k3-d1-wmin-0.1-wmax0.2-imin-0.2-imax0.3", "1pool", MEv2("Generic")],
  "0-rtl-1avgpool_wave_h71c192m192k3d1"  : [ "trivnet_pool", "tfloat16-b1-h71-r1-s1-c192-m192-SAME-AvgPool-k3-d1-wmin-0.1-wmax0.2-imin-0.2-imax0.3", "1pool", MEv2("Generic")],
  "0-rtl-1avgpool_wave_h71c1m1k3d1_same"  : [ "trivnet_pool", "tfloat16-b1-h71-r1-s1-c1-m1-SAME-AvgPool-k3-d1-wmin-0.1-wmax0.2-imin-0.2-imax0.3", "1pool", MEv2("Generic")],
  "0-rtl-1avgpool_wave_h1c192m192k1d1_same"  : [ "trivnet_pool", "tfloat16-b1-h1-r1-s1-c192-m192-SAME-AvgPool-k1-d1-wmin-0.1-wmax0.2-imin-0.2-imax0.3", "1pool", MEv2("Generic")],
  "0-1avgpool_wave_h71c1m1k3d1_valid"  : [ "trivnet_pool", "tfloat16-b1-h71-r1-s1-c1-m1-VALID-AvgPool-k3-d1-wmin-0.1-wmax0.2-imin-0.2-imax0.3", "1pool", MEv2("Generic")],
  "0-1avgpool_wave_h1c192m192k1d1_valid"  : [ "trivnet_pool", "tfloat16-b1-h1-r1-s1-c192-m192-VALID-AvgPool-k1-d1-wmin-0.1-wmax0.2-imin-0.2-imax0.3", "1pool", MEv2("Generic")],
  "0-1maxpool_wave_h71c1m1k3d1_same"  : [ "trivnet_pool", "tfloat16-b1-h71-r1-s1-c1-m1-SAME-MaxPool-k3-d1-wmin-0.1-wmax0.2-imin-0.2-imax0.3", "1pool", MEv2("Generic")],
  "0-1maxpool_wave_h71c1m1k3d2_valid"  : [ "trivnet_pool", "tfloat16-b1-h71-r1-s1-c1-m1-VALID-MaxPool-k3-d2-wmin-0.1-wmax0.2-imin-0.2-imax0.3", "1pool", MEv2("Generic")],
  "0-1maxpool_wave_h71c1m1k3d2_same"  : [ "trivnet_pool", "tfloat16-b1-h71-r1-s1-c1-m1-SAME-MaxPool-k3-d2-wmin-0.1-wmax0.2-imin-0.2-imax0.3", "1pool", MEv2("Generic")],
  "0-1maxpool_wave_h71c192m192k3d2_valid"  : [ "trivnet_pool", "tfloat16-b1-h71-r1-s1-c192-m192-VALID-MaxPool-k3-d2-wmin-0.1-wmax0.2-imin-0.2-imax0.3", "1pool", MEv2("Generic")],
  "0-1maxpool_wave_h1c192m192k1d1_same"  : [ "trivnet_pool", "tfloat16-b1-h1-r1-s1-c192-m192-SAME-MaxPool-k1-d1-wmin-0.1-wmax0.2-imin-0.2-imax0.3", "1pool", MEv2("Generic")],
  "0-1maxpool_wave_h71c1m1k3d1_valid"  : [ "trivnet_pool", "tfloat16-b1-h71-r1-s1-c1-m1-VALID-MaxPool-k3-d1-wmin-0.1-wmax0.2-imin-0.2-imax0.3", "1pool", MEv2("Generic")],
  "0-1maxpool_wave_h1c192m192k1d1_valid"  : [ "trivnet_pool", "tfloat16-b1-h1-r1-s1-c192-m192-VALID-MaxPool-k1-d1-wmin-0.1-wmax0.2-imin-0.2-imax0.3", "1pool", MEv2("Generic")],
  # Start of AvgPools in InceptionV3
  "0-1avgpool_wave_h35c192m192k3d1_same"  : [ "trivnet_pool", "tfloat16-b1-h35-r1-s1-c192-m192-SAME-AvgPool-k3-d1-wmin-0.1-wmax0.2-imin-0.2-imax0.3", "1pool", MEv2("Generic")],
  "0-1avgpool_wave_h35c256m256k3d1_same"  : [ "trivnet_pool", "tfloat16-b1-h35-r1-s1-c256-m256-SAME-AvgPool-k3-d1-wmin-0.1-wmax0.2-imin-0.2-imax0.3", "1pool", MEv2("Generic")],
  "0-1avgpool_wave_h35c288m288k3d1_same"  : [ "trivnet_pool", "tfloat16-b1-h35-r1-s1-c288-m288-SAME-AvgPool-k3-d1-wmin-0.1-wmax0.2-imin-0.2-imax0.3", "1pool", MEv2("Generic")],
  "0-1avgpool_wave_h17c768m768k3d1_same"  : [ "trivnet_pool", "tfloat16-b1-h17-r1-s1-c768-m768-SAME-AvgPool-k3-d1-wmin-0.1-wmax0.2-imin-0.2-imax0.3", "1pool", MEv2("Generic")],
  "0-1avgpool_wave_h8c1280m1280k3d1_same"  : [ "trivnet_pool", "tfloat16-b1-h8-r1-s1-c1280-m1280-SAME-AvgPool-k3-d1-wmin-0.1-wmax0.2-imin-0.2-imax0.3", "1pool", MEv2("Generic")],
  "0-1avgpool_wave_h8c2048m2048k3d1_same"  : [ "trivnet_pool", "tfloat16-b1-h8-r1-s1-c2048-m2048-SAME-AvgPool-k3-d1-wmin-0.1-wmax0.2-imin-0.2-imax0.3", "1pool", MEv2("Generic")],

  "0-1avgpool_wave_h8c2048m2048k3d1_same_qemu_sem"  : [
    "trivnet_pool",
    "tfloat16-b1-h8-r1-s1-c2048-m2048-SAME-AvgPool-k3-d1-wmin-0.1-wmax0.2-imin-0.2-imax0.3",
    "1pool",
    ( "--scheduler qemu_wave2 "
    + " --schedule_options ' --nname=generic ' "
    + " --be_options sync-with-semaphores "
    )
  ],

  # End of AvgPools in InceptionV3
  "0-1conv1avgpool_wave_h35c192m192k3d1"  : [ "trivnet_conv_pool", "tfloat16-b1-h35-r1-s1-c192-m192-SAME-AvgPool-k3-d1-wmin-0.1-wmax0.2-imin-0.2-imax0.3", "1conv1pool", MEv2("Generic")],
  "0-1conv1avgpool_wave_h35c128m128k3d1"  : [ "trivnet_conv_pool", "tfloat16-b1-h35-r1-s1-c128-m128-SAME-AvgPool-k3-d1-wmin-0.1-wmax0.2-imin-0.2-imax0.3", "1conv1pool", MEv2("Generic")],
  "0-1conv1avgpool_wave_k3d1"  : [ "trivnet_conv_pool", "tfloat16-b1-h4-r1-s1-c1-m1-SAME-AvgPool-k3-d1-wmin2-wmax2.2-imin1-imax16", "1conv1pool", MEv2("Generic")],
  "0-1conv1avgpool_wave_h17k3d1"  : [ "trivnet_conv_pool", "tfloat16-b1-h17-r1-s1-c1-m1-SAME-AvgPool-k3-d1-wmin2-wmax2.2-imin1-imax16", "1conv1pool", MEv2("Generic")],
  "0-1conv1avgpool_wave_h16k3d1"  : [ "trivnet_conv_pool", "tfloat16-b1-h16-r1-s1-c1-m1-SAME-AvgPool-k3-d1-wmin2-wmax2.2-imin1-imax16", "1conv1pool", MEv2("Generic")],
  "0-1conv1avgpool_wave_h18k3d1"  : [ "trivnet_conv_pool", "tfloat16-b1-h18-r1-s1-c1-m1-SAME-AvgPool-k3-d1-wmin2-wmax2.2-imin1-imax16", "1conv1pool", MEv2("Generic")],
  "0-1conv1avgpool_wave_h32k3d1"  : [ "trivnet_conv_pool", "tfloat16-b1-h32-r1-s1-c1-m1-SAME-AvgPool-k3-d1-wmin2-wmax2.2-imin1-imax16", "1conv1pool", MEv2("Generic")],
  "0-1conv1avgpool_wave_h35k3d1"  : [ "trivnet_conv_pool", "tfloat16-b1-h32-r1-s1-c1-m1-SAME-AvgPool-k3-d1-wmin2-wmax2.2-imin1-imax16", "1conv1pool", MEv2("Generic")],
  "0-1conv1avgpool_wave_h35c16k3d1"  : [ "trivnet_conv_pool", "tfloat16-b1-h32-r1-s1-c16-m1-SAME-AvgPool-k3-d1-wmin2-wmax2.2-imin1-imax16", "1conv1pool", MEv2("Generic")],
  "0-1conv1avgpool_wave_h35c196k3d1"  : [ "trivnet_conv_pool", "tfloat16-b1-h32-r1-s1-c196-m1-SAME-AvgPool-k3-d1-wmin2-wmax2.2-imin1-imax16", "1conv1pool", MEv2("Generic")],
  "0-1conv_h17c196r1_wave"  : [ "trivnet_conv1",  "tfloat16-b1-h17-r1-s1-c196-m1-wmin-0.1-wmax0.12-imin-0.2-imax0.25", "1conv", MEv2("Generic")],
  "0-1maxpool_wave_h65c1m1k3d1_valid"  : [ "trivnet_pool", "tfloat16-b1-h65-r1-s1-c1-m1-VALID-MaxPool-k3-d1-wmin-0.1-wmax0.2-imin-0.2-imax0.3", "1pool", MEv2("Generic")],

  "0-1conv-h8r3c4m4-relu_wave"              : [ "trivnet_lin",      "tfloat16-l2-b1-h8-r3-s1-c4-m4-relu-wmin-0.2-wmax0.4-imin-1000-imax1010", "1cr", MEv2("Generic")],
  "0-rtl-1conv1maxpool_h8r3c5m4_val_wave"    : [ "trivnet_conv_pool","tfloat16-b1-h8-r3-s1-c5-m4-VALID-MaxPool-k2-d2-wmin2-wmax2.2-imin1-imax16", "1conv1pool", MEv2("Generic")],


  "0-1conv1maxpool_c128m64h16_val_wave"    : [ "trivnet_conv_pool","tfloat16-b1-h16-r3-s1-c128-m64-VALID-MaxPool-k2-d2-wmin2-wmax2.2-imin1-imax16", "1conv1pool", MEv2("Generic")],
  "0-1conv1maxpool_c128m64h128_val_wave"    : [ "trivnet_conv_pool","tfloat16-b1-h128-r3-s1-c128-m64-VALID-MaxPool-k2-d2-wmin2-wmax2.2-imin1-imax16", "1conv1pool", MEv2("Generic")],

  "0-1conv_c128m64h128_wave"  : [ "trivnet_conv1",  "tfloat16-b1-h128-r3-s1-c128-m64-wmin2-wmax2.2-imin-1-imax1.6", "1conv", MEv2("Generic")],

  "0-11conv_tanh_wave" : [ "trivnet_lin",   "tfloat16-l11-b1-h4-r3-s1-c1-m1-tanh-wmin-0.2-wmax0.8-imin-4-imax8", "11ct", MEv2("Generic")],
  "0-2conv_tanh_wave" : [ "trivnet_lin",   "tfloat16-l2-b1-h4-r3-s1-c1-m1-tanh-wmin-0.2-wmax0.8-imin-4-imax8", "2ct", MEv2("Generic")],
  "0-3conv_h16r1c128m128_wave" : [ "trivnet_lin",   "tfloat16-l4-b1-h16-r1-s1-c128-m128-wmin-0.1-wmax0.1-imin-0.3-imax0.5", "3conv", MEv2("Generic")],

  ### AmoebaNet tests

  # AvgPool in AmeobaNet
  "0-1avgpool_wave_h149c1m1k1d2_valid"  : [ "trivnet_pool", "tfloat16-b1-h149-r1-s1-c1-m1-VALID-AvgPool-k1-d2-wmin-0.1-wmax0.2-imin-0.2-imax0.3", "1pool", MEv2("Generic")],
  "7-amoebanet_fp16_host" : [ "tf_s3", "s3://kaena-nn-models", "amoebanet_inference_graph_fp16.pb", "--input_node=transpose --partition from predictions --executors host all --preprocessor $KAENA_PATH/compiler/util/res50_preprocessor.py  --preprocessor-args '--data-type fp16 --size 299' --images %s" % (rnDogJpg), "--input_files %s" % (rnDogJpg)],
  "7-amoebanet_fp16_pool" : [ "tf_s3", "s3://kaena-nn-models", "amoebanet_inference_graph_fp16.pb", "--input_node=transpose --focus_to cell_stem_1/AvgPool_1 --partition from cell_stem_1/Relu %s --preprocessor $KAENA_PATH/compiler/util/res50_preprocessor.py  --preprocessor-args '--data-type fp32 --size 299' --images %s" % (MEv2("Generic"), rnDogJpg), "--input_files %s" % (rnDogJpg)],
  "7-amoebanet_fp16_evict" : [ "tf_s3", "s3://kaena-nn-models", "amoebanet_inference_graph_fp16.pb", "--input_node=transpose --partition from_multi cell_stem_1/Relu,cell_stem_1/Relu_1,cell_0/Relu --executors wave 0 host 1 %s --preprocessor $KAENA_PATH/compiler/util/res50_preprocessor.py  --preprocessor-args '--data-type fp32 --size 299' --images %s" % (MEv2("Generic"), rnDogJpg), "--input_files %s" % (rnDogJpg)],
  "7-amoebanet_fp16_cell0" : [ "tf_s3", "s3://kaena-nn-models", "amoebanet_inference_graph_fp16.pb", "--input_node=transpose --focus_to cell_0/cell_output/concat --partition from_multi cell_0/Relu,cell_0/Relu_1 --executors host all wave 1 %s --preprocessor $KAENA_PATH/compiler/util/res50_preprocessor.py  --preprocessor-args '--data-type fp16 --size 299' --images %s" % (MEv2("Generic"), rnDogJpg), "--input_files %s" % (rnDogJpg)],
  "7-amoebanet_fp16_cell1" : [ "tf_s3", "s3://kaena-nn-models", "amoebanet_inference_graph_fp16.pb", "--input_node=transpose --focus_to cell_1/cell_output/concat --partition from_multi cell_1/Relu,cell_1/Relu_1 --executors host all wave 1 %s --preprocessor $KAENA_PATH/compiler/util/res50_preprocessor.py  --preprocessor-args '--data-type fp16 --size 299' --images %s" % (MEv2("Generic"), rnDogJpg), "--input_files %s" % (rnDogJpg)],
  "7-amoebanet_fp16_cell2" : [ "tf_s3", "s3://kaena-nn-models", "amoebanet_inference_graph_fp16.pb", "--input_node=transpose --focus_to cell_2/cell_output/concat --partition from_multi cell_2/Relu,cell_2/Relu_1 --executors host all wave 1 %s --preprocessor $KAENA_PATH/compiler/util/res50_preprocessor.py  --preprocessor-args '--data-type fp16 --size 299' --images %s" % (MEv2("Generic"), rnDogJpg), "--input_files %s" % (rnDogJpg)],
  "7-amoebanet_fp16_cell3" : [ "tf_s3", "s3://kaena-nn-models", "amoebanet_inference_graph_fp16.pb", "--input_node=transpose --focus_to cell_3/cell_output/concat --partition from_multi cell_3/Relu,cell_3/Relu_1 --executors host all wave 1 %s --preprocessor $KAENA_PATH/compiler/util/res50_preprocessor.py  --preprocessor-args '--data-type fp16 --size 299' --images %s" % (MEv2("Generic"), rnDogJpg), "--input_files %s" % (rnDogJpg)],
  "7-amoebanet_fp16_cell4" : [ "tf_s3", "s3://kaena-nn-models", "amoebanet_inference_graph_fp16.pb", "--input_node=transpose --focus_to cell_4/cell_output/concat --partition from_multi cell_4/Relu,cell_4/Relu_1 --executors host all wave 1 %s --preprocessor $KAENA_PATH/compiler/util/res50_preprocessor.py  --preprocessor-args '--data-type fp16 --size 299' --images %s" % (MEv2("Generic"), rnDogJpg), "--input_files %s" % (rnDogJpg)],
  "7-amoebanet_fp16_cell5" : [ "tf_s3", "s3://kaena-nn-models", "amoebanet_inference_graph_fp16.pb", "--input_node=transpose --focus_to cell_5/cell_output/concat --partition from_multi cell_5/Relu,cell_5/Relu_1 --executors host all wave 1 %s --preprocessor $KAENA_PATH/compiler/util/res50_preprocessor.py  --preprocessor-args '--data-type fp16 --size 299' --images %s" % (MEv2("Generic"), rnDogJpg), "--input_files %s" % (rnDogJpg)],
  "7-amoebanet_fp16_cell6" : [ "tf_s3", "s3://kaena-nn-models", "amoebanet_inference_graph_fp16.pb", "--input_node=transpose --focus_to cell_6/cell_output/concat --partition from_multi cell_6/Relu,cell_6/Relu_1 --executors host all wave 1 %s --preprocessor $KAENA_PATH/compiler/util/res50_preprocessor.py  --preprocessor-args '--data-type fp16 --size 299' --images %s" % (MEv2("Generic"), rnDogJpg), "--input_files %s" % (rnDogJpg)],
  "7-amoebanet_fp16_cell7" : [ "tf_s3", "s3://kaena-nn-models", "amoebanet_inference_graph_fp16.pb", "--input_node=transpose --focus_to cell_7/cell_output/concat --partition from_multi cell_7/Relu,cell_7/Relu_1 --executors host all wave 1 %s --preprocessor $KAENA_PATH/compiler/util/res50_preprocessor.py  --preprocessor-args '--data-type fp16 --size 299' --images %s" % (MEv2("Generic"), rnDogJpg), "--input_files %s" % (rnDogJpg)],
  "7-amoebanet_fp16_cell8" : [ "tf_s3", "s3://kaena-nn-models", "amoebanet_inference_graph_fp16.pb", "--input_node=transpose --focus_to cell_8/cell_output/concat --partition from_multi cell_8/Relu,cell_8/Relu_1 --executors host all wave 1 %s --preprocessor $KAENA_PATH/compiler/util/res50_preprocessor.py  --preprocessor-args '--data-type fp16 --size 299' --images %s" % (MEv2("Generic"), rnDogJpg), "--input_files %s" % (rnDogJpg)],
  "7-amoebanet_fp16_cell9" : [ "tf_s3", "s3://kaena-nn-models", "amoebanet_inference_graph_fp16.pb", "--input_node=transpose --focus_to cell_9/cell_output/concat --partition from_multi cell_9/Relu,cell_9/Relu_1 --executors host all wave 1 %s --preprocessor $KAENA_PATH/compiler/util/res50_preprocessor.py  --preprocessor-args '--data-type fp16 --size 299' --images %s --euler_options '--max_events 220'" % (MEv2("Generic-WaiveWC"), rnDogJpg), "--input_files %s" % (rnDogJpg)],
  "7-amoebanet_fp16_cell10" : [ "tf_s3", "s3://kaena-nn-models", "amoebanet_inference_graph_fp16.pb", "--input_node=transpose --focus_to cell_10/cell_output/concat --partition from_multi cell_10/Relu,cell_10/Relu_1 --executors host all wave 1 %s --preprocessor $KAENA_PATH/compiler/util/res50_preprocessor.py  --preprocessor-args '--data-type fp16 --size 299' --images %s" % (MEv2("Generic"), rnDogJpg), "--input_files %s" % (rnDogJpg)],
  "7-amoebanet_fp16_cell11" : [ "tf_s3", "s3://kaena-nn-models", "amoebanet_inference_graph_fp16.pb", "--input_node=transpose --focus_to cell_11/cell_output/concat --partition from_multi cell_11/Relu,cell_11/Relu_1 --executors host all wave 1 %s --preprocessor $KAENA_PATH/compiler/util/res50_preprocessor.py  --preprocessor-args '--data-type fp16 --size 299' --images %s --euler_options '--max_events 210'" % (MEv2("Generic"), rnDogJpg), "--input_files %s" % (rnDogJpg)],
  "7-amoebanet_fp16_rcell0" : [ "tf_s3", "s3://kaena-nn-models", "amoebanet_inference_graph_fp16.pb", "--input_node=transpose --focus_to reduction_cell_0/cell_output/concat --partition from_multi reduction_cell_0/Relu,reduction_cell_0/Relu_1 --executors host all wave 1 %s --preprocessor $KAENA_PATH/compiler/util/res50_preprocessor.py  --preprocessor-args '--data-type fp16 --size 299' --images %s" % (MEv2("Generic"), rnDogJpg), "--input_files %s" % (rnDogJpg)],
  "7-amoebanet_fp16_rcell1" : [ "tf_s3", "s3://kaena-nn-models", "amoebanet_inference_graph_fp16.pb", "--input_node=transpose --focus_to reduction_cell_1/cell_output/concat --partition from_multi reduction_cell_1/Relu,reduction_cell_1/Relu_1 --executors host all wave 1 %s --preprocessor $KAENA_PATH/compiler/util/res50_preprocessor.py  --preprocessor-args '--data-type fp16 --size 299' --images %s" % (MEv2("Generic"), rnDogJpg), "--input_files %s" % (rnDogJpg)],

  #
  # watchpoint tests
  #

  # c < 128
  "0-watchpoint-3conv_b1_h4_c32_m32_relu_wave" : [ "trivnet_lin",    "tfloat16-l3-b1-h4-r1-s1-c32-m32-relu-wmin-0.2-wmax0.4-imin-10-imax11", "10cr", " --wavegraph_transform cleaner watchpoint 10cr/relu1 euler cleaner" + MEv2("Generic"), "--check_against_ref all_available"],
  # c, m = 128
  "0-watchpoint-3conv_b1_h14_c128_m128_relu_wave" : [ "trivnet_lin",    "tfloat16-l3-b1-h14-r1-s1-c128-m128-relu-wmin-0.2-wmax0.4-imin-10-imax11", "10cr", " --wavegraph_transform cleaner watchpoint 10cr/relu1 euler cleaner" + MEv2("Generic"), "--check_against_ref all_available"],
  # c, m > 128
  "0-watchpoint-3conv_b1_h4_c256_m256_relu_wave" : [ "trivnet_lin",    "tfloat16-l3-b1-h4-r1-s1-c256-m256-relu-wmin-0.2-wmax0.4-imin-10-imax11", "10cr", " --wavegraph_transform cleaner watchpoint 10cr/relu1 euler cleaner" + MEv2("Generic"), "--check_against_ref all_available"],
  # b=2
  "0-watchpoint-2conv3_relu_b2_h20_m128_wave" : [ "trivnet_lin",    "tfloat16-l3-b2-h20-r1-s1-c128-m128-relu-wmin-0.2-wmax0.24-imin-10-imax11", "1conv3", " --wavegraph_transform cleaner watchpoint 1conv3/relu1 euler cleaner " + MEv2("RN50"), "--check_against_ref all_available"],
  # rn50 tests
  "5-watchpoint-rn50_nne_to_act4_wave-repl"     : [
    "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50",
    "--input_node input_1  --depth 2  --debug 1 %s --partition from activation_4/Relu --executors wave 0 host 1 %s --images %s --wavegraph_transform cleaner watchpoint activation_3/Relu euler cleaner" %(rnPreFp16, MEv2("RN50-Repl"), rnDogJpg),
    "--input_files %s --check_against_ref all_available" % rnDogJpg
  ],
  "5-watchpoint-rn50_nne_to_act13_wave-no_repl"     : [
    "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb",
    "resnet50",
    "--input_node input_1  --depth 2  --debug 1 %s --partition from activation_13/Relu --executors wave 0 host 1 %s --images %s --wavegraph_transform cleaner watchpoint 'activation_3/Relu activation_9/Relu' euler cleaner" %(rnPreFp16, MEv2("RN50"), rnDogJpg),
    "--input_files %s --check_against_ref all_available" % rnDogJpg
  ],

  # transformer tests

  "0-transformer-matmul": [
    "tf_s3", "s3://kaena-nn-models", "transformer_frozen_fp16.pb",
    "--input_node infer_x infer_y --depth 2 "
    "--focus_to 'infer_preds' "
    "--partition from_multi '"
      "dense/Tensordot/MatMul,dense/Tensordot/concat_2"
      "' '"
      "ArgMax"
      "' "
    "--dot_timeout 1 "
    "--executors host 0 2 wave 1 --scheduler wave2 --schedule_options ' --nname=generic' --images {} --wavegraph_checks structure data-race".format(transformerIn), "--input_files {}".format(transformerRtIn)
  ],

  "0-transformer-mul": [
    "tf_s3", "s3://kaena-nn-models", "transformer_frozen_fp16.pb",
    "--input_node infer_x --depth 2 "
    "--focus_to 'encoder/num_blocks_1/multihead_attention/Sum' "
    "--partition from_multi '"
      "encoder/num_blocks_0/multihead_attention_1/ln/add_1"
      "' '"
      "encoder/num_blocks_0/multihead_attention_1/ln/mul"
      "' "
    "--dot_timeout 1 "
    "--executors host 0 2 wave 1 --scheduler wave2 --schedule_options ' --nname=generic' --images {} --wavegraph_checks structure data-race".format(transformerInX), "--input_files {}".format(transformerRtInX)
  ],

  #WaveRNN Tests #cb-compiler bug workaround - no support for constants to Sub and Mul operators - WA to remove sub operatoins in sigmoid and change Mul constants to placeholders
  "2-wavernn_tf_ts0_w1_h02_cb"     : [
    #"tf_pb", "wavernn/wavernn_ts1_cb_seed1_f_opt_fp16.pb","wavernn",
    "tf_s3", "s3://kaena-nn-models", "wavernn_ts1_cb_seed1_f_opt_fp16.pb",
    "--input_node prev cond init_state --depth -1 --focus_to Softmax  --show_op_name_in_kgraph "
    #"--input_node prev cond init_state --depth -1 --focus_to add_13 multinomial/Multinomial  --show_op_name_in_kgraph "
    "--images $KAENA_EXT_PATH/apps/tf/wavernn/prev_samp.npy $KAENA_EXT_PATH/apps/tf/wavernn/cond.npy $KAENA_EXT_PATH/apps/tf/wavernn/init_state.npy "
    " --executors host 0 2 wave 1 --scheduler qemu_wave2 --schedule_options ' --nname=generic' --wavegraph_checks structure data-race --parallel_streams --partition from_multi MatMul,MatMul_2,MatMul_4,MatMul_3,MatMul_1,Mul,Mul_1 Softmax ",
    "--input_files prev:0=$KAENA_EXT_PATH/apps/tf/wavernn/prev_samp.npy cond:0=$KAENA_EXT_PATH/apps/tf/wavernn/cond.npy init_state:0=$KAENA_EXT_PATH/apps/tf/wavernn/init_state.npy "
  ],

  "5-transformer-encoder": [
    "tf_s3", "s3://kaena-nn-models",
    "transformer_infer_encoder_v2_fp16.pb",
    "--input_node transformer_infer_encoder/encoder_inputs --depth 2 "
    "--partition from_multi '"
      "transformer_infer_encoder/encoder_attention_bias/bias"
      "','"
      "transformer_infer_encoder/encoder_embedding/embedding_and_positional"
      "' "
      "--dot_timeout 1 "
      "--executors host 0 wave 1 {} --images {} --wavegraph_checks structure data-race".format(MEv2("generic"), transformerEncoderIn),
    "--input_files {}".format(transformerEncoderRtIn)
   ],
}

def gen_rn50_nne_to_act_norepl(act_num, batch):
    return [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50",
    "--input_node input_1  --depth 2  --debug 1 %s --partition from activation_%d/Relu --executors wave 0 host 1 %s --batch %d --images %s "%(rnPreFp16, act_num, MEv2("RN50"), batch, getBatchedJpgs(batch)),
    "--input_files %s" % (getBatchedJpgs(batch))]

def gen_rn50_nne_to_act_repl(act_num, batch):
    return [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50",
    "--input_node input_1  --depth 2  --debug 1 %s --partition from activation_%d/Relu --executors wave 0 host 1 %s --batch %d --images %s "%(rnPreFp16, act_num, MEv2("RN50-Repl"), batch, getBatchedJpgs(batch)),
    "--input_files %s" % (getBatchedJpgs(batch))]

for i in [22, 25, 28, 37, 40, 43, 46, 49]:
    testConfigMap["6-rn50_nne_to_act%d_wave-no_repl"%i] = gen_rn50_nne_to_act_norepl(i, 1)
    testConfigMap["6-rn50_nne_to_act%d_wave-repl"%i] = gen_rn50_nne_to_act_repl(i, 1)

# In order to get add_24 in subgraph, has to cut graph before sub_1, and also mark all 2D nodes with NW format
def gen_parwavenet_10_fp16_in_to2(node, sgnum):
    return  [ "tf_pb", "parallel_wavenet/example1/parwavenet_10_frozen_fp16.pb", "parallel_wavenet",
          "--level_order_seed 0 --input_node Placeholder Log Log_1 --focus_to %s --show_op_name_in_kgraph --depth -1 "%node
          + "--sg_input_format Log NW Log_1 NW sub_1 NW mul_13 NW add_24 NW Squeeze_36 NW Squeeze_35 NW "
          + "--executors host all wave %d  --partition from_multi sub_1,Reshape "%sgnum
          + "%s --images %s linspace1 linspace1" % (MEv2("Generic-NoVerify"), melSpectra),
          "--input_files Placeholder:0=%s Log:0=trivnet_Log:0.npy Log_1:0=trivnet_Log_1:0.npy " % melSpectra]

for i in [1, 2, 3, 6]:
    testConfigMap["5-parwavenet_10_fp16_in_to_add%s_wave"%i] = gen_parwavenet_10_fp16_in_to2("add_%s"%i, 1)

for i in [9, 12, 15, 18, 22]:
    testConfigMap["6-parwavenet_10_fp16_in_to_add%s_wave"%i] = gen_parwavenet_10_fp16_in_to2("add_%s"%i, 1)

for i in [24]:
    testConfigMap["7-parwavenet_10_fp16_in_to_add%s_wave"%i] = gen_parwavenet_10_fp16_in_to2("add_%s"%i, 1)

# Deconvolution requires very larger weights files, which can be fit into circular buffer, but this causes too many descriptors (8x maximum).
# The following subgraphs starts after deconvolution to avoid descriptor explosion, so we can run on QEMU/EMU
def gen_parwavenet_10_fp16_tanh1_to(node, sgnum):
    return  [ "tf_pb", "parallel_wavenet/example1/parwavenet_10_frozen_fp16.pb", "parallel_wavenet",
          "--level_order_seed 3 --input_node Placeholder Log Log_1 --focus_to %s --show_op_name_in_kgraph --depth -1 "%node
          + "--sg_input_format Log NW Log_1 NW sub_1 NW mul_13 NW add_24 NW Squeeze_36 NW Squeeze_35 NW "
          + "--executors host all wave %d  --partition from_multi sub_1,Tanh_1 "%sgnum
          + "%s --images %s linspace1 linspace1" % (MEv2("Generic-NoVerify"), melSpectra),
          "--input_files Placeholder:0=%s Log:0=trivnet_Log:0.npy Log_1:0=trivnet_Log_1:0.npy " % melSpectra]

for i in [1, 2, 3, 6]:
    testConfigMap["5-parwavenet_10_fp16_tanh_to_add%s_wave"%i] = gen_parwavenet_10_fp16_tanh1_to("add_%s"%i, 1)

# kaena-943: For tanh_to_add6, Slice_4 kept only 1536 elements from 2000 elements of BiasAdd_13 output (similarly for the others)
for i in [2, 6]:
    testConfigMap["5-parwavenet_10_fp16_tanh_to_add%s_wave"%i][TFFE_OPTION_IDX] += " --waive_wavegraph_checks "
    testConfigMap["5-parwavenet_10_fp16_in_to_add%s_wave"%i][TFFE_OPTION_IDX] += " --waive_wavegraph_checks "

for i in [9, 12, 15, 18, 22]:
    testConfigMap["6-parwavenet_10_fp16_tanh_to_add%s_wave"%i] = gen_parwavenet_10_fp16_tanh1_to("add_%s"%i, 1)

for i in [24]:
    testConfigMap["7-parwavenet_10_fp16_tanh_to_add%s_wave"%i] = gen_parwavenet_10_fp16_tanh1_to("add_%s"%i, 1)

def gen_7_rn50_nne_fp16_wave_no_repl_save(layer_name):
    return [
      "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50",
    ( "--input_node input_1  --depth 2  --debug 1 %s --partition from fc1000/Softmax "
      + " --executors wave 0 host 1  --scheduler wave2 --images %s"
      + " --schedule_options ' --save_layer_regex %s ' "%layer_name
    ) %(rnPreFp16, rnDogJpg),
    "--input_files %s --check_against_ref all_available" % rnDogJpg ]

# Tests for per activation layer debug on emulator
#for i in range(49):
#    testConfigMap["7-rn50_nne_fp16_wave-no_repl-save-act%s"%i] = gen_7_rn50_nne_fp16_wave_no_repl_save("activation_%s"%i)

# Tests for rcell0 debug (kaena-972)
#for i in [1,2,3,4,5]:
#    testConfigMap["7-amoebanet_fp16_rcell0_save_last%s"%i] = [ "tf_s3",
#            "s3://kaena-nn-models", "amoebanet_inference_graph_fp16.pb",
#            "--input_node=transpose --focus_to reduction_cell_0/cell_output/concat --partition from_multi reduction_cell_0/Relu,reduction_cell_0/Relu_1 --executors host all wave 1 %s --preprocessor $KAENA_PATH/compiler/util/res50_preprocessor.py  --preprocessor-args '--data-type fp16 --size 299' --images %s" % ("--scheduler wave2 --schedule_options \' --save_layer_output -%s \' "%i, rnDogJpg), "--input_files %s" % (rnDogJpg)]

# Regression waiver mechanism
# If the testname matches the regexp then the FAIL status is replaced with
# with the string
testWaiver = [
    ['0-1conv1maxpool_k3d2_wave',   'WAIVE_WAVESC'],
    ['0-1conv1pool_b5_wave',        'WAIVE_WAVESC'],
    ['0-1conv1pool_b5m3_wave',      'WAIVE_WAVESC'],
    ['5-inceptionv3_wave_dog_sg00_tpb_concat1_concat3$', 'WAIVE_INCEPTIONV3'],
    ['5-inceptionv3_wave_dog_sg00_tpb_upto_concat[3]$', 'WAIVE_INCEPTIONV3'],
    ['6-inceptionv3_wave_dog_sg00_tpb_upto_concat[89]$', 'WAIVE_INCEPTIONV3'],
    ['7-inceptionv3_wave_dog_sg00_tpb_upto_concat10$', 'WAIVE_INCEPTIONV3'],
    ['7-inceptionv3_wave_dog_sg00_tpb_upto_concat11$', 'WAIVE_INCEPTIONV3'],
    ['8-inceptionv3_wave_dog_sg00_tpb$', 'WAIVE_INCEPTIONV3'],
    ['0-1conv1maxpool_wave_h17c196k3d1', 'WAIVE_INCEPTIONV3'],
    ['0-1maxpool_wave_h65c1m1k3d1_valid', 'WAIVE_INCEPTIONV3'],
    ['0-1maxpool_wave_h71c1m1k3d2_same', 'WAIVE_INCEPTIONV3'],

    # AmoebaNet currently produces all NaN outputs when run on CPU
    ['7-amoebanet_fp16_host', 'WAIVE_AMOEBA_NAN'],
    ['7-amoebanet_fp16_pool', 'WAIVE_AMOEBA_POOL'],
    ['7-amoebanet_fp16_evict', 'WAIVE_AMOEBA_SBEVICT'],
    ['0-1conv_evict_wave', 'WAIVE_AMOEBA_SBEVICT'], # added by taemin

    # Parallel wavenet
    #['.*clipbyvalue.*', 'WAIVE_KAENA636'],
    #['.*softplus.*', 'WAIVE_KAENA634'],
    ['.*squeeze.*', 'WAIVE_KAENA634'],
    ['0-1transpose_wave', 'WAIVE_KAENA711'],
    ['0-1stridedslice_tanh_sigmoid_wave', 'WAIVE_KAENA711'],
    ['0-1stridedslice_wave', 'WAIVE_KAENA711'],
    #['.*reshape.*', 'WAIVE_KAENA597'],
    ['3-parwavenet_.*_waveopt$', 'WAIVE_KAENA711'],
    ['3-1conv_transpose_1d_h100r80s20_wave', 'WAIVE_KAENA768'],
    ['3-1conv_transpose_1d_h10r40s10_wave', 'WAIVE_KAENA768'],
    ['6-parwavenet_10_fp16_in_to_add15_wave$', 'WAIVE-KAENA902'],
    ['6-parwavenet_10_fp16_in_to_add18_wave$', 'WAIVE-KAENA902'],

    #['^0-act_wave$',   'WAIVE-KAENA452'],

    # UINT8 support
    ['^0-resadd_uint8_wave$', 'WAIVE-UINT8'],
    ['0-resadd_2in_wave', 'WAIVE-2INPUTS'],

    ['4-rn50_matmul_plus_softmax_fp32_wave$',      'WAIVE-S10_BE_SOFTMAX'],
    ['4-rn50_matmul_plus_softmax_fp16_wave$',      'WAIVE-S10_BE_SOFTMAX'],

    ['^[6]-alexnet',  'WAIVE-BENCH'],
    #['7-rn50_nne_fc_wave$', 'WAIVE-WAVESC'],

    ['^[8]-resnet152',  'WAIVE-BENCH'],
    ['^[8]-resnet50',  'WAIVE-BENCH'],
    ['8-rn50_nne_auto', 'WAIVE-NNE'],

    # ME accuracy failure
    #['0-116conv_tanh_wave', 'WAIVE-ME_ACC'],

    # LSMT
    ['4-ptb_word_lm1_host$', 'WAIVE-LSTM_HOST'],
    ['4-ptb_word_lm1$', 'WAIVE-LSTM'],
    ['4-ptb_word_small1_wave$', 'WAIVE-LSTM'],
    ['2-ptb_word_unstack_.*',             'WAIVE-LSTM'],
    ['4-ptb_word_small_sigmoid_2l_auto_waveopt',   'WAIVE-L_PART'],
    ['4-ptb_word_small_sigmoid_2l_b64_wave',   'WAIVE-LSTM_ME'],
    ['4-ptb_word_small_sigmoid_2l_fp16_b32_wave', 'WAIVE-LSTM-NUMERICAL'],

    # Multi-tpb partitioning - passes in host mode so no need to waive
    #['7-rn50_fp16_multi_tpb_o_wave', 'WAIVE_MTPB'],
    #['7-rn50_fp16_multi_tpb_w_wave', 'WAIVE_MTPB'],

    # batching
    #['7-rn50_nne_fp16_waveopt_b\d+$', 'WAIVE_BATCH'],
    ['7-rn50_nne_fp32_wave$', 'WAIVE_SB_PRESERVE'],
    ['8-rn50_nne_fp32_meauto$', 'WAIVE_SB_PRESERVE'],

    # bugs
    ['0-act_exp_sweep_wave', 'WAIVE-KAENA773'],
    #['0-act_lrelu_sweep_wave', 'WAIVE-KAENA773'],
    ['0-act_fused_lrelu_sweep_wave', 'WAIVE-KAENA773'],
    ['0-act_exp_minisweep_wave', 'WAIVE-KAENA773'],
    #['0-act_lrelu_minisweep_wave', 'WAIVE-KAENA773'],
    ['0-act_fused_lrelu_minisweep_wave', 'WAIVE-KAENA773'],
    ['7-rn50_nne_fp16_wave-no_repl-all-layers$', 'WAIVE_KAENA734'],

    # Replication
    ['3-h128r2s1c6m32_wave_repl', 'WAIVE_KAENA817'],
    ['3-h128r2s1c4m32_wave_repl', 'WAIVE_KAENA817'],
    ['3-h128r2s1c2m32_wave_repl', 'WAIVE_KAENA817'],
    ['3-h128r2s1c1m32_wave_repl', 'WAIVE_KAENA817'],
    ['3-h128r3s1c6m32_wave_repl', 'WAIVE_KAENA817'],

    # Resnet 152
    ['^9-resnet152', 'WAIVE_RN152'],
    #['0-10conv_relu_wave', 'WAIVE_BUG_KAENA411'],
    #['0-3conv_relu_wave', 'WAIVE_BUG_KAENA411'],

    # Qemu only works on C5 (till we add VDI to S3)
    #['^0-1conv0_qemu_wave$', 'WAIVE_QEMU'],

    # Transformer
    # Comment calling kp.reportOpAndSizes() to see ME failure
    ['0-transformer-matmul', 'WAIVE_KAENA964'],
    ['0-transformer-mul', 'WAIVE_KAENA961'],
    ['5-transformer-encoder', 'WAIVE_KAENA974']


  ]

noGpuTestWaiver = [
]

qemuTestWaiver = [
    ['7-rn50_nne_fp16_wave-no_repl-save-last-n-layers', 'WAIVE-QEMU-KAENA922'],
    ['8-rn50_nne_conv_wave$',  'WAIVE-QEMU'],
    ['5-rn50_nne_to_act4_b16_wave-repl$',  'WAIVE-QEMU-KAENA1001'],
    ['8-rn50_nne_fp16_b16_wave-fast_dram$',  'WAIVE-QEMU-KAENA1001'],
    ['8-rn50_nne_fp16_b16_wave-two_banks$',  'WAIVE-QEMU-KAENA1001'],
    ['8-rn50_nne_fp16_b16_wave$',  'WAIVE-QEMU-KAENA1001'],
    ['8-rn50_nne_fp16_b16_wave-no_repl', 'WAIVE-V49850304'],
    ['7-rn50_nne_fp16_wave-no_repl-all-layers', 'WAIVE-ManyDescr-SIM742'],
    ['0-300conv_tanh_wave-all-layers', 'WAIVE-TooManyOfmaps-SIM746'],
]
