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

melSpectra = "%s/%s" % (kePath, "apps/tf/parallel_wavenet/melspec_input.npy") 

testConfigMap = {

  "0-rtl-1conv_wave"                   : [ "trivnet_conv1",    "tfloat16-b1-h1-r1-s1-c1-m1-wmin2-wmax2.2-imin3-imax3.2",         "1conv" , "--scheduler wave2"],
  "0-rtl-1conv_m8_wave"                : [ "trivnet_conv1",    "tfloat16-b1-h1-r1-s1-c1-m8-wmin2-wmax2.2-imin3-imax3.2",         "1conv" , "--scheduler wave2"],
  "0-rtl-1conv_h2c16_wave"             : [ "trivnet_conv1",    "tfloat16-b1-h2-r1-s1-c16-m1-wmin2-wmax2.2-imin1-imax7",          "1conv" , "--scheduler wave2"],
  "0-rtl-1conv_h4r2c8m4_wave"          : [ "trivnet_conv1",    "tfloat16-b1-h4-r2-s1-c8-m4-wmin-0.1-wmax0.12-imin-0.2-imax0.25", "1conv" , "--scheduler wave2"],
  "0-rtl-1conv_h4r3c4m2_wave"          : [ "trivnet_conv1",    "tfloat16-b1-h4-r3-s1-c4-m2-wmin0-wmax9-imin0-imax15",            "1conv" , "--scheduler wave2"],
  "0-rtl-1conv_relu_wave"              : [ "trivnet_lin",      "tfloat16-l2-b1-h4-r3-s1-c1-m1-relu-wmin-0.2-wmax0.4-imin-1000-imax1010", "1cr", "--scheduler wave2"],
  "0-rtl-1conv_h8r4c64m16_wave"        : [ "trivnet_conv1",    "tfloat16-b1-h8-r4-s1-c64-m16-wmin0-wmax9-imin0-imax15",          "1conv" , "--scheduler wave2"],
  "0-rtl-1conv3_h4r3_relu_wave"        : [ "trivnet_lin",      "tfloat16-l2-b1-h4-r3-s1-c1-m1-relu-wmin-0.2-wmax0.24-imin-10000-imax10100", "1conv3", "--scheduler wave2"],
  "0-rtl-1conv1avgpool_b1h4_same_wave" : [ "trivnet_conv_pool","tfloat16-b1-h4-r1-s1-c1-m1-SAME-AvgPool-k2-d2-wmin2-wmax2.2-imin1-imax16", "1conv1pool", "--scheduler wave2"],
  "0-rtl-1conv1maxpool_h4_val_wave"    : [ "trivnet_conv_pool","tfloat16-b1-h4-r1-s1-c1-m1-VALID-MaxPool-k2-d2-wmin2-wmax2.2-imin1-imax16", "1conv1pool", "--scheduler wave2"],
  "0-rtl-act_h2c16_wave"               : [ "trivnet_act",      "tfloat16-b1-h2-c16-tanh-wmin2-wmax2.2-imin-1-imax2",                        "act", "--scheduler wave2"],


  "0-1conv0_wave" : [ "trivnet_conv1",  "tfloat16-b1-h1-r1-s1-c1-m1-wmin2-wmax2.2-imin3-imax3.2", "1conv", "--scheduler wave2 --wavegraph_checks structure data-race"],
  "0-1conv0_wave_h35c288m64" : [ "trivnet_conv1",  "tfloat16-b1-h35-r1-s1-c288-m64-wmin0.1-wmax0.2-imin0.2-imax0.3", "1conv", "--scheduler wave2 --schedule_options ' --nname=generic' --wavegraph_checks structure data-race"],
  "0-1conv0_ckpt_wave" : [ "ckpt_conv1",  "tfloat16-b1-h1-r1-s1-c1-m1-wmin2-wmax2.2-imin3-imax3.2", "1conv", "--scheduler wave2 --wavegraph_checks structure data-race --show_op_name_in_kgraph --exclude_ops_from_capture 'save|Save|restore' --debug 1"],
  "0-1conv0_qemu_wave" : [ "trivnet_conv1",  "tfloat16-b1-h1-r1-s1-c1-m1-wmin2-wmax2.2-imin3-imax3.2", "1conv", "--scheduler qemu_wave --wavegraph_checks structure data-race"],
  "0-1conv0_b16_wave" : [ "trivnet_conv1",  "tfloat16-b16-h1-r1-s1-c1-m1-wmin2-wmax2.2-imin3-imax3.2", "1conv", "--scheduler wave2 --wavegraph_checks structure data-race"],
  "0-1conv0m4_wave" : [ "trivnet_conv1",  "tfloat16-b1-h1-r1-s1-c1-m4-wmin2-wmax2.2-imin3-imax3.2", "1conv", "--scheduler wave2 --wavegraph_checks structure data-race"],
  "0-1conv0m8_wave" : [ "trivnet_conv1",  "tfloat16-b1-h1-r1-s1-c1-m8-wmin2-wmax2.2-imin3-imax3.2", "1conv", "--scheduler wave2 --wavegraph_checks structure data-race"],
  "0-1conv0m16_wave" : [ "trivnet_conv1",  "tfloat16-b1-h1-r1-s1-c1-m16-wmin2-wmax2.2-imin3-imax3.2", "1conv", "--scheduler wave2 --wavegraph_checks structure data-race"],
  "0-1conv0_padvalid_wave" : [ "trivnet_conv1_padvalid",  "tfloat16-b1-h229-r7-s2-c1-m1-wmin2-wmax2.2-imin3-imax3.2", "1conv", "--scheduler wave2 --schedule_options ' --nname=generic' --debug 1"],
  "0-1conv0_h16r2s2_wave"      : [ "trivnet_conv1",  "tfloat16-b1-h16-r2-s2-c1-m1-wmin-2-wmax2.2-imin-3-imax3.3", "1conv", "--scheduler wave2 --wavegraph_checks structure data-race"],
  "0-1conv0_h16r3s2_wave"      : [ "trivnet_conv1",  "tfloat16-b1-h16-r3-s2-c1-m1-wmin-2-wmax2.2-imin-3-imax3.3", "1conv", "--scheduler wave2 --wavegraph_checks structure data-race"],
  "0-1conv0_c1h2_wave" : [ "trivnet_conv1",  "tfloat16-b1-h2-r1-s1-c1-m1-wmin2-wmax2.2-imin1-imax7", "1conv", "--scheduler wave2 --wavegraph_checks structure data-race"],
  "0-1conv0_c1h16_wave" : [ "trivnet_conv1",  "tfloat16-b1-h16-r1-s1-c1-m1-wmin2-wmax2.2-imin3-imax3.2", "1conv", "--scheduler wave2 --wavegraph_checks structure data-race"],
  "0-neg"         : [ "trivnet_conv2",  "b1-Zh1-r1-s1-c1-m1-wmin2-wmax3-imin5-imax5.5", "I_ALWAYS_FAIL"],
  "0-1conv_tile_wave"  : [ "trivnet_conv1",  "tfloat16-b1-h35-r3-s1-c1-m1-wmin-0.1-wmax0.12-imin-0.2-imax0.25", "1conv", "--scheduler wave2 --wavegraph_checks structure data-race"],

  "0-1conv_h4r1_wave"  : [ "trivnet_conv1",  "tfloat16-b1-h4-r1-s1-c1-m1-wmin-0.1-wmax0.12-imin-0.2-imax0.25", "1conv", "--scheduler wave2 --wavegraph_checks structure data-race"],
  "0-1conv_h4r1_b2_wave"  : [ "trivnet_conv1",  "tfloat16-b2-h4-r1-s1-c1-m1-wmin-0.1-wmax0.12-imin-0.2-imax0.25", "1conv", "--scheduler wave2 --wavegraph_checks structure data-race"],
  "0-1conv_b1h1r1s1c2m2_tile_wave"  : [ "trivnet_conv1",  "tfloat16-b1-h1-r1-s1-c2-m2-wmin-0.1-wmax0.12-imin-0.2-imax0.25", "1conv", "--scheduler wave2"],
  "0-1conv_h4r2s2_wave"  : [ "trivnet_conv1",  "tfloat16-b1-h4-r2-s2-c1-m1-wmin-0.1-wmax0.12-imin-0.2-imax0.25", "1conv", "--scheduler wave2 --wavegraph_checks structure data-race"],
  "0-1conv_h6r2s3_wave"  : [ "trivnet_conv1",  "tfloat16-b1-h6-r2-s3-c1-m1-wmin-0.1-wmax0.12-imin-0.2-imax0.25", "1conv", "--scheduler wave2 --wavegraph_checks structure data-race"],
  "0-1conv_h6r3s2_wave"  : [ "trivnet_conv1",  "tfloat16-b1-h6-r3-s2-c1-m1-wmin-0.1-wmax0.12-imin-0.2-imax0.25", "1conv", "--scheduler wave2 --wavegraph_checks structure data-race"],
  "0-1conv_h6r3s2_b2_wave"  : [ "trivnet_conv1",  "tfloat16-b2-h6-r3-s2-c1-m1-wmin-0.1-wmax0.12-imin-0.2-imax0.25", "1conv", "--scheduler wave2 --wavegraph_checks structure data-race"],
  "0-1conv_h4r3s1_wave"  : [ "trivnet_conv1",  "tfloat16-b1-h4-r3-s1-c1-m1-wmin0-wmax9-imin0-imax15", "1conv", "--scheduler wave2 --wavegraph_checks structure data-race"],
  "0-1conv_h4r3s1_b2_wave"  : [ "trivnet_conv1",  "tfloat16-b2-h4-r3-s1-c1-m1-wmin0-wmax9-imin0-imax15", "1conv", "--scheduler wave2 --wavegraph_checks structure data-race"],

  "0-1conv_tile_r1_wave"  : [ "trivnet_conv1",  "tfloat16-b1-h35-r1-s1-c1-m1-wmin-0.1-wmax0.12-imin-0.2-imax0.25", "1conv", "--scheduler wave2 --wavegraph_checks structure data-race"],
  "0-1conv_tile_r1h32_wave"  : [ "trivnet_conv1",  "tfloat16-b1-h32-r1-s1-c1-m1-wmin-0.1-wmax0.12-imin-0.2-imax0.25", "1conv", "--scheduler wave2 --wavegraph_checks structure data-race"],
  "0-1conv_tile_r1_e1_wave"  : [ "trivnet_conv1",  "tfloat16-b1-h35-r1-s1-c1-m1-F_31_31=3-wmin2-wmax2-imin-0-imax0", "1conv", "--scheduler wave2 --wavegraph_checks structure data-race"],
  #"0-2conv3_relu" : [ "trivnet_lin",    "tfloat16-l2-b1-h4-r3-s1-c1-m1-relu-wmin-0.2-wmax0.24-imin-10000-imax10100", "1conv3"],
  "0-3conv_1concat_host" : [ "trivnet_concat2",  "tfloat16-b1-h1-r1-s1-c1-m1-wmin2-wmax2.2-imin3-imax3.2", "1concat", "--scheduler wave2 --schedule_options ' --nname=generic --save_layer_output' --partition from 1concat/i3"],
  "0-3conv_1concat" : [ "trivnet_concat2",  "tfloat16-b1-h1-r1-s1-c1-m1-wmin2-wmax2.2-imin3-imax3.2", "1concat", "--scheduler wave2 --schedule_options ' --nname=generic --save_layer_output'"],
  "0-3conv_1concat_c32m32" : [ "trivnet_concat2",  "tfloat16-b1-h1-r1-s1-c32-m32-wmin2-wmax2.2-imin3-imax3.2", "1concat", "--scheduler wave2 --schedule_options ' --nname=generic --save_layer_output'"],
  "0-3conv_1concat_h16c32m32" : [ "trivnet_concat2",  "tfloat16-b1-h16-r1-s1-c32-m32-wmin2-wmax2.2-imin3-imax3.2", "1concat", "--scheduler wave2 --schedule_options ' --nname=generic --save_layer_output'"],
  "0-3conv_1concat_h16c32m63" : [ "trivnet_concat2",  "tfloat16-b1-h16-r1-s1-c32-m63-wmin2-wmax2.2-imin3-imax3.2", "1concat", "--scheduler wave2 --schedule_options ' --nname=generic --save_layer_output'"],
  "0-1concat_h1c1m1ni2" : [ "trivnet_concat_variable_inputs",  "tfloat16-b1-h1-r1-s1-c1-m1-ni2-wmin2-wmax2.2-imin3-imax3.2", "1concat", "--scheduler wave2 --schedule_options ' --nname=generic --save_layer_output'"],
  "0-1concat_h17c1m1ni2" : [ "trivnet_concat_variable_inputs",  "tfloat16-b1-h17-r1-s1-c1-m1-ni2-wmin2-wmax2.2-imin3-imax3.2", "1concat", "--scheduler wave2 --schedule_options ' --nname=generic --save_layer_output'"],
  "0-1concat_h1c1m10ni5" : [ "trivnet_concat_variable_inputs",  "tfloat16-b1-h1-r1-s1-c1-m10-ni5-wmin2-wmax2.2-imin3-imax3.2", "1concat", "--scheduler wave2 --schedule_options ' --nname=generic --save_layer_output'"],
  "0-1concat_h16c63m127ni5" : [ "trivnet_concat_variable_inputs",  "tfloat16-b1-h16-r1-s1-c63-m127-ni5-wmin2-wmax2.2-imin3-imax3.2", "1concat", "--scheduler wave2 --schedule_options ' --nname=generic --save_layer_output'"],
  "0-1concat_h35c1m1ni2" : [ "trivnet_concat_variable_inputs",  "tfloat16-b1-h35-r1-s1-c1-m1-ni2-wmin2-wmax2.2-imin3-imax3.2", "1concat", "--scheduler wave2 --schedule_options ' --nname=generic --save_layer_output'"],
  "0-1concat_h35c64m64ni4" : [ "trivnet_concat_variable_inputs",  "tfloat16-b1-h35-r1-s1-c64-m64-ni4-wmin2-wmax2.2-imin3-imax3.2", "1concat", "--scheduler wave2 --schedule_options ' --nname=generic --save_layer_output'"],
  "0-1concat_h35c63m127ni5" : [ "trivnet_concat_variable_inputs",  "tfloat16-b1-h35-r1-s1-c63-m127-ni5-wmin2-wmax2.2-imin3-imax3.2", "1concat", "--scheduler wave2 --schedule_options ' --nname=generic --save_layer_output'"],
  "0-2conv3_relu_wave" : [ "trivnet_lin",    "tfloat16-l2-b1-h4-r3-s1-c1-m1-relu-wmin-0.2-wmax0.24-imin-10000-imax10100", "1conv3", "--scheduler wave2 --wavegraph_checks structure data-race"],
  "0-2conv3_relu_b16_wave" : [ "trivnet_lin",    "tfloat16-l2-b16-h4-r3-s1-c1-m1-relu-wmin-0.2-wmax0.24-imin-10000-imax10100", "1conv3", "--scheduler wave2"],
  "3-rn50_relu_fp16_wave"  : [ "trivnet_lin","tfloat16-l2-b1-h224-r7-s2-c3-m3-relu-wmin-1-wmax1.1-imin-3-imax3.2", "2conv32b", "--scheduler wave2"],
  "3-rn50_ba_relu_fp16_wave"  : [ "trivnet_conv_ba","tfloat16-b1-h224-r7-s2-c3-m64-SAME-relu-wmin-1-wmax1.1-imin-3-imax3.2-amin-3-amax3", "2conv32b", "--scheduler wave2"],
  "0-ba_relu_fp32_wave"  : [ "trivnet_conv_ba","tfloat32-b1-h1-r1-s1-c1-m1-SAME-relu-wmin-2-wmax2-imin3-imax10-amin-7-amax7", "2conv32b", "--scheduler wave --wavegraph_checks structure data-race"],
  #"0-10conv_relu" : [ "trivnet_lin",    "tfloat16-l10-b1-h4-r3-s1-c1-m1-relu-wmin-0.2-wmax0.4-imin-1000-imax1010", "10cr"],
  "0-10conv_relu_wave" : [ "trivnet_lin",    "tfloat16-l10-b1-h4-r3-s1-c1-m1-relu-wmin-0.2-wmax0.4-imin-1000-imax1010", "10cr", "--scheduler wave2 --wavegraph_checks structure data-race"],
  "0-3conv_relu_wave" : [ "trivnet_lin",    "tfloat16-l3-b1-h1-r1-s1-c1-m1-relu-wmin-0.2-wmax0.4-imin-1000-imax1010", "10cr", "--scheduler wave2 --wavegraph_checks structure data-race"],
  #"0-116conv_tanh" : [ "trivnet_lin",   "tfloat16-l116-b1-h4-r3-s1-c1-m1-tanh-wmin-0.2-wmax0.8-imin-4-imax8", "116ct"],
  "0-116conv_tanh_wave" : [ "trivnet_lin",   "tfloat16-l116-b1-h4-r3-s1-c1-m1-tanh-wmin-0.2-wmax0.8-imin-4-imax8", "116ct", "--scheduler wave2 --wavegraph_checks structure data-race"],
  "0-1conv_s8_wave"    : [ "trivnet_conv1",  "tfloat16-b1-h16-r1-s8-c1-m1-wmin2-wmax22-imin1-imax256", "1conv", "--scheduler wave2 --wavegraph_checks structure data-race"],
  "0-1mp_r3s2_16_wave"  : [ "trivnet_mp1", "b1-h16-r3-s2-c1-m1-wmin0-wmax0.1-imin1-imax12544", "1mp", "--scheduler wave2"],
  "0-1conv1pool_wave"  : [ "trivnet_conv_pool", "tfloat16-b1-h4-r1-s1-c1-m128-SAME-MaxPool-k2-d2-wmin1-wmax1-imin0-imax127", "1conv1pool", "--scheduler wave2 --wavegraph_checks structure data-race"],
  "0-1conv1avgpool_wave"  : [ "trivnet_conv_pool", "tfloat16-b1-h4-r1-s1-c1-m1-SAME-AvgPool-k2-d2-wmin2-wmax2.2-imin1-imax16", "1conv1pool", "--scheduler wave2 --wavegraph_checks structure data-race"],
  "0-1conv1pool_b5_wave" : [ "trivnet_conv_pool", "tfloat16-b5-h4-r1-s1-c1-m1-SAME-AvgPool-k2-d2-wmin2-wmax2.2-imin1-imax80", "1conv1pool", "--scheduler wave --wavegraph_checks structure data-race"],
  "0-1conv1pool_b5m3_wave" : [ "trivnet_conv_pool", "tfloat16-b5-h4-r1-s1-c1-m3-SAME-AvgPool-k2-d2-wmin2-wmax2.2-imin1-imax240", "1conv1pool", "--scheduler wave --wavegraph_checks structure data-race"],
  "0-1conv1maxpool_k3d2_wave"  : [ "trivnet_conv_pool", "tfloat16-b16-h1-r3-s2-c1-m1-SAME-MaxPool-k3-d2-wmin-0.2-wmax0.3-imin-0.2-imax0.3", "1conv1pool", "--scheduler wave2 --schedule_options ' --nname=generic' --wavegraph_checks structure data-race"],
  
  # Conv, BiasAdd

  "0-1conv1ba1_h1c1m1_wave"  : [ "trivnet_conv_ba", "tfloat16-b1-h1-r1-s1-c1-m1-SAME-wmin2-wmax2.2-imin1-imax1.6-amin3-amax3.2", "1conv1ba", "--scheduler wave2 --schedule_options ' --nname=generic'"],

  "0-1conv1ba1_h4c1m256_wave"  : [ "trivnet_conv_ba", "tfloat16-b1-h4-r1-s1-c1-m256-SAME-wmin2-wmax2.2-imin1-imax1.6-amin3-amax3.2", "1conv1ba", "--scheduler wave2 --schedule_options ' --nname=generic'"],


  "0-1conv1ba1_wave"  : [ "trivnet_conv_ba", "tfloat16-b1-h4-r1-s1-c1-m256-SAME-wmin2-wmax2.2-imin1-imax16-amin3-amax3.2", "1conv1ba", "--scheduler wave2 --schedule_options ' --nname=generic'"],
  "0-1conv1ba1_h4c1m1_fp32_wave"  : [ "trivnet_conv_ba", "tfloat32-b1-h4-r1-s1-c1-m1-SAME-wmin2-wmax2.2-imin1-imax16-amin3-amax3.2", "1conv1ba", "--scheduler wave --wavegraph_checks structure data-race"],
  "0-1conv1ba1_h4c2m2_fp32_wave"  : [ "trivnet_conv_ba", "tfloat32-b1-h4-r1-s1-c2-m2-SAME-wmin2-wmax2.2-imin1-imax16-amin3-amax3.2", "1conv1ba", "--scheduler wave --wavegraph_checks structure data-race"],
  "0-1conv1ba1_h55c1m1_fp32_wave"  : [ "trivnet_conv_ba", "tfloat32-b1-h55-r1-s1-c1-m1-SAME-wmin2-wmax2.2-imin1-imax16-amin3-amax3.2", "1conv1ba", "--scheduler wave"],
  "0-1conv1ba1_h1c64m64_fp32_wave"  : [ "trivnet_conv_ba", "tfloat32-b1-h1-r1-s1-c64-m64-SAME-wmin2-wmax2.2-imin1-imax16-amin3-amax3.2", "1conv1ba", "--scheduler wave --wavegraph_checks structure data-race"],
  #"0-1conv1ba1_h55c64m64_fp32_wave"  : [ "trivnet_conv_ba", "tfloat32-b1-h55-r1-s1-c64-m64-SAME-wmin2-wmax2.2-imin1-imax16-amin3-amax3.2", "1conv1ba", "--scheduler wave --wavegraph_checks structure data-race"],
  "0-1conv1ba1_h4c2m2_wave"  : [ "trivnet_conv_ba", "tfloat16-b1-h4-r1-s1-c2-m2-SAME-wmin2-wmax2.2-imin1-imax16-amin3-amax3.2", "1conv1ba", "--scheduler wave --wavegraph_checks structure data-race"],
  "0-1conv1ba1_h55c1m1_wave"  : [ "trivnet_conv_ba", "tfloat16-b1-h55-r1-s1-c1-m1-SAME-wmin2-wmax2.2-imin1-imax16-amin3-amax3.2", "1conv1ba", "--scheduler wave2"],
  "0-1conv1ba1_h55c1m1_b2_wave"  : [ "trivnet_conv_ba", "tfloat16-b2-h55-r1-s1-c1-m1-SAME-wmin2-wmax2.2-imin1-imax16-amin3-amax3.2", "1conv1ba", "--scheduler wave2"],
  "0-1conv1ba1_h1c64m64_wave"  : [ "trivnet_conv_ba", "tfloat16-b1-h1-r1-s1-c64-m64-SAME-wmin2-wmax2.2-imin1-imax16-amin3-amax3.2", "1conv1ba", "--scheduler wave --wavegraph_checks structure data-race"],
  #"0-1conv1ba1_h55c64m64_wave"  : [ "trivnet_conv_ba", "tfloat16-b1-h55-r1-s1-c64-m64-SAME-wmin2-wmax2.2-imin1-imax16-amin3-amax3.2", "1conv1ba", "--scheduler wave --wavegraph_checks structure data-race"],
  "0-1conv1ba1_fp32_wave"  : [ "trivnet_conv_ba", "tfloat32-b1-h224-r7-s2-c3-m64-SAME-wmin1-wmax1-imin0-imax50175-amin-20000-amax-20000", "1conv1ba", "--scheduler wave"],

  "0-act_wave"     : [ "trivnet_act", "tfloat16-b1-h2-c128-tanh-wmin2-wmax2.2-imin-1-imax2", "act", "--scheduler wave2 --wavegraph_checks structure data-race"],
  "0-scaleadd_wave"       : [ "trivnet_scaleadd",    "tfloat16-b1-h1-c16-wmin2-wmax2.2-imin3-imax6", "scaleadd", "--scheduler wave --wavegraph_checks structure data-race"],
  "0-resadd_wave"         : [ "trivnet_add",    "tfloat16-b1-h2-c1-wmin2-wmax2.2-imin3-imax6", "add", "--scheduler wave2 --wavegraph_checks structure data-race"],
  "0-resadd_fp32_wave"    : [ "trivnet_add",    "tfloat32-b1-h17-c4-wmin-0.1-wmax0.11-imin1-imax5", "add", "--scheduler wave --wavegraph_checks structure data-race"],
  "0-resadd_uint8_wave"   : [ "trivnet_add",    "tuint8-b1-h4-c3-wmin1-wmax4-imin5-imax53", "add", "--scheduler wave --wavegraph_checks structure data-race"],


  "0-3resadd_fp16_wave"  : [ "trivnet_conv_ba_add",
    "tfloat16-b1-h4-r1-s1-c1-m1-SAME-wmin-1-wmax2-imin-0.1-imax0.3-amin-0.01-amax-0.03",
    "add", "--scheduler wave --schedule_options ' --nname=generic'  --partition from add/i3 --executor host 0 wave 1"
    ],

  "0-3conv_ba_resadd_fp32_wave"  : [ "trivnet_conv_ba_add", "tfloat32-b1-h55-r3-s1-c1-m1-SAME-wmin-1-wmax2-imin-0.1-imax0.3-amin-0.01-amax-0.03", "add", "--scheduler wave"],
  "0-3conv_ba_resadd_fp16_wave"  : [ "trivnet_conv_ba_add", "tfloat16-b1-h55-r3-s1-c1-m1-SAME-wmin-1-wmax2-imin-0.1-imax0.3-amin-0.01-amax-0.03", "add", "--scheduler wave2 --schedule_options ' --nname=generic'"],
  "0-3conv_ba_resadd_h1_fp16_wave"  : [ "trivnet_conv_ba_add", "tfloat16-b1-h1-r1-s1-c1-m1-SAME-wmin-1-wmax2-imin-0.1-imax0.3-amin-0.01-amax-0.03", "add", "--scheduler wave2 --schedule_options ' --nname=generic'"],
  "0-3conv_ba_mult_fp32_wave"  : [ "trivnet_conv_ba_mult", "tfloat32-b1-h55-r3-s2-c1-m1-SAME-wmin-1-wmax2-imin-0.1-imax0.3-amin-0.01-amax-0.03", "mult", "--scheduler wave"],
  "0-2matmult_add_fp32_wave"  : [ "trivnet_matmul_add", "tfloat32-b1-h1-r1-s1-c512-m2048-SAME-wmin-1-wmax2-imin-0.1-imax0.3-amin-0.01-amax-0.03", "matmult", "--scheduler wave --schedule_options ' --save_layer_output ' "],

  "0-1conv_s8_32b_wave": [ "trivnet_lin",    "tfloat32-l2-b1-h16-r1-s8-c1-m1-wmin-0.1-wmax0.11-imin-0.2-imax0.21", "1conv32", "--scheduler wave --wavegraph_checks structure data-race"],
  "1-1conv7_64_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h16-r7-s1-c64-m64-wmin-0.1-wmax0.11-imin-0.2-imax0.21", "1conv", "--scheduler wave --wavegraph_checks structure data-race"],
  "1-1conv9_64_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h16-r9-s1-c64-m64-wmin-0.1-wmax0.11-imin-0.2-imax0.21", "1conv", "--scheduler wave"],

  # Wave graph development tcc reference and tests
  "1-1conv0_wave"      : [ "trivnet_conv1",  "tfloat16-b1-h1-r1-s1-c33-m1-wmin-0.01-wmax0.011-imin-0.02-imax0.022", "1conv", "--scheduler wave --wavegraph_checks structure data-race"],
  "1-1conv0_c128_wave"      : [ "trivnet_conv1",  "tfloat16-b1-h1-r1-s1-c128-m1-wmin-0.01-wmax0.011-imin-0.022-imax0.023", "1conv", "--scheduler wave"],
  "1-1conv0_c256_wave"      : [ "trivnet_conv1",  "tfloat16-b1-h1-r1-s1-c256-m1-wmin-0.01-wmax0.011-imin-0.022-imax0.023", "1conv", "--scheduler wave2 --schedule_options ' --nname=generic' --wavegraph_checks structure data-race"],
  "1-1conv0_m64_wave"      : [ "trivnet_conv1",  "tfloat16-b1-h1-r1-s1-c1-m64-wmin1-wmax1.1-imin2-imax2.2", "1conv", "--scheduler wave --wavegraph_checks structure data-race"],
  "1-1conv0_m128_wave"      : [ "trivnet_conv1",  "tfloat16-b1-h1-r1-s1-c1-m128-wmin-0.01-wmax0.011-imin-0.02-imax0.022", "1conv", "--scheduler wave2 --wavegraph_checks structure data-race"],
  "1-1conv0_m2_wave"      : [ "trivnet_conv1",  "tfloat16-b1-h1-r1-s1-c1-m2-wmin-0.01-wmax0.011-imin-0.02-imax0.022", "1conv", "--scheduler wave --wavegraph_checks structure data-race"],
  "1-1conv0_h16c128_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h16-r1-s1-c128-m1-wmin2-wmax2.2-imin3-imax3.2", "1conv", "--scheduler wave --wavegraph_checks structure data-race"],
  "1-1conv0_h16c256_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h16-r1-s1-c256-m1-wmin2-wmax2.2-imin3-imax3.2", "1conv", "--scheduler wave2 --schedule_options ' --nname=generic' --wavegraph_checks structure data-race"],
  "1-1conv0_h16c1m1_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h16-r1-s1-c2-m1-wmin-0.2-wmax0.3-imin-0.1-imax0.2", "1conv", "--scheduler wave --wavegraph_checks structure data-race"],
  "1-1conv0_h16c2m1_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h16-r1-s1-c2-m1-wmin-0.2-wmax0.3-imin-0.1-imax0.2", "1conv", "--scheduler wave --wavegraph_checks structure data-race"],

  "1-1conv0_h16c256m128_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h16-r1-s1-c256-m128-wmin-1-wmax2-imin-1-imax3", "1conv", "--scheduler wave --wavegraph_checks structure data-race"],
  "1-1conv0_h16c256m128_fp32_wave"   : [ "trivnet_conv1",  "tfloat32-b1-h16-r1-s1-c256-m128-wmin-1-wmax2-imin-1-imax3", "1conv", "--scheduler wave --wavegraph_checks structure data-race"],

  "1-1conv0_h40c1m1_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h40-r1-s1-c1-m1-wmin2-wmax2.2-imin3-imax3.2", "1conv", "--scheduler wave --wavegraph_checks structure data-race"],
  #"1-1conv0_h40c128m64_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h40-r1-s1-c128-m64-wmin2-wmax2.2-imin3-imax3.2", "1conv", "--scheduler wave --wavegraph_checks structure data-race"],
  #"1-1conv0_h40c256m128_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h40-r1-s1-c256-m128-wmin2-wmax2.2-imin3-imax3.2", "1conv", "--scheduler wave --wavegraph_checks structure data-race"],
  #"1-1conv0_h40c128m128_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h40-r1-s1-c128-m128-wmin2-wmax2.2-imin3-imax3.2", "1conv", "--scheduler wave --wavegraph_checks structure data-race"],
  #"1-1conv0_h40c256m64_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h40-r1-s1-c256-m64-wmin2-wmax2.2-imin3-imax3.2", "1conv", "--scheduler wave --wavegraph_checks structure data-race"],

  #"1-1conv0_h32c128m64_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h32-r1-s1-c128-m64-wmin2-wmax2.2-imin3-imax3.2", "1conv", "--scheduler wave --wavegraph_checks structure data-race"],
  #"1-1conv0_h32c256m128_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h32-r1-s1-c256-m128-wmin2-wmax2.2-imin3-imax3.2", "1conv", "--scheduler wave --wavegraph_checks structure data-race"],
  #"1-1conv0_h32c128m128_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h32-r1-s1-c128-m128-wmin2-wmax2.2-imin3-imax3.2", "1conv", "--scheduler wave --wavegraph_checks structure data-race"],
  #"1-1conv0_h32c256m64_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h32-r1-s1-c256-m64-wmin2-wmax2.2-imin3-imax3.2", "1conv", "--scheduler wave --wavegraph_checks structure data-race"],

  "1-1conv0_h64c2m1_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h64-r1-s1-c2-m1-wmin2-wmax2.2-imin3-imax3.2", "1conv", "--scheduler wave --wavegraph_checks structure data-race"],
  ##"1-1conv0_h256c2m1_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h256-r1-s1-c2-m1-wmin2-wmax2.2-imin3-imax3.2", "1conv", "--scheduler wave2 --wavegraph_checks structure data-race"],

  "1-1conv0_h32c1m1_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h32-r1-s1-c1-m1-wmin0-wmax1-imin0-imax1023", "1conv", "--scheduler wave --wavegraph_checks structure data-race"],
  "1-1conv0_h32c2m1_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h32-r1-s1-c2-m1-wmin0-wmax1-imin0-imax1023", "1conv", "--scheduler wave"],
  "1-1conv0_h32c4m1_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h32-r1-s1-c4-m1-wmin0-wmax1-imin0-imax1023", "1conv", "--scheduler wave --wavegraph_checks structure data-race"],
  "1-1conv0_h32c8m1_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h32-r1-s1-c8-m1-wmin0-wmax1-imin0-imax1023", "1conv", "--scheduler wave"],
  "1-1conv0_h32c64m1_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h32-r1-s1-c64-m1-wmin2-wmax2.2-imin3-imax3.2", "1conv", "--scheduler wave --wavegraph_checks structure data-race"],
  "1-1conv0_h32c128m1_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h32-r1-s1-c128-m1-wmin2-wmax2.2-imin3-imax3.2", "1conv", "--scheduler wave --wavegraph_checks structure data-race"],
  "1-1conv0_h32c1m2_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h32-r1-s1-c1-m2-wmin2-wmax2.2-imin3-imax3.2", "1conv", "--scheduler wave"],

  "1-1conv0_h28c1m1_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h28-r1-s1-c1-m1-wmin2-wmax2.2-imin3-imax3.2", "1conv", "--scheduler wave --wavegraph_checks structure data-race"],
  "1-1conv0_h56c1m1_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h56-r1-s1-c1-m1-wmin-1-wmax2-imin-2-imax3", "1conv", "--scheduler wave --wavegraph_checks structure data-race"],
  "1-1conv0_h112c1m1_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h112-r1-s1-c1-m1-wmin2-wmax2.2-imin3-imax3.2", "1conv", "--scheduler wave"],
  "1-1conv0_h224c1m1_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h224-r1-s2-c1-m1-wmin2-wmax2.2-imin3-imax3.2", "1conv", "--scheduler wave --wavegraph_checks structure data-race"],

  "1-1conv0_h55c256_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h55-r1-s1-c256-m1-wmin-0.1-wmax0.11-imin-0.2-imax0.22", "1conv", "--scheduler wave2 --schedule_options ' --nname=generic'"],
  #"1-1conv0_h55c64m256_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h55-r1-s1-c64-m256-wmin-0.1-wmax0.11-imin-0.2-imax0.22", "1conv", "--scheduler wave --wavegraph_checks structure data-race"],
  "0-1conv0_h55c256m1_wave" : [ "trivnet_conv1",  "tfloat16-b1-h55-r1-s1-c256-m1-wmin2-wmax2.2-imin3-imax3.2", "1conv", "--scheduler wave2 --schedule_options ' --nname=generic' --wavegraph_checks structure data-race"],
  "1-1conv0_h55m64_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h55-r1-s1-c1-m64-wmin-0.1-wmax0.11-imin-0.2-imax0.22", "1conv", "--scheduler wave --wavegraph_checks structure data-race"],
  "1-1conv0_r3h16c128_wave" : [ "trivnet_conv1",  "tfloat16-b1-h16-r3-s1-c128-m1-wmin2-wmax2.2-imin3-imax3.2", "1conv", "--scheduler wave --wavegraph_checks structure data-race"],
  "1-1conv0_r3h55c256_wave" : [ "trivnet_conv1",  "tfloat16-b1-h55-r3-s1-c256-m1-wmin-0.1-wmax0.11-imin-0.2-imax0.22", "1conv", "--scheduler wave --wavegraph_checks structure data-race"],

  "1-1conv_transpose_wave" : [ "trivnet_conv_transpose",  "tfloat16-b1-h4-r1-s1-c1-m1-wmin2-wmax2.2-imin3-imax3.2", "1conv", "--scheduler wave2 --wavegraph_checks structure data-race"],
  "1-1conv_transpose_s2_wave" : [ "trivnet_conv_transpose",  "tfloat16-b1-h4-r1-s2-c1-m1-wmin2-wmax2.2-imin3-imax3.2", "1conv", "--scheduler wave2 --wavegraph_checks structure data-race"],

  "2-1conv3_64s8_wave" : [ "trivnet_conv1",  "tfloat16-b1-h16-r3-s8-c64-m64-wmin-0.1-wmax0.11-imin-0.2-imax0.22", "1conv", "--scheduler wave"],
  "2-1conv9_64s8_wave" : [ "trivnet_conv1",  "tfloat16-b1-h16-r9-s8-c64-m64-wmin-0.1-wmax0.11-imin-0.2-imax0.22", "1conv", "--scheduler wave --wavegraph_checks structure data-race"],

  "2-padasym_strd_h112r7s2_wave" : [ "trivnet_conv1", "tfloat16-b1-h112-r7-s2-c1-m1-wmin-0.1-wmax0.11-imin-0.2-imax0.22", "1conv", "--scheduler wave --wavegraph_checks structure data-race"],
  "2-padasym_strd_h224r7s2_wave" : [ "trivnet_conv1", "tfloat16-b1-h224-r7-s2-c3-m64-wmin-0.1-wmax0.2-imin-0.2-imax0.3", "1conv", "--scheduler wave"],
  "2-padasym_strd_h224r7s2_fp32_wave" : [ "trivnet_conv1", "tfloat32-b1-h224-r7-s2-c3-m64-wmin-0.1-wmax0.2-imin-0.2-imax0.3", "1conv", "--scheduler wave"],

  # Full c, m in resnet50 are 512, 2048
  "3-rn50_pool2_wave"  : [ "trivnet_conv_pool", "tfloat16-b1-h7-r1-s1-c128-m64-SAME-AvgPool-k7-d7-PERM-wmin-0.1-wmax0.1-imin-1-imax2", "1conv1pool", "--scheduler wave"],
  "3-1conv1maxpool_k3d2_wave"  : [ "trivnet_conv_pool_conv", "tfloat16-b1-h224-r3-s2-c128-m64-VALID-MaxPool-k3-d2-wmin-0.2-wmax0.3-imin-0.2-imax0.3", "1conv1pool", "--scheduler wave"],
  "3-1conv0_padvalid_wave" : [ "trivnet_conv1_padvalid",  "tfloat16-b1-h230-r7-s2-c3-m64-wmin-2-wmax2.2-imin-3-imax3.2", "1conv", "--scheduler wave2 --debug 1 --wavegraph_checks structure data-race"],

  # Sprint9 Story 1 milestone - all resnet50 float32 Conv2D layers as unit test
  # The 00 is just for testing the regression harness
  "3-rn50-t00_fp32_wave" : [ "trivnet_conv1",  "tfloat32-b1-h1-r1-s1-c1-m1-wmin-1-wmax1.1-imin-3-imax3.2",       "1conv", "--scheduler wave --wavegraph_checks structure data-race"],
  "3-rn50-01_fp32_wave" : [ "trivnet_conv1",  "tfloat32-b1-h14-r3-s1-c256-m256-wmin-1-wmax1.1-imin-3-imax3.2",  "1conv", "--scheduler wave --wavegraph_checks structure data-race"],
  "3-rn50-02_fp32_wave" : [ "trivnet_conv1",  "tfloat32-b1-h14-r1-s1-c256-m1024-wmin-1-wmax1.1-imin-3-imax3.2", "1conv", "--scheduler wave --wavegraph_checks structure data-race"],
  "3-rn50-03_fp32_wave" : [ "trivnet_conv1",  "tfloat32-b1-h14-r1-s1-c1024-m256-wmin-1-wmax1.1-imin-3-imax3.2", "1conv", "--scheduler wave"],
  "3-rn50-04_fp32_wave" : [ "trivnet_conv1",  "tfloat32-b1-h55-r1-s1-c64-m256-wmin-1-wmax1.1-imin-3-imax3.2",   "1conv", "--scheduler wave --wavegraph_checks structure data-race"],
  "3-rn50-05_fp32_wave" : [ "trivnet_conv1",  "tfloat32-b1-h28-r3-s1-c128-m128-wmin-1-wmax1.1-imin-3-imax3.2",  "1conv", "--scheduler wave --wavegraph_checks structure data-race"],
  "3-rn50-06_fp32_wave" : [ "trivnet_conv1",  "tfloat32-b1-h28-r1-s1-c128-m512-wmin-1-wmax1.1-imin-3-imax3.2",  "1conv", "--scheduler wave --wavegraph_checks structure data-race"],
  "3-rn50-07_fp32_wave" : [ "trivnet_conv1",  "tfloat32-b1-h55-r3-s1-c64-m64-wmin-1-wmax1.1-imin-3-imax3.2",    "1conv", "--scheduler wave --wavegraph_checks structure data-race"],
  "3-rn50-08_fp32_wave" : [ "trivnet_conv1",  "tfloat32-b1-h7-r3-s1-c512-m512-wmin-1-wmax1.1-imin-3-imax3.2",   "1conv", "--scheduler wave --wavegraph_checks structure data-race"], 
  "3-rn50-09_fp32_wave" : [ "trivnet_conv1",  "tfloat32-b1-h7-r1-s1-c512-m2048-wmin-1-wmax1.1-imin-3-imax3.2",  "1conv", "--scheduler wave --wavegraph_checks structure data-race"],
  "3-rn50-10_fp32_wave" : [ "trivnet_conv1",  "tfloat32-b1-h28-r1-s1-c512-m128-wmin-1-wmax1.1-imin-3-imax3.2",  "1conv", "--scheduler wave --wavegraph_checks structure data-race"],
  "3-rn50-11_fp32_wave" : [ "trivnet_conv1",  "tfloat32-b1-h55-r1-s1-c256-m64-wmin-1-wmax1.1-imin-3-imax3.2",   "1conv", "--scheduler wave --wavegraph_checks structure data-race"], 
  "3-rn50-12_fp32_wave" : [ "trivnet_conv1",  "tfloat32-b1-h7-r1-s1-c2048-m512-wmin-1-wmax1.1-imin-3-imax3.2",  "1conv", "--scheduler wave --wavegraph_checks structure data-race"], 
  "3-rn50-13_fp32_wave" : [ "trivnet_conv1",  "tfloat32-b1-h55-r1-s1-c64-m64-wmin-1-wmax1.1-imin-3-imax3.2",    "1conv", "--scheduler wave --wavegraph_checks structure data-race"], 
  "3-rn50-14_fp32_wave" : [ "trivnet_conv1",  "tfloat32-b1-h28-r1-s2-c512-m256-wmin-1-wmax1.1-imin-3-imax3.2",  "1conv", "--scheduler wave --wavegraph_checks structure data-race"],
  "3-rn50-15_fp32_wave" : [ "trivnet_conv1",  "tfloat32-b1-h28-r1-s2-c512-m1024-wmin-1-wmax1.1-imin-3-imax3.2", "1conv", "--scheduler wave --wavegraph_checks structure data-race"],

  "3-rn50-16_fp32_wave" : [
    "trivnet_conv1",
    "tfloat32-b1-h224-r7-s2-c3-m64-wmin-1-wmax1.1-imin-3-imax3.2",
    "1conv",
    "--scheduler wave"
  ],

  "3-rn50-16_fp32_wave-fast_dram" : [
    "trivnet_conv1",
    "tfloat32-b1-h224-r7-s2-c3-m64-wmin-1-wmax1.1-imin-3-imax3.2",
    "1conv", "--scheduler wave",
    "--env SIM_ADD_FLAGS=' --dram_frequency 0 --dram_latency 1'"
  ],

  "3-rn50-17_fp32_wave" : [ "trivnet_conv1",  "tfloat32-b1-h55-r1-s2-c256-m512-wmin-1-wmax1.1-imin-3-imax3.2",  "1conv", "--scheduler wave --wavegraph_checks structure data-race"],
  "3-rn50-18_fp32_wave" : [ "trivnet_conv1",  "tfloat32-b1-h55-r1-s2-c256-m128-wmin-1-wmax1.1-imin-3-imax3.2",  "1conv", "--scheduler wave --wavegraph_checks structure data-race"],
  "3-rn50-19_fp32_wave" : [ "trivnet_conv1",  "tfloat32-b1-h14-r1-s2-c1024-m512-wmin-1-wmax1.1-imin-3-imax3.2",  "1conv", "--scheduler wave --wavegraph_checks structure data-race"],
  "3-rn50-20_fp32_wave" : [ "trivnet_conv1",  "tfloat32-b1-h14-r1-s2-c1024-m2048-wmin-1-wmax1.1-imin-3-imax3.2", "1conv", "--scheduler wave --wavegraph_checks structure data-race"],

  "3-rn50-t00_wave" : [ "trivnet_conv1",  "tfloat16-b1-h1-r1-s1-c1-m1-wmin-1-wmax1.1-imin-3-imax3.2",       "1conv", "--scheduler wave --wavegraph_checks structure data-race"],
  "3-rn50-01_wave" : [ "trivnet_conv1",  "tfloat16-b1-h14-r3-s1-c256-m256-wmin-1-wmax1.1-imin-3-imax3.2",  "1conv", "--scheduler wave --wavegraph_checks structure data-race"],
  "3-rn50-02_wave" : [ "trivnet_conv1",  "tfloat16-b1-h14-r1-s1-c256-m1024-wmin-1-wmax1.1-imin-3-imax3.2", "1conv", "--scheduler wave --wavegraph_checks structure data-race"],
  "3-rn50-03_wave" : [ "trivnet_conv1",  "tfloat16-b1-h14-r1-s1-c1024-m256-wmin-1-wmax1.1-imin-3-imax3.2", "1conv", "--scheduler wave --wavegraph_checks structure data-race"],
  "3-rn50-04_wave" : [ "trivnet_conv1",  "tfloat16-b1-h55-r1-s1-c64-m256-wmin-1-wmax1.1-imin-3-imax3.2",   "1conv", "--scheduler wave --wavegraph_checks structure data-race"],
  "3-rn50-05_wave" : [ "trivnet_conv1",  "tfloat16-b1-h28-r3-s1-c128-m128-wmin-1-wmax1.1-imin-3-imax3.2",  "1conv", "--scheduler wave --wavegraph_checks structure data-race"],
  "3-rn50-06_wave" : [ "trivnet_conv1",  "tfloat16-b1-h28-r1-s1-c128-m512-wmin-1-wmax1.1-imin-3-imax3.2",  "1conv", "--scheduler wave --wavegraph_checks structure data-race"],
  "3-rn50-07_wave" : [ "trivnet_conv1",  "tfloat16-b1-h55-r3-s1-c64-m64-wmin-1-wmax1.1-imin-3-imax3.2",    "1conv", "--scheduler wave"],
  "3-rn50-08_wave" : [ "trivnet_conv1",  "tfloat16-b1-h7-r3-s1-c512-m512-wmin-1-wmax1.1-imin-3-imax3.2",   "1conv", "--scheduler wave2 --schedule_options ' --nname=generic' --wavegraph_checks structure data-race"], 
  #"3-rn50-08_b2_wave" : [ "trivnet_conv1",  "tfloat16-b2-h7-r3-s1-c512-m512-wmin-1-wmax1.1-imin-3-imax3.2",   "1conv", "--scheduler wave --wavegraph_checks structure data-race"], 
  "3-rn50-09_wave" : [ "trivnet_conv1",  "tfloat16-b1-h7-r1-s1-c512-m2048-wmin-1-wmax1.1-imin-3-imax3.2",  "1conv", "--scheduler wave2 --schedule_options ' --nname=generic' --wavegraph_checks structure data-race"],
  "3-rn50-10_wave" : [ "trivnet_conv1",  "tfloat16-b1-h28-r1-s1-c512-m128-wmin-1-wmax1.1-imin-3-imax3.2",  "1conv", "--scheduler wave --wavegraph_checks structure data-race"],
  "3-rn50-11_wave" : [ "trivnet_conv1",  "tfloat16-b1-h55-r1-s1-c256-m64-wmin-1-wmax1.1-imin-3-imax3.2",   "1conv", "--scheduler wave --wavegraph_checks structure data-race"], 
  "3-rn50-12_wave" : [ "trivnet_conv1",  "tfloat16-b1-h7-r1-s1-c2048-m512-wmin-1-wmax1.1-imin-3-imax3.2",  "1conv", "--scheduler wave2 --schedule_options ' --nname=generic' --wavegraph_checks structure data-race"], 
  #"3-rn50-12_b2_wave" : [ "trivnet_conv1",  "tfloat16-b2-h7-r1-s1-c2048-m512-wmin-1-wmax1.1-imin-3-imax3.2",  "1conv", "--scheduler wave2 --wavegraph_checks structure data-race"], 
  "3-rn50-13_wave" : [ "trivnet_conv1",  "tfloat16-b1-h55-r1-s1-c64-m64-wmin-1-wmax1.1-imin-3-imax3.2",    "1conv", "--scheduler wave --wavegraph_checks structure data-race"], 
  "3-rn50-14_wave" : [ "trivnet_conv1",  "tfloat16-b1-h28-r1-s2-c512-m256-wmin-1-wmax1.1-imin-3-imax3.2",  "1conv", "--scheduler wave --wavegraph_checks structure data-race"],
  "3-rn50-15_wave" : [ "trivnet_conv1",  "tfloat16-b1-h28-r1-s2-c512-m1024-wmin-1-wmax1.1-imin-3-imax3.2", "1conv", "--scheduler wave --wavegraph_checks structure data-race"],
  "3-rn50-16_wave" : [ "trivnet_conv1",  "tfloat16-b1-h224-r7-s2-c3-m64-wmin-1-wmax1.1-imin-3-imax3.2",    "1conv", "--scheduler wave2 --wavegraph_checks structure data-race --schedule_options ' ' "],
  "3-rn50-17_wave" : [ "trivnet_conv1",  "tfloat16-b1-h55-r1-s2-c256-m512-wmin-1-wmax1.1-imin-3-imax3.2",  "1conv", "--scheduler wave2 --wavegraph_checks structure data-race"],
  "3-rn50-18_wave" : [ "trivnet_conv1",  "tfloat16-b1-h55-r1-s2-c256-m128-wmin0-wmax1-imin0-imax3",  "1conv", "--scheduler wave2 --wavegraph_checks structure data-race"],
  "3-rn50-19_wave" : [ "trivnet_conv1",  "tfloat16-b1-h14-r1-s2-c1024-m512-wmin-1-wmax1.1-imin-3-imax3.2",  "1conv", "--scheduler wave2 --schedule_options ' --nname=generic' --wavegraph_checks structure data-race"],
  "3-rn50-20_wave" : [ "trivnet_conv1",  "tfloat16-b1-h14-r1-s2-c1024-m2048-wmin-1-wmax1.1-imin-3-imax3.2", "1conv", "--scheduler wave2 --schedule_options ' --nname=generic' --wavegraph_checks structure data-race"],

  "0-rn50-16_wave_repl" : [ "trivnet_conv1",  "tfloat16-b1-h14-r7-s2-c3-m2-wmin-1-wmax1.1-imin-3-imax3.2",    "1conv", "--scheduler wave2 --schedule_options ' --enable_replication ' "],
  "1-rn50-16_wave_repl" : [ "trivnet_conv1",  "tfloat16-b1-h56-r7-s2-c3-m2-wmin-1-wmax1.1-imin-3-imax3.2",    "1conv", "--scheduler wave2 --schedule_options ' --enable_replication ' "],
  "2-rn50-16_wave_repl" : [ "trivnet_conv1",  "tfloat16-b1-h224-r7-s2-c3-m2-wmin-1-wmax1.1-imin-3-imax3.2",    "1conv", "--scheduler wave2 --schedule_options ' --enable_replication ' "],

  "3-rn50-16_wave_repl" : [
    "trivnet_conv1",
    "tfloat16-b1-h224-r7-s2-c3-m64-wmin-1-wmax1.1-imin-3-imax3.2",
    "1conv",
    "--scheduler wave2 --schedule_options ' --enable_replication ' "
  ],

  ## db
  "3-rn50-16_wave_repl-fast_dram" : [
    "trivnet_conv1",
    "tfloat16-b1-h224-r7-s2-c3-m64-wmin-1-wmax1.1-imin-3-imax3.2",
    "1conv",
    "--scheduler wave2 --schedule_options ' --enable_replication ' ",
    "--env SIM_ADD_FLAGS=' --dram_frequency 0 --dram_latency 1 '"
  ],


  "3-rn50-16_b2_wave_repl" : [ "trivnet_conv1",  "tfloat16-b2-h224-r7-s2-c3-m64-wmin1-wmax1-imin0-imax223",    "1conv", "--scheduler wave2 --schedule_options ' --enable_replication ' "],

  #"5-lstm_ptb"     : [ "tf_pb",          "lstm_ptb_word_lm/ptb_word_lm.pb",  "lstm_ptb", "--input_node Valid/ValidInput/StridedSlice ", "linspace1"],
  "6-alexnet"     : [ "tf_pb",          "alexnet_v100/alexnet_fp32.pb",  "alexnet", "--input_node Variable ", "linspace1"],
  "8-resnet50"                : [ "tf_pb",   "resnet50/resnet_v1_50_fp32_opt.pb",        "resnet50", " --depth 2", "linspace1"],
  "8-resnet50_fp32_keras"     : [ "tf_pb",   "resnet50_keras/resnet50_fp32_keras.pb",    "resnet50", "--input_node input_1  --depth 2", "linspace1"],
  "8-resnet50_fp32_keras_opt" : [ "tf_pb",   "resnet50_keras/resnet50_fp32_keras_opt.pb","resnet50", "--input_node input_1  --depth 2", "linspace1"],
  "8-resnet50_fp16_keras"     : [ "tf_pb",   "resnet50_keras/resnet50_fp16_keras.pb",    "resnet50", "--input_node input_1  --depth 2", "linspace1"],
  "8-resnet50_fp16_keras_opt_b16" : [ "tf_pb",   "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2 --batch 16", "linspace1"],
  "8-resnet50_fp16_keras_opt" : [ "tf_pb",   "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2", "linspace1"],
  "8-resnet50_fp16_wave"      : [ "tf_pb",   "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2 --scheduler wave --wavegraph_checks structure data-race", "linspace1"],
  "8-resnet50_fp32_wave"      : [ "tf_pb",   "resnet50_keras/resnet50_fp32_keras_opt.pb","resnet50", "--input_node input_1  --depth 2 --scheduler wave --wavegraph_checks structure data-race", "linspace1"],
  "8-resnet50_fp16_wave_b2"   : [ "tf_pb",   "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2 --scheduler wave --batch 2 --wavegraph_checks structure data-race", "linspace1"],
  "9-resnet152"               : [ "tf_pb",   "resnet_v2_152/pb/resnet_v2_152_fp32.pb",   "resnet152", " --depth 2", "linspace1"],
  "9-resnet152_waveopt"       : [ "tf_pb",   "resnet_v2_152/pb/resnet_v2_152_fp32.pb",   "resnet152", "--partition from resnet_v2_152/conv1/convolution resnet_v2_152/postnorm/batchnorm/mul_1 --executors host all waveopt 1  --depth 2 --scheduler wave --images %s --wavegraph_checks structure data-race" % rnDogJpg, "--input_files %s" % rnDogJpg],

  #"10-parwavenet_ckpt"            : [ "tf_pb",   "parallel_wavenet/saved_model",        "parallel_wavenet", "--input_node Placeholder --depth 2", "--input_files %s" % melSpectra],
  
  # Subgraph partioned flow using neural network executor
  "0-4conv_relu_nne" : [ "trivnet_lin",    "tfloat16-l3-b1-h4-r3-s1-c1-m1-relu-wmin-0.2-wmax0.4-imin-1-imax2", "4conv_nne", "--partition conv --executors wave 1 3 host 0 2 4 --debug 1 --scheduler wave --wavegraph_checks structure data-race"],


  # Resnet
  "8-rn50_nne_auto"             : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition auto --executors wave all  --scheduler wave --images %s --wavegraph_checks structure data-race" %(rnPreFp16, rnDogJpg), "--input_files %s" % rnDogJpg ],
  #"8-rn50_nne_fp32_meauto"      : [ "tf_pb", "resnet50_keras/resnet50_fp32_keras_opt.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition meauto --executors wave all host 17  --scheduler wave --images %s" %(rnPreFp32, rnDogJpg), "--input_files %s" % rnDogJpg ],
  "8-rn50_nne_fp16_meauto"      : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition meauto --executors wave all host 17  --scheduler wave --images %s" %(rnPreFp16, rnDogJpg), "--input_files %s" % rnDogJpg ],
  #"8-rn50_nne_conv"            : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition conv --executors tcc 2 6 8 13 15 20 22 host 0 --images %s" %(rnPreFp16, rnDogJpg), "linspace1"],
  "4-rn50_nne_fc"               : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from avg_pool/AvgPool --executors host 0 host 1 --images %s" %(rnPreFp16, rnDogJpg), "--input_files %s" % rnDogJpg ],
  "4-rn50_matmul_fp32_wave"     : [ "tf_pb", "resnet50_keras/resnet50_fp32_keras_opt.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from avg_pool/AvgPool --executors host 0 wave 1 --images %s" %(rnPreFp32, rnDogJpg), "--input_files %s" % rnDogJpg ],
  #"4-rn50_nne_from1_fp32_wave"  : [ "tf_pb", "resnet50_keras/resnet50_fp32_keras_opt.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from bn_conv1/batchnorm_1/add_1   --executors wave 0 host 1  --scheduler wave --images %s --wavegraph_checks structure data-race" %(rnPreFp32, rnDogJpg), "--input_files %s" % rnDogJpg ],
  #"4-rn50_nne_from2_fp32_wave"  : [ "tf_pb", "resnet50_keras/resnet50_fp32_keras_opt.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from activation_1/Relu   --executors wave 0 host 1  --scheduler wave --images %s --wavegraph_checks structure data-race" %(rnPreFp32, rnDogJpg), "--input_files %s" % rnDogJpg ],
  "4-rn50_nne_from3_fp32_wave"  : [ "tf_pb", "resnet50_keras/resnet50_fp32_keras_opt.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from max_pooling2d_1/MaxPool   --executors wave 0 host 1  --scheduler wave --images %s" %(rnPreFp32, rnDogJpg), "--input_files %s" % rnDogJpg ],
  #"4-rn50_nne_from1_wave"       : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from bn_conv1/batchnorm_1/add_1   --executors wave 0 host 1  --scheduler wave --images %s" %(rnPreFp16, rnDogJpg), "--input_files %s" % rnDogJpg ],
  #"4-rn50_nne_from2_wave"       : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from activation_1/Relu   --executors wave 0 host 1  --scheduler wave --images %s" %(rnPreFp16, rnDogJpg), "--input_files %s" % rnDogJpg ],
  "4-rn50_nne_from3_wave"       : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from max_pooling2d_1/MaxPool   --executors wave 0 host 1  --scheduler wave2 --images %s" %(rnPreFp16, rnDogJpg), "--input_files %s" % rnDogJpg ],
  #"5-rn50_nne_to_act4_wave"     : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from activation_4/Relu --executors wave 0 host 1  --scheduler wave2 --images %s" %(rnPreFp16, rnDogJpg), "--input_files %s" % rnDogJpg ],

  "5-rn50_nne_to_act4_wave-no_repl"     : [
    "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50",
    "--input_node input_1  --depth 2  --debug 1 %s --partition from activation_4/Relu --executors wave 0 host 1  --scheduler wave2 --images %s --wavegraph_checks structure data-race " %(rnPreFp16, rnDogJpg),
    "--input_files %s" % rnDogJpg
  ],

  "5-rn50_nne_to_act4_wave-repl"     : [
    "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50",
    "--input_node input_1  --depth 2  --debug 1 %s --partition from activation_4/Relu --executors wave 0 host 1  --scheduler wave2 --images %s --schedule_options ' --enable_replication ' --wavegraph_checks structure data-race " %(rnPreFp16, rnDogJpg),
    "--input_files %s" % rnDogJpg
  ],

  #"5-rn50_nne_to_act4_b2_wave"     : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from activation_4/Relu --executors wave 0 host 1  --scheduler wave2 --batch 2 --images %s"%(rnPreFp16, getBatchedJpgs(2)), "--input_files %s" % (getBatchedJpgs(2))],
  "5-rn50_nne_to_act4_b4_wave"     : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from activation_4/Relu --executors wave 0 host 1  --scheduler wave2 --batch 4 --images %s"%(rnPreFp16, getBatchedJpgs(4)), "--input_files %s" % (getBatchedJpgs(4))],
  #"5-rn50_nne_to_act4_b16_wave"     : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from activation_4/Relu --executors wave 0 host 1  --scheduler wave2 --batch 16 --images %s"%(rnPreFp16, getBatchedJpgs(16)), "--input_files %s" % (getBatchedJpgs(16))],
  #"5-rn50_nne_to_act4_wave_b16"     : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from activation_4/Relu --executors wave 0 host 1  --scheduler wave --batch 16 --images %s"%(rnPreFp16, getBatchedJpgs(16)), "--input_files %s" % (getBatchedJpgs(16))],

  #"5-rn50_nne_to_act13_wave"     : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from activation_13/Relu --executors wave 0 host 1  --scheduler wave2 --batch 1 --images %s"%(rnPreFp16, getBatchedJpgs(1)), "--input_files %s" % (getBatchedJpgs(1))],

  "5-rn50_nne_to_act13_wave-repl"     : [
    "tf_pb",
    "resnet50_keras/resnet50_fp16_keras_opt2.pb",
    "resnet50",
    "--input_node input_1  --depth 2  --debug 1 %s --partition from activation_13/Relu --executors wave 0 host 1  --scheduler wave2 --batch 1 --images %s --schedule_options ' --enable_replication ' "%(rnPreFp16, getBatchedJpgs(1)),
    "--input_files %s" % (getBatchedJpgs(1))
  ],

  #"5-rn50_nne_to_act13_b8_wave"     : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from activation_13/Relu --executors wave 0 host 1  --scheduler wave2 --batch 8 --images %s"%(rnPreFp16, getBatchedJpgs(8)), "--input_files %s" % (getBatchedJpgs(8))],
  #"5-rn50_nne_to_act13_b8_wave-fast_dram"     : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from activation_13/Relu --executors wave 0 host 1  --scheduler wave2 --batch 8 --images %s"%(rnPreFp16, getBatchedJpgs(8)), "--env SIM_ADD_FLAGS=' --dram_frequency 0 --dram_latency 1' --input_files %s" % (getBatchedJpgs(8))],
  #"5-rn50_nne_to_act13_b16_wave"     : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from activation_13/Relu --executors wave 0 host 1  --scheduler wave2 --batch 16 --images %s"%(rnPreFp16, getBatchedJpgs(16)), "--input_files %s" % (getBatchedJpgs(16))],
  #"5-rn50_nne_to_act13_b16_wave-fast_dram"     : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from activation_13/Relu --executors wave 0 host 1  --scheduler wave2 --batch 16 --images %s"%(rnPreFp16, getBatchedJpgs(16)), "--env SIM_ADD_FLAGS=' --dram_frequency 0 --dram_latency 1' --input_files %s" % (getBatchedJpgs(16))],
  #"6-rn50_nne_to_act22_wave"     : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from activation_22/Relu --executors wave 0 host 1  --scheduler wave2 --images %s"%(rnPreFp16, rnDogJpg), "--input_files %s" % rnDogJpg ],

  "6-rn50_nne_to_act22_wave-repl"     : [
    "tf_pb",
    "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50",
    "--input_node input_1  --depth 2  --debug 1 %s --partition from activation_22/Relu --executors wave 0 host 1  --scheduler wave2 --images %s --schedule_options ' --enable_replication ' "%(rnPreFp16, rnDogJpg),
    "--input_files %s" % rnDogJpg ],

  #"6-rn50_nne_to_act22_b4_wave"     : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from activation_22/Relu --executors wave 0 host 1  --scheduler wave2 --batch 4 --images %s"%(rnPreFp16, getBatchedJpgs(4)), "--input_files %s" % (getBatchedJpgs(4))],
  #"6-rn50_nne_to_act22_b16_wave"     : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from activation_22/Relu --executors wave 0 host 1  --scheduler wave2 --batch 16 --images %s"%(rnPreFp16, getBatchedJpgs(16)), "--input_files %s" % (getBatchedJpgs(16))],
  #"6-rn50_nne_to_act25_wave"     : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from activation_25/Relu --executors wave 0 host 1  --scheduler wave2 --images %s"%(rnPreFp16, getBatchedJpgs(1)), "--input_files %s" % (getBatchedJpgs(1))],

  "6-rn50_nne_to_act25_wave-repl"     : [
    "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50",
    "--input_node input_1  --depth 2  --debug 1 %s --partition from activation_25/Relu --executors wave 0 host 1  --scheduler wave2 --images %s --schedule_options ' --enable_replication ' "%(rnPreFp16, getBatchedJpgs(1)),
    "--input_files %s" % (getBatchedJpgs(1))],

  #"6-rn50_nne_to_act25_b4_wave"     : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from activation_25/Relu --executors wave 0 host 1  --scheduler wave2 --batch 4 --images %s"%(rnPreFp16, getBatchedJpgs(4)), "--input_files %s" % (getBatchedJpgs(4))],
  #"6-rn50_nne_to_act28_wave"     : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from activation_28/Relu --executors wave 0 host 1  --scheduler wave2 --images %s"%(rnPreFp16, getBatchedJpgs(1)), "--input_files %s" % (getBatchedJpgs(1))],

  "6-rn50_nne_to_act28_wave-repl"     : [
    "tf_pb",
    "resnet50_keras/resnet50_fp16_keras_opt2.pb",
    "resnet50",
    "--input_node input_1  --depth 2  --debug 1 %s --partition from activation_28/Relu --executors wave 0 host 1  --scheduler wave2 --images %s --schedule_options ' --enable_replication ' "%(rnPreFp16, getBatchedJpgs(1)),
    "--input_files %s" % (getBatchedJpgs(1))
  ],

  #"6-rn50_nne_to_act28_b16_wave"     : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from activation_28/Relu --executors wave 0 host 1  --scheduler wave2 --batch 16 --images %s"%(rnPreFp16, getBatchedJpgs(16)), "--input_files %s" % (getBatchedJpgs(16))],
  #"6-rn50_nne_to_act37_wave"     : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from activation_37/Relu --executors wave 0 host 1  --scheduler wave2 --images %s" %(rnPreFp16, rnDogJpg), "--input_files %s" % rnDogJpg ],

  "6-rn50_nne_to_act37_wave-repl"     : [
    "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50",
    "--input_node input_1  --depth 2  --debug 1 %s --partition from activation_37/Relu --executors wave 0 host 1  --scheduler wave2 --images %s --schedule_options ' --enable_replication ' " %(rnPreFp16, rnDogJpg),
    "--input_files %s" % rnDogJpg 
  ],

  #"6-rn50_nne_to_act37_b2_wave"     : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from activation_37/Relu --executors wave 0 host 1  --scheduler wave2 --batch 2 --images %s"%(rnPreFp16, getBatchedJpgs(2)), "--input_files %s" % (getBatchedJpgs(2))],
  #"6-rn50_nne_to_act37_b4_wave"     : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from activation_37/Relu --executors wave 0 host 1  --scheduler wave2 --batch 4 --images %s"%(rnPreFp16, getBatchedJpgs(4)), "--input_files %s" % (getBatchedJpgs(4))],
  #"6-rn50_nne_to_act37_b16_wave"     : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from activation_37/Relu --executors wave 0 host 1  --scheduler wave2 --batch 16 --images %s"%(rnPreFp16, getBatchedJpgs(16)), "--input_files %s" % (getBatchedJpgs(16))],
  "6-rn50_nne_to_act40_wave"     : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from activation_40/Relu --executors wave 0 host 1  --scheduler wave2 --images %s"%(rnPreFp16, getBatchedJpgs(1)), "--input_files %s" % (getBatchedJpgs(1))],
  #"6-rn50_nne_to_act40_b4_wave"     : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from activation_40/Relu --executors wave 0 host 1  --scheduler wave2 --batch 4 --images %s"%(rnPreFp16, getBatchedJpgs(4)), "--input_files %s" % (getBatchedJpgs(4))],
  #"6-rn50_nne_to_act40_b16_wave"     : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from activation_40/Relu --executors wave 0 host 1  --scheduler wave2 --batch 16 --images %s"%(rnPreFp16, getBatchedJpgs(16)), "--input_files %s" % (getBatchedJpgs(16))],
  #"6-rn50_nne_to_act40_b16_wave-fast_dram"     : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from activation_40/Relu --executors wave 0 host 1  --scheduler wave2 --batch 16 --images %s"%(rnPreFp16, getBatchedJpgs(16)), "--env SIM_ADD_FLAGS=' --dram_frequency 0 --dram_latency 1' --input_files %s" % (getBatchedJpgs(16))],
  "6-rn50_nne_to_act43_wave"     : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from activation_43/Relu --executors wave 0 host 1  --scheduler wave2 --images %s"%(rnPreFp16, getBatchedJpgs(1)), "--input_files %s" % (getBatchedJpgs(1))],
  #"6-rn50_nne_to_act43_b2_wave"     : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from activation_43/Relu --executors wave 0 host 1  --scheduler wave2 --batch 2 --images %s"%(rnPreFp16, getBatchedJpgs(2)), "--input_files %s" % (getBatchedJpgs(2))],
  #"6-rn50_nne_to_act43_b4_wave"     : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from activation_43/Relu --executors wave 0 host 1  --scheduler wave2 --batch 4 --images %s"%(rnPreFp16, getBatchedJpgs(4)), "--input_files %s" % (getBatchedJpgs(4))],
  #"6-rn50_nne_to_act43_b16_wave"     : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from activation_43/Relu --executors wave 0 host 1  --scheduler wave2 --batch 16 --images %s"%(rnPreFp16, getBatchedJpgs(16)), "--input_files %s" % (getBatchedJpgs(16))],
  #"6-rn50_nne_to_act46_wave"     : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from activation_46/Relu --executors wave 0 host 1  --scheduler wave2 --images %s"%(rnPreFp16, getBatchedJpgs(1)), "--input_files %s" % (getBatchedJpgs(1))],

  "6-rn50_nne_to_act46_wave-repl"     : [
    "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50",
    "--input_node input_1  --depth 2  --debug 1 %s --partition from activation_46/Relu --executors wave 0 host 1  --scheduler wave2 --images %s --schedule_options ' --enable_replication ' "%(rnPreFp16, getBatchedJpgs(1)),
    "--input_files %s" % (getBatchedJpgs(1))],

  #"6-rn50_nne_to_act46_b4_wave"     : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from activation_46/Relu --executors wave 0 host 1  --scheduler wave2 --batch 4 --images %s"%(rnPreFp16, getBatchedJpgs(4)), "--input_files %s" % (getBatchedJpgs(4))],
  #"6-rn50_nne_to_act46_b16_wave"     : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from activation_46/Relu --executors wave 0 host 1  --scheduler wave2 --batch 16 --images %s"%(rnPreFp16, getBatchedJpgs(16)), "--input_files %s" % (getBatchedJpgs(16))],
  #"7-rn50_nne_fc_waveopt"       : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from fc1000/Softmax --executors waveopt 0 host 1  --scheduler wave --images %s" %(rnPreFp16, rnDogJpg), "--input_files %s" % rnDogJpg ],
  #"7-rn50_nne_fp32_waveopt"     : [ "tf_pb", "resnet50_keras/resnet50_fp32_keras_opt.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from fc1000/Softmax --executors waveopt 0 host 1  --scheduler wave --images %s" %(rnPreFp32, rnDogJpg), "--input_files %s" % rnDogJpg ],
  #"7-rn50_nne_fp32_wave"        : [ "tf_pb", "resnet50_keras/resnet50_fp32_keras_opt.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from fc1000/Softmax --executors wave 0 host 1  --scheduler wave --images %s" %(rnPreFp32, rnDogJpg), "--input_files %s" % rnDogJpg ],

  "7-rn50_nne_fp16_wave-no_repl"        : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from fc1000/Softmax --executors wave 0 host 1  --scheduler wave2 --images %s" %(rnPreFp16, rnDogJpg), "--input_files %s" % rnDogJpg ],
  #"7-rn50_nne_fp16_ap_wave-no_repl"        : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from_multi flatten_1/Shape,flatten_1/Reshape --executors wave 0 host 1  --scheduler wave2 --images %s" %(rnPreFp16, rnDogJpg), "--input_files %s" % rnDogJpg ],
  "7-rn50_nne_fp16_wave"        : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from fc1000/Softmax --executors wave 0 host 1  --scheduler wave2 --schedule_options ' --enable_replication ' --images %s" %(rnPreFp16, rnDogJpg), "--input_files %s" % rnDogJpg ],
  "7-rn50_nne_fp16_ap_wave"        : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from_multi flatten_1/Shape,flatten_1/Reshape --executors wave 0 host 1  --scheduler wave2 --schedule_options ' --enable_replication ' --images %s" %(rnPreFp16, rnDogJpg), "--input_files %s" % rnDogJpg ],

  "7-rn50_nne_fp16_wave-two_banks" : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from fc1000/Softmax --executors wave 0 host 1  --scheduler wave2 --schedule_options ' --enable_replication ' --images %s" %(rnPreFp16, rnDogJpg), "--env SIM_ADD_FLAGS=' --dram_frequency 6400'  --input_files %s" % rnDogJpg ],
  "7-rn50_nne_fp16_wave-fast_dram" : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from fc1000/Softmax --executors wave 0 host 1  --scheduler wave2 --schedule_options ' --enable_replication ' --images %s" %(rnPreFp16, rnDogJpg), "--env SIM_ADD_FLAGS=' --dram_frequency 0 --dram_latency 1' --input_files %s" % rnDogJpg ],
  #"7-rn50_nne_fp16_b2_wave"        : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from fc1000/Softmax --executors wave 0 host 1  --scheduler wave2 --batch 2 --images %s"%(rnPreFp16, getBatchedJpgs(2)), "--input_files %s" % (getBatchedJpgs(2))],
  "7-rn50_nne_fp16_b4_wave"        : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from fc1000/Softmax --executors wave 0 host 1  --scheduler wave2 --schedule_options ' --enable_replication ' --batch 4 --images %s"%(rnPreFp16, getBatchedJpgs(4)), "--input_files %s" % (getBatchedJpgs(4))],
  "7-rn50_nne_fp16_b4_wave-two_banks"        : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from fc1000/Softmax --executors wave 0 host 1  --scheduler wave2 --schedule_options ' --enable_replication ' --batch 4 --images %s"%(rnPreFp16, getBatchedJpgs(4)), "--env SIM_ADD_FLAGS=' --dram_frequency 6400' --input_files %s" % (getBatchedJpgs(4))],
  "7-rn50_nne_fp16_b4_wave-fast_dram"        : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from fc1000/Softmax --executors wave 0 host 1  --scheduler wave2 --schedule_options ' --enable_replication ' --batch 4 --images %s"%(rnPreFp16, getBatchedJpgs(4)), "--env SIM_ADD_FLAGS=' --dram_frequency 0 --dram_latency 1' --input_files %s" % (getBatchedJpgs(4))],
  "8-rn50_nne_fp16_b16_wave"        : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from fc1000/Softmax --executors wave 0 host 1  --scheduler wave2 --schedule_options ' --relax_dependencies --enable_replication ' --batch 16 --images %s"%(rnPreFp16, getBatchedJpgs(16)), "--input_files %s" % (getBatchedJpgs(16))],
  "8-rn50_nne_fp16_b16_wave-two_banks"        : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from fc1000/Softmax --executors wave 0 host 1  --scheduler wave2 --schedule_options ' --relax_dependencies --enable_replication ' --batch 16 --images %s"%(rnPreFp16, getBatchedJpgs(16)), "--env SIM_ADD_FLAGS=' --dram_frequency 6400' --input_files %s" % (getBatchedJpgs(16))],
  "8-rn50_nne_fp16_b16_wave-fast_dram"        : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from fc1000/Softmax --executors wave 0 host 1  --scheduler wave2 --schedule_options ' --relax_dependencies --enable_replication ' --batch 16 --images %s"%(rnPreFp16, getBatchedJpgs(16)), "--env SIM_ADD_FLAGS=' --dram_frequency 0 --dram_latency 1' --input_files %s" % (getBatchedJpgs(16))],
  #"7-rn50_nne_fp16_wave-two_banks"        : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from fc1000/Softmax --executors wave 0 host 1  --scheduler wave --images %s" %(rnPreFp16, rnDogJpg), "--input_files %s" % rnDogJpg,  ],
  #"7-rn50_nne_fp16_wave-fast_dram"        : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from fc1000/Softmax --executors wave 0 host 1  --scheduler wave --images %s" %(rnPreFp16, rnDogJpg), "--input_files %s" % rnDogJpg,  ],
  "7-rn50_nne_fp16_host"        : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from fc1000/Softmax --executors host all --batch 2 --images %s %s" % (rnPreFp16, rnDogJpg, rnCatJpg), "--input_files %s %s" % (rnDogJpg, rnCatJpg)],
  "7-rn50_nne_fc_wave"          : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from avg_pool/AvgPool --executors wave 0 host 1  --scheduler wave2 --images %s" %(rnPreFp16, rnDogJpg), "--input_files %s" % rnDogJpg ],
  #"7-rn50_nne_fc_b16_wave"          : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from avg_pool/AvgPool --executors wave 0 host 1  --scheduler wave2 --batch 16 --images %s"%(rnPreFp16, getBatchedJpgs(16)), "--input_files %s" % (getBatchedJpgs(16))],
  "8-rn50_nne_conv_wave"        : [ "tf_pb", "resnet50_keras/resnet50_fp32_keras_opt.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition conv --executors host all wave 01 03 04 07 09 12 14 16 19 21 23 26 27 30 32 35 37 39 42 44 46 49 51 53 56 57 60 62 65 67 69 72 74 76 79 81 83 86 88 90 93 95 97 100 101 104 106 109 111 113 116 118 120  --scheduler wave --images %s" %(rnPreFp32, rnDogJpg), "--input_files %s" % rnDogJpg ],
  "7-rn50_nne_fp32_host"        : [ "tf_pb", "resnet50_keras/resnet50_fp32_keras_opt.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from fc1000/Softmax --executors host all --batch 2 --images %s" %(rnPreFp32, rnDogCatB2Fp32), "--input_files %s %s" % (rnDogJpg, rnCatJpg)],

  # Matmult
  "4-rn50_matmul_fp32_wave"   : [ "tf_pb",   "resnet50_keras/resnet50_fp32_keras_opt.pb","resnet50", " --input_node input_1  --depth 2  --debug 1 %s --partition from avg_pool/AvgPool --executors host 0 wave 1 --scheduler wave --images %s --wavegraph_checks structure data-race"% (rnPreFp32, rnDogJpg),"--input_files %s" % rnDogJpg ],
  #"4-rn50_matmul_fp32_waveopt"   : [ "tf_pb",   "resnet50_keras/resnet50_fp32_keras_opt.pb","resnet50", " --input_node input_1  --depth 2  --debug 1 %s --partition from avg_pool/AvgPool --executors host 0 waveopt 1 --scheduler wave --images %s"% (rnPreFp32, rnDogJpg),"--input_files %s" % rnDogJpg ],
  #"4-rn50_matmul_nosm_fp32_wave"   : [ "tf_pb",   "resnet50_keras/resnet50_fp32_keras_opt.pb","resnet50", " --input_node input_1  --depth 2  --debug 1 %s  --partition from avg_pool/AvgPool fc1000/Softmax --executors host 0 2 wave 1 --scheduler wave --images %s" %(rnPreFp32, rnDogJpg),"--input_files %s" % rnDogJpg ],
  "4-rn50_matmul_nosm_wave"   : [ "tf_pb",   "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", " --input_node input_1  --depth 2  --debug 1 %s  --partition from avg_pool/AvgPool fc1000/Softmax --executors host 0 2 wave 1 --scheduler wave2 --schedule_options ' --nname=generic' --images %s" %(rnPreFp16, rnDogJpg),"--input_files %s" % rnDogJpg ],

  # Resnet50 batching
  #"7-rn50_nne_fp16_waveopt"   : [ "tf_pb",   "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 --partition from fc1000/Softmax --executors waveopt 0 host 1  --scheduler wave2 --batch 1 --images linspace1", ],
  #"7-rn50_nne_fp16_b2_waveopt"   : [ "tf_pb",   "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from fc1000/Softmax --executors waveopt 0 host 1  --scheduler wave2 --batch 2 --images %s %s"%(rnPreFp16, rnDogJpg, rnCatJpg), "--input_files %s %s" % (rnDogJpg, rnCatJpg)],
  #"7-rn50_nne_fp16_b16_waveopt"   : [ "tf_pb",   "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 %s --partition from fc1000/Softmax --executors waveopt 0 host 1  --scheduler wave2 --batch 16 --images %s"%(rnPreFp16, getBatchedJpgs(16)), "--input_files %s" % (getBatchedJpgs(16))],
  
  # Multi-tpb
  "7-rn50_fp16_multi_tpb_o_host"        : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--show_op_name_in_kgraph --input_node input_1  --depth 2  --debug 1 %s --partition multi_tpb ops 6.7 --executors host all host 7  --scheduler wave --images %s --wavegraph_checks structure data-race" %(rnPreFp16, rnDogJpg), "--input_files %s" % rnDogJpg ],
  "7-rn50_fp16_multi_tpb_w_host"        : [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50", "--show_op_name_in_kgraph --input_node input_1  --depth 2  --debug 1 %s --partition multi_tpb weights 4 --executors host all host 7  --scheduler wave --images %s --wavegraph_checks structure data-race" %(rnPreFp16, rnDogJpg), "--input_files %s" % rnDogJpg ],
  
  # LSTM
  "4-ptb_word_lm1_host"      : [ "tf_pb",   "ptb_word_lm/keras_unrolled/model-b16s32h512.pb","lm", " --input_node embedding_1_input_1  --depth 3  --debug 0   --partition from  lstm_2_1/transpose_1  --executors host all --scheduler wave --schedule_options ' --nname=lm' --input_constants dropout_1/keras_learning_phase:0 False --exclude_ops_from_capture ^dropout_1_1/cond/ --images %s --wavegraph_checks structure data-race" % lstmD0T32, "--input_files %s" % lstmD0T32],
  "4-ptb_word_lm1"   : [ "tf_pb",   "ptb_word_lm/keras_unrolled/model-b16s32h512.pb", "lm", " --input_node input_1  --depth 3  --debug 0 %s --partition from avg_pool/AvgPool --executors host 0 wave 1 --scheduler wave --schedule_options ' --nname=lm' --images %s --wavegraph_checks structure data-race"% (rnPreFp32, rnDogJpg), "--input_files %s" % rnDogJpg ],
  "4-ptb_word_small1_host"   : [ "tf_pb",   "ptb_word_lm/keras_unrolled/model-b32s4h512.pb","lm", " --input_node embedding_1_input_1  --depth 3  --debug 0   --partition from  lstm_2_1/transpose_1  --executors host all --scheduler wave --schedule_options ' --nname=lm' --input_constants dropout_1/keras_learning_phase:0 False --exclude_ops_from_capture ^dropout_1_1/cond/ --images %s --wavegraph_checks structure data-race" % lstmD0T4, "--input_files %s" % lstmD0T4],
  "4-ptb_word_small1_wave"   : [ "tf_pb",   "ptb_word_lm/keras_unrolled/model-b32s4h512.pb","lm", " --input_node embedding_1_input_1  --depth 3  --debug 0   --partition from_multi  lstm_1_1/unstack,lstm_1_1/Tile_1  lstm_2_1/transpose_1  --executors host 0 2 wave 1 --scheduler wave --schedule_options ' --nname=lm' --input_constants dropout_1/keras_learning_phase:0 False --exclude_ops_from_capture ^dropout_1_1/cond/ --images %s --wavegraph_checks structure data-race" % lstmD0T4, "--input_files %s" % lstmD0T4],


  "4-ptb_word_small_sigmoid_wave"   : [ "tf_pb",   "ptb_word_lm/keras_unrolled/sigmoid/model-b32s4h512.pb","lm", " --input_node embedding_1_input_1  --depth 3  --debug 1   --partition from_multi  lstm_1_1/unstack,lstm_1_1/Tile_1,lstm_1_1/Tile,lstm_1_1/Tile_1  lstm_1_1/stack  lstm_2_1/transpose_1  --adjust_node_color lstm_1_1/Tile 0 lstm_1_1/Tile_1 0  --executors host 0 2 wave 1 --scheduler wave --schedule_options ' --nname=lm' --input_constants dropout_1/keras_learning_phase:0 False  --exclude_ops_from_capture ^dropout_1_1/cond/ --images %s" % lstmD0T4, "--input_files %s" % lstmD0T4],

  
  # LSTM small: 5 color 2-layer small host-tpb-host-tpb-host - waveopt and wave versions
  "4-ptb_word_small_sigmoid_2l_waveopt"  : [ "tf_pb",   "ptb_word_lm/keras_unrolled/sigmoid/model-b32s4h512.pb","lm", " --show_op_name_in_kgraph --input_node embedding_1_input_1  --depth 3  --debug 1   --partition from_multi  lstm_1_1/unstack,lstm_1_1/Tile_1,lstm_1_1/Tile,lstm_1_1/Tile_1  lstm_1_1/stack   lstm_2_1/unstack,lstm_2_1/Tile_1,lstm_2_1/Tile,lstm_2_1/Tile_1  lstm_2_1/stack     --adjust_node_color  lstm_1_1/Tile 0 lstm_1_1/Tile_1 0 lstm_2_1/Tile 2 lstm_2_1/Tile_1 2   --executors  waveopt 1 3  --scheduler wave --schedule_options ' --nname=lm' --input_constants dropout_1/keras_learning_phase:0 False  --exclude_ops_from_capture ^dropout_1_1/cond/ --images %s --wavegraph_checks structure data-race" % lstmD0T4, "--input_files %s" % lstmD0T4],


  "4-ptb_word_small_sigmoid_2l_wave"  : [ "tf_pb",   "ptb_word_lm/keras_unrolled/sigmoid/model-b32s4h512.pb","lm", " --show_op_name_in_kgraph --input_node embedding_1_input_1  --depth 3  --debug 1   --partition from_multi  lstm_1_1/unstack,lstm_1_1/Tile_1,lstm_1_1/Tile,lstm_1_1/Tile_1  lstm_1_1/stack   lstm_2_1/unstack,lstm_2_1/Tile_1,lstm_2_1/Tile,lstm_2_1/Tile_1  lstm_2_1/stack     --adjust_node_color  lstm_1_1/Tile 0 lstm_1_1/Tile_1 0 lstm_2_1/Tile 2 lstm_2_1/Tile_1 2   --executors  wave 1 3  --scheduler wave --schedule_options ' --nname=lm' --input_constants dropout_1/keras_learning_phase:0 False  --exclude_ops_from_capture ^dropout_1_1/cond/ --images %s" % lstmD0T4, "--input_files %s" % lstmD0T4],


  # Batched small LSTM
  "4-ptb_word_small_sigmoid_2l_b64_wave"  : [ "tf_pb",   "ptb_word_lm/keras_unrolled/sigmoid_b64/model-b64s4h512.pb","lm", " --show_op_name_in_kgraph   --depth 3  --debug 1   --partition from_multi  lstm_1_1/unstack,lstm_1_1/Tile_1,lstm_1_1/Tile,lstm_1_1/Tile_1  lstm_1_1/stack   lstm_2_1/unstack,lstm_2_1/Tile_1,lstm_2_1/Tile,lstm_2_1/Tile_1  lstm_2_1/stack     --adjust_node_color  lstm_1_1/Tile 0 lstm_1_1/Tile_1 0 lstm_2_1/Tile 2 lstm_2_1/Tile_1 2   --executors  wave 1 3  --scheduler wave --schedule_options ' --nname=lm' --input_constants dropout_1/keras_learning_phase:0 False  --exclude_ops_from_capture ^dropout_1_1/cond/ --images %s --batch 64 --wavegraph_checks structure data-race" % lstmD0T4B64, "--input_files %s" % lstmD0T4B64],
 
  # LSTM small: levelauto partitioning
  "4-ptb_word_small_sigmoid_2l_auto_waveopt"  : [ "tf_pb",   "ptb_word_lm/keras_unrolled/sigmoid/model-b32s4h512.pb","lm", " --show_op_name_in_kgraph --input_node embedding_1_input_1  --depth 3  --debug 1   --partition levelauto   --executors  waveopt 0 2 4 --scheduler wave --schedule_options ' --nname=lm' --input_constants dropout_1/keras_learning_phase:0 False  --exclude_ops_from_capture ^dropout_1_1/cond/ --images %s --wavegraph_checks structure data-race" % lstmD0T4, "--input_files %s" % lstmD0T4],
 
   # LSTM debug of matmult, only sg00 is usable
  "2-ptb_word_unstack_matmul4"  : [ "tf_pb",   "ptb_word_lm/keras_unrolled/sigmoid/model-b32s4h512.pb","lm", " --show_op_name_in_kgraph --input_node embedding_1_input_1  --depth 3  --debug 1   --partition from_multi  lstm_1_1/unstack,lstm_1_1/Tile_1,lstm_1_1/Tile,lstm_1_1/Tile_1   lstm_1_1/add_6,lstm_1_1/add_4,lstm_1_1/add_2,lstm_1_1/MatMul,lstm_1_1/MatMul_1,lstm_1_1/mul,lstm_1_1/mul_1  --adjust_node_color lstm_1_1/Tile 0 lstm_1_1/Tile_1 0  --executors wave 0 --scheduler wave --schedule_options ' --nname=lm' --input_constants dropout_1/keras_learning_phase:0 False  --exclude_ops_from_capture ^dropout_1_1/cond/ --images %s --wavegraph_checks structure data-race" % lstmD0T4, "--input_files %s" % lstmD0T4],
  
   # LSTM debug of matmult, only sg00 is usable
  "2-ptb_word_unstack_matmul1"  : [ "tf_pb",   "ptb_word_lm/keras_unrolled/sigmoid/model-b32s4h512.pb","lm", " --show_op_name_in_kgraph --input_node embedding_1_input_1  --depth 3  --debug 1   --partition from_multi  lstm_1_1/unstack,lstm_1_1/Tile_1,lstm_1_1/Tile,lstm_1_1/Tile_1   lstm_1_1/add_6,lstm_1_1/add_4,lstm_1_1/add_2,lstm_1_1/MatMul,lstm_1_1/MatMul_1,lstm_1_1/mul,lstm_1_1/mul_1  --adjust_node_color lstm_1_1/Tile 0 lstm_1_1/Tile_1 0 lstm_1_1/MatMul_2 0  lstm_1_1/MatMul_4 0  --executors wave 0 --scheduler wave --schedule_options ' --nname=lm' --input_constants dropout_1/keras_learning_phase:0 False  --exclude_ops_from_capture ^dropout_1_1/cond/ --images %s --wavegraph_checks structure data-race" % lstmD0T4, "--input_files %s" % lstmD0T4],
  #InceptionV3
#  "9-inceptionv3_wave" : ["tf_pb", "inceptionv3_fp16_keras_opt.pb", "inceptionv3", "--partition from mixed0/concat --executors host all --scheduler wave --schedule_options '--nname=generic --save_layer_output' --input_node input_1 --images linspace1", "--input_files trivnet_input_1:0.npy"],
#  "9-inceptionv3_wave_dog" : ["tf_pb", "inceptionv3_fp16_keras_opt.pb", "inceptionv3", "%s --partition from mixed0/concat --executors host all --scheduler wave --schedule_options '--nname=generic --save_layer_output' --input_node input_1 --images %s" %(incPreFp16, rnDogJpg), "--input_files %s" % rnDogJpg],
#"9-inceptionv3_wave_dog_pool" : ["tf_pb", "inceptionv3_fp16_keras_opt.pb", "inceptionv3", "%s --partition from max_pooling2d_2/MaxPool --executors host all --scheduler wave --schedule_options '--nname=generic --save_layer_output' --input_node input_1 --images %s" %(incPreFp16, rnDogJpg), "--input_files %s" % rnDogJpg],
#"9-inceptionv3_wave_dog_pool_sg00_tpb" : ["tf_pb", "inceptionv3_fp16_keras_opt.pb", "inceptionv3", "%s --partition from max_pooling2d_2/MaxPool --executors host 1 wave 0 --scheduler wave --schedule_options '--nname=generic --save_layer_output' --input_node input_1 --images %s" %(incPreFp16, rnDogJpg), "--input_files %s" % rnDogJpg],
#  "9-inceptionv3_wave_dog_2nd_conv2d" : ["tf_pb", "inceptionv3_fp16_keras_opt.pb", "inceptionv3", "%s --partition from conv2d_2/convolution --executors host all --scheduler wave --schedule_options '--nname=generic --save_layer_output' --input_node input_1 --images %s" %(incPreFp16, rnDogJpg), "--input_files %s" % rnDogJpg],
#  "9-inceptionv3_wave_dog_2nd_conv2d_sg00_tpb" : ["tf_pb", "inceptionv3_fp16_keras_opt.pb", "inceptionv3", "%s --partition from conv2d_2/convolution --executors host 1 wave 0 --scheduler wave --schedule_options '--nname=generic --save_layer_output' --input_node input_1 --images %s" %(incPreFp16, rnDogJpg), "--input_files %s" % rnDogJpg],
#  "9-inceptionv3_wave_dog_1st_pool" : ["tf_pb", "inceptionv3_fp16_keras_opt.pb", "inceptionv3", "%s --partition from max_pooling2d_1/MaxPool --executors host all --scheduler wave --schedule_options '--nname=generic --save_layer_output' --input_node input_1 --images %s" %(incPreFp16, rnDogJpg), "--input_files %s" % rnDogJpg],
#  "9-inceptionv3_wave_dog_1st_pool_sg00_tpb" : ["tf_pb", "inceptionv3_fp16_keras_opt.pb", "inceptionv3", "%s --partition from max_pooling2d_1/MaxPool --executors host 1 wave 0 --scheduler wave --schedule_options '--nname=generic --save_layer_output' --input_node input_1 --images %s" %(incPreFp16, rnDogJpg), "--input_files %s" % rnDogJpg],
#  "9-inceptionv3_wave_dog_3rd_biasadd" : ["tf_pb", "inceptionv3_fp16_keras_opt.pb", "inceptionv3", "%s --partition from batch_normalization_3/FusedBatchNorm_1 --executors host all --scheduler wave --schedule_options '--nname=generic --save_layer_output' --input_node input_1 --images %s" %(incPreFp16, rnDogJpg), "--input_files %s" % rnDogJpg],
#  "9-inceptionv3_wave_dog_3rd_biasadd_sg00_tpb" : ["tf_pb", "inceptionv3_fp16_keras_opt.pb", "inceptionv3", "%s --partition from batch_normalization_3/FusedBatchNorm_1 --executors host 1 wave 0 --scheduler wave --schedule_options '--nname=generic --save_layer_output' --input_node input_1 --images %s" %(incPreFp16, rnDogJpg), "--input_files %s" % rnDogJpg],
"9-inceptionv3_wave_dog_sg00_tpb" : ["tf_pb", "inceptionv3/inceptionv3_fp16_keras_opt.pb", "inceptionv3", "%s --partition from mixed0/concat --executors host 1 wave 0 --scheduler wave2 --schedule_options '--nname=generic --save_layer_output' --input_node input_1 --images %s" %(incPreFp16, rnDogJpg), "--input_files %s" % rnDogJpg],
"9-inceptionv3_wave_dog_sg00_tpb_upto_concat1" : ["tf_pb", "inceptionv3/inceptionv3_fp16_keras_opt.pb", "inceptionv3", "%s --partition from_multi conv2d_14/convolution conv2d_16/convolution conv2d_13/convolution average_pooling2d_2/AvgPool --executors host 1 wave 0 --scheduler wave2 --schedule_options '--nname=generic --save_layer_output' --input_node input_1 --images %s" %(incPreFp16, rnDogJpg), "--input_files %s" % rnDogJpg],
"9-inceptionv3_wave_dog_sg00_tpb_upto_concat2" : ["tf_pb", "inceptionv3/inceptionv3_fp16_keras_opt.pb", "inceptionv3", "%s --partition from_multi conv2d_20/convolution conv2d_21/convolution conv2d_23/convolution average_pooling2d_3/AvgPool --executors host 1 wave 0 --scheduler wave2 --schedule_options '--nname=generic --save_layer_output' --input_node input_1 --images %s" %(incPreFp16, rnDogJpg), "--input_files %s" % rnDogJpg],
#"9-inceptionv3_wave_dog_sg00_tpb_upto_concat3" : ["tf_pb", "inceptionv3/inceptionv3_fp16_keras_opt.pb", "inceptionv3", "%s --partition from_multi conv2d_28/convolution conv2d_27/convolution max_pooling2d_3/MaxPool --executors host 1 wave 0 --scheduler wave2 --schedule_options '--nname=generic --save_layer_output' --input_node input_1 --images %s" %(incPreFp16, rnDogJpg), "--input_files %s" % rnDogJpg],
#"9-inceptionv3_wave_dog_sg00_tpb_before_concat3" : ["tf_pb", "inceptionv3/inceptionv3_fp16_keras_opt.pb", "inceptionv3", "%s --partition from mixed2/concat --executors host 1 wave 0 --scheduler wave2 --schedule_options '--nname=generic --save_layer_output' --input_node input_1 --images %s" %(incPreFp16, rnDogJpg), "--input_files %s" % rnDogJpg],
#"9-inceptionv3_wave_dog_sg00_tpb_before_relus" : ["tf_pb", "inceptionv3/inceptionv3_fp16_keras_opt.pb", "inceptionv3", "%s --partition from_multi batch_normalization_22/FusedBatchNorm_1  batch_normalization_20/FusedBatchNorm_1 batch_normalization_25/FusedBatchNorm_1 batch_normalization_26/FusedBatchNorm_1 --executors host 1 wave 0 --scheduler wave2 --schedule_options '--nname=generic --save_layer_output' --input_node input_1 --images %s" %(incPreFp16, rnDogJpg), "--input_files %s" % rnDogJpg],
#"9-inceptionv3_wave_dog_sg00_tpb_upto_concat4" : ["tf_pb", "inceptionv3/inceptionv3_fp16_keras_opt.pb", "inceptionv3", "%s --partition from_multi conv2d_31/convolution conv2d_32/convolution conv2d_35/convolution average_pooling2d_4/AvgPool --executors host 1 wave 0 --scheduler wave2 --schedule_options '--nname=generic --save_layer_output' --input_node input_1 --images %s" %(incPreFp16, rnDogJpg), "--input_files %s" % rnDogJpg],
#"9-inceptionv3_wave_dog_sg00_tpb_upto_concat5" : ["tf_pb", "inceptionv3/inceptionv3_fp16_keras_opt.pb", "inceptionv3", "%s --partition from_multi conv2d_41/convolution conv2d_42/convolution conv2d_45/convolution average_pooling2d_5/MaxPool --executors host 1 wave 0 --scheduler wave2 --schedule_options '--nname=generic --save_layer_output' --input_node input_1 --images %s" %(incPreFp16, rnDogJpg), "--input_files %s" % rnDogJpg],
#"9-inceptionv3_wave_dog" : ["tf_pb", "inceptionv3/inceptionv3_fp16_keras_opt.pb", "inceptionv3", "%s --executors host 1 wave 0 --scheduler wave2 --schedule_options '--nname=generic --save_layer_output' --input_node input_1 --images %s" %(incPreFp16, rnDogJpg), "--input_files %s" % rnDogJpg],
  "0-1conv1maxpool_wave_k3d1"  : [ "trivnet_conv_pool", "tfloat16-b1-h4-r1-s1-c1-m1-VALID-MaxPool-k3-d1-wmin2-wmax2.2-imin1-imax16", "1conv1pool", "--scheduler wave2 --schedule_options ' --nname=generic ' --wavegraph_checks structure data-race"],
  "0-1conv1maxpool_wave_h17k3d1"  : [ "trivnet_conv_pool", "tfloat16-b1-h17-r1-s1-c1-m1-VALID-MaxPool-k3-d1-wmin2-wmax2.2-imin1-imax16", "1conv1pool", "--scheduler wave2 --schedule_options ' --nname=generic ' --wavegraph_checks structure data-race"],
  "0-1conv1maxpool_wave_h17c128k3d1"  : [ "trivnet_conv_pool", "tfloat16-b1-h17-r1-s1-c128-m1-VALID-MaxPool-k3-d1-wmin2-wmax2.2-imin1-imax16", "1conv1pool", "--scheduler wave2 --schedule_options ' --nname=generic ' --wavegraph_checks structure data-race"],
  "0-1conv1maxpool_wave_h17c196k3d1"  : [ "trivnet_conv_pool", "tfloat16-b1-h17-r1-s1-c196-m1-VALID-MaxPool-k3-d1-wmin2-wmax2.2-imin1-imax16", "1conv1pool", "--scheduler wave2 --schedule_options ' --nname=generic ' --wavegraph_checks structure data-race"],
  "0-1conv1maxpool_wave_h71k3d1"  : [ "trivnet_conv_pool", "tfloat16-b1-h71-r1-s1-c1-m1-VALID-MaxPool-k3-d1-wmin2-wmax2.2-imin1-imax16", "1conv1pool", "--scheduler wave2 --schedule_options ' --nname=generic ' --wavegraph_checks structure data-race"],
  #"0-1conv1maxpool_wave_h71c192k3d1"  : [ "trivnet_conv_pool", "tfloat16-b1-h71-r1-s1-c192-m1-VALID-MaxPool-k3-d1-wmin2-wmax2.2-imin1-imax16", "1conv1pool", "--scheduler wave2 --schedule_options ' --nname=generic ' --wavegraph_checks structure data-race"],
  "0-1maxpool_wave_h71c192m192k3d1_valid"  : [ "trivnet_pool", "tfloat16-b1-h71-r1-s1-c192-m192-VALID-MaxPool-k3-d1-wmin-0.1-wmax0.2-imin-0.2-imax0.3", "1pool", "--scheduler wave2 --schedule_options ' --nname=generic ' --wavegraph_checks structure data-race"],
  "0-1maxpool_wave_h71c192m192k3d1_same"  : [ "trivnet_pool", "tfloat16-b1-h71-r1-s1-c192-m192-SAME-MaxPool-k3-d1-wmin-0.1-wmax0.2-imin-0.2-imax0.3", "1pool", "--scheduler wave2 --schedule_options ' --nname=generic ' --wavegraph_checks structure data-race"],
  "0-1avgpool_wave_h71c192m192k3d1"  : [ "trivnet_pool", "tfloat16-b1-h71-r1-s1-c192-m192-SAME-AvgPool-k3-d1-wmin-0.1-wmax0.2-imin-0.2-imax0.3", "1pool", "--scheduler wave2 --schedule_options ' --nname=generic ' --wavegraph_checks structure data-race"],
  "0-1avgpool_wave_h71c1m1k3d1_same"  : [ "trivnet_pool", "tfloat16-b1-h71-r1-s1-c1-m1-SAME-AvgPool-k3-d1-wmin-0.1-wmax0.2-imin-0.2-imax0.3", "1pool", "--scheduler wave2 --schedule_options ' --nname=generic ' --wavegraph_checks structure data-race"],
  "0-1avgpool_wave_h1c192m192k1d1_same"  : [ "trivnet_pool", "tfloat16-b1-h1-r1-s1-c192-m192-SAME-AvgPool-k1-d1-wmin-0.1-wmax0.2-imin-0.2-imax0.3", "1pool", "--scheduler wave2 --schedule_options ' --nname=generic ' --wavegraph_checks structure data-race"],
  "0-1avgpool_wave_h71c1m1k3d1_valid"  : [ "trivnet_pool", "tfloat16-b1-h71-r1-s1-c1-m1-VALID-AvgPool-k3-d1-wmin-0.1-wmax0.2-imin-0.2-imax0.3", "1pool", "--scheduler wave2 --schedule_options ' --nname=generic ' --wavegraph_checks structure data-race"],
  "0-1avgpool_wave_h1c192m192k1d1_valid"  : [ "trivnet_pool", "tfloat16-b1-h1-r1-s1-c192-m192-VALID-AvgPool-k1-d1-wmin-0.1-wmax0.2-imin-0.2-imax0.3", "1pool", "--scheduler wave2 --schedule_options ' --nname=generic ' --wavegraph_checks structure data-race"],
  "0-1maxpool_wave_h71c1m1k3d1_same"  : [ "trivnet_pool", "tfloat16-b1-h71-r1-s1-c1-m1-SAME-MaxPool-k3-d1-wmin-0.1-wmax0.2-imin-0.2-imax0.3", "1pool", "--scheduler wave2 --schedule_options ' --nname=generic ' --wavegraph_checks structure data-race"],
  "0-1maxpool_wave_h71c1m1k3d2_valid"  : [ "trivnet_pool", "tfloat16-b1-h71-r1-s1-c1-m1-VALID-MaxPool-k3-d2-wmin-0.1-wmax0.2-imin-0.2-imax0.3", "1pool", "--scheduler wave2 --schedule_options ' --nname=generic ' --wavegraph_checks structure data-race"],
  "0-1maxpool_wave_h71c1m1k3d2_same"  : [ "trivnet_pool", "tfloat16-b1-h71-r1-s1-c1-m1-SAME-MaxPool-k3-d2-wmin-0.1-wmax0.2-imin-0.2-imax0.3", "1pool", "--scheduler wave2 --schedule_options ' --nname=generic ' --wavegraph_checks structure data-race"],
  "0-1maxpool_wave_h71c192m192k3d2_valid"  : [ "trivnet_pool", "tfloat16-b1-h71-r1-s1-c192-m192-VALID-MaxPool-k3-d2-wmin-0.1-wmax0.2-imin-0.2-imax0.3", "1pool", "--scheduler wave2 --schedule_options ' --nname=generic ' --wavegraph_checks structure data-race"],
  "0-1maxpool_wave_h1c192m192k1d1_same"  : [ "trivnet_pool", "tfloat16-b1-h1-r1-s1-c192-m192-SAME-MaxPool-k1-d1-wmin-0.1-wmax0.2-imin-0.2-imax0.3", "1pool", "--scheduler wave2 --schedule_options ' --nname=generic ' --wavegraph_checks structure data-race"],
  "0-1maxpool_wave_h71c1m1k3d1_valid"  : [ "trivnet_pool", "tfloat16-b1-h71-r1-s1-c1-m1-VALID-MaxPool-k3-d1-wmin-0.1-wmax0.2-imin-0.2-imax0.3", "1pool", "--scheduler wave2 --schedule_options ' --nname=generic ' --wavegraph_checks structure data-race"],
  "0-1maxpool_wave_h1c192m192k1d1_valid"  : [ "trivnet_pool", "tfloat16-b1-h1-r1-s1-c192-m192-VALID-MaxPool-k1-d1-wmin-0.1-wmax0.2-imin-0.2-imax0.3", "1pool", "--scheduler wave2 --schedule_options ' --nname=generic ' --wavegraph_checks structure data-race"],
  # Start of AvgPools in InceptionV3
  "0-1avgpool_wave_h35c192m192k3d1_same"  : [ "trivnet_pool", "tfloat16-b1-h35-r1-s1-c192-m192-SAME-AvgPool-k3-d1-wmin-0.1-wmax0.2-imin-0.2-imax0.3", "1pool", "--scheduler wave2 --schedule_options ' --nname=generic ' --wavegraph_checks structure data-race"],
  "0-1avgpool_wave_h35c256m256k3d1_same"  : [ "trivnet_pool", "tfloat16-b1-h35-r1-s1-c256-m256-SAME-AvgPool-k3-d1-wmin-0.1-wmax0.2-imin-0.2-imax0.3", "1pool", "--scheduler wave2 --schedule_options ' --nname=generic ' --wavegraph_checks structure data-race"],
  "0-1avgpool_wave_h35c288m288k3d1_same"  : [ "trivnet_pool", "tfloat16-b1-h35-r1-s1-c288-m288-SAME-AvgPool-k3-d1-wmin-0.1-wmax0.2-imin-0.2-imax0.3", "1pool", "--scheduler wave2 --schedule_options ' --nname=generic ' --wavegraph_checks structure data-race"],
  "0-1avgpool_wave_h17c768m768k3d1_same"  : [ "trivnet_pool", "tfloat16-b1-h17-r1-s1-c768-m768-SAME-AvgPool-k3-d1-wmin-0.1-wmax0.2-imin-0.2-imax0.3", "1pool", "--scheduler wave2 --schedule_options ' --nname=generic ' --wavegraph_checks structure data-race"],
  "0-1avgpool_wave_h8c1280m1280k3d1_same"  : [ "trivnet_pool", "tfloat16-b1-h8-r1-s1-c1280-m1280-SAME-AvgPool-k3-d1-wmin-0.1-wmax0.2-imin-0.2-imax0.3", "1pool", "--scheduler wave2 --schedule_options ' --nname=generic ' --wavegraph_checks structure data-race"],
  "0-1avgpool_wave_h8c2048m2048k3d1_same"  : [ "trivnet_pool", "tfloat16-b1-h8-r1-s1-c2048-m2048-SAME-AvgPool-k3-d1-wmin-0.1-wmax0.2-imin-0.2-imax0.3", "1pool", "--scheduler wave2 --schedule_options ' --nname=generic ' --wavegraph_checks structure data-race"],
  # End of AvgPools in InceptionV3
  "0-1conv1avgpool_wave_h35c192m192k3d1"  : [ "trivnet_conv_pool", "tfloat16-b1-h35-r1-s1-c192-m192-SAME-AvgPool-k3-d1-wmin-0.1-wmax0.2-imin-0.2-imax0.3", "1conv1pool", "--scheduler wave2 --schedule_options ' --nname=generic ' --wavegraph_checks structure data-race"],
  "0-1conv1avgpool_wave_h35c128m128k3d1"  : [ "trivnet_conv_pool", "tfloat16-b1-h35-r1-s1-c128-m128-SAME-AvgPool-k3-d1-wmin-0.1-wmax0.2-imin-0.2-imax0.3", "1conv1pool", "--scheduler wave2 --schedule_options ' --nname=generic ' --wavegraph_checks structure data-race"],
  "0-1conv1avgpool_wave_k3d1"  : [ "trivnet_conv_pool", "tfloat16-b1-h4-r1-s1-c1-m1-SAME-AvgPool-k3-d1-wmin2-wmax2.2-imin1-imax16", "1conv1pool", "--scheduler wave2 --wavegraph_checks structure data-race"],
  "0-1conv1avgpool_wave_h17k3d1"  : [ "trivnet_conv_pool", "tfloat16-b1-h17-r1-s1-c1-m1-SAME-AvgPool-k3-d1-wmin2-wmax2.2-imin1-imax16", "1conv1pool", "--scheduler wave2 --wavegraph_checks structure data-race"],
  "0-1conv1avgpool_wave_h16k3d1"  : [ "trivnet_conv_pool", "tfloat16-b1-h16-r1-s1-c1-m1-SAME-AvgPool-k3-d1-wmin2-wmax2.2-imin1-imax16", "1conv1pool", "--scheduler wave2 --wavegraph_checks structure data-race"],
  "0-1conv1avgpool_wave_h18k3d1"  : [ "trivnet_conv_pool", "tfloat16-b1-h18-r1-s1-c1-m1-SAME-AvgPool-k3-d1-wmin2-wmax2.2-imin1-imax16", "1conv1pool", "--scheduler wave2 --wavegraph_checks structure data-race"],
  "0-1conv1avgpool_wave_h32k3d1"  : [ "trivnet_conv_pool", "tfloat16-b1-h32-r1-s1-c1-m1-SAME-AvgPool-k3-d1-wmin2-wmax2.2-imin1-imax16", "1conv1pool", "--scheduler wave2 --wavegraph_checks structure data-race"],
  "0-1conv1avgpool_wave_h35k3d1"  : [ "trivnet_conv_pool", "tfloat16-b1-h32-r1-s1-c1-m1-SAME-AvgPool-k3-d1-wmin2-wmax2.2-imin1-imax16", "1conv1pool", "--scheduler wave2 --schedule_options ' --nname=generic ' --wavegraph_checks structure data-race"],
  "0-1conv1avgpool_wave_h35c16k3d1"  : [ "trivnet_conv_pool", "tfloat16-b1-h32-r1-s1-c16-m1-SAME-AvgPool-k3-d1-wmin2-wmax2.2-imin1-imax16", "1conv1pool", "--scheduler wave2 --schedule_options ' --nname=generic ' --wavegraph_checks structure data-race"],
  "0-1conv1avgpool_wave_h35c196k3d1"  : [ "trivnet_conv_pool", "tfloat16-b1-h32-r1-s1-c196-m1-SAME-AvgPool-k3-d1-wmin2-wmax2.2-imin1-imax16", "1conv1pool", "--scheduler wave2 --schedule_options ' --nname=generic ' --wavegraph_checks structure data-race"],
  "0-1conv_h17c196r1_wave"  : [ "trivnet_conv1",  "tfloat16-b1-h17-r1-s1-c196-m1-wmin-0.1-wmax0.12-imin-0.2-imax0.25", "1conv", "--scheduler wave2 --wavegraph_checks structure data-race"],
  "0-1maxpool_wave_h65c1m1k3d1_valid"  : [ "trivnet_pool", "tfloat16-b1-h65-r1-s1-c1-m1-VALID-MaxPool-k3-d1-wmin-0.1-wmax0.2-imin-0.2-imax0.3", "1pool", "--scheduler wave2 --schedule_options ' --nname=generic ' --wavegraph_checks structure data-race"],
  

  "0-1conv-h8r3c4m4-relu_wave"              : [ "trivnet_lin",      "tfloat16-l2-b1-h8-r3-s1-c4-m4-relu-wmin-0.2-wmax0.4-imin-1000-imax1010", "1cr", "--scheduler wave2"],
  "0-rtl-1conv1maxpool_h8r3c5m4_val_wave"    : [ "trivnet_conv_pool","tfloat16-b1-h8-r3-s1-c5-m4-VALID-MaxPool-k2-d2-wmin2-wmax2.2-imin1-imax16", "1conv1pool", "--scheduler wave2"],


  "0-1conv1maxpool_c128m64h16_val_wave"    : [ "trivnet_conv_pool","tfloat16-b1-h16-r3-s1-c128-m64-VALID-MaxPool-k2-d2-wmin2-wmax2.2-imin1-imax16", "1conv1pool", "--scheduler wave2"],
  ##"0-1conv1maxpool_c128m64h128_val_wave"    : [ "trivnet_conv_pool","tfloat16-b1-h128-r3-s1-c128-m64-VALID-MaxPool-k2-d2-wmin2-wmax2.2-imin1-imax16", "1conv1pool", "--scheduler wave2"],
  "0-1conv1maxpool_c128m64h128_val_wave"    : [ "trivnet_conv_pool","tfloat16-b1-h128-r3-s1-c128-m64-VALID-MaxPool-k2-d2-wmin2-wmax2.2-imin1-imax16", "1conv1pool", "--scheduler wave  --schedule_options ' --nname=generic '  "],

  ##"0-1conv_c128m64h128_wave"  : [ "trivnet_conv1",  "tfloat16-b1-h128-r3-s1-c128-m64-wmin2-wmax2.2-imin1-imax16", "1conv", "--scheduler wave2 --wavegraph_checks structure data-race"],
  "0-1conv_c128m64h128_waveg"  : [ "trivnet_conv1",  "tfloat16-b1-h128-r3-s1-c128-m64-wmin2-wmax2.2-imin-1-imax1.6", "1conv", "--scheduler wave --schedule_options ' --nname=generic ' --wavegraph_checks structure data-race"],

  "0-11conv_tanh_wave" : [ "trivnet_lin",   "tfloat16-l11-b1-h4-r3-s1-c1-m1-tanh-wmin-0.2-wmax0.8-imin-4-imax8", "11ct", "--scheduler wave2 --wavegraph_checks structure data-race"],
  "0-2conv_tanh_wave" : [ "trivnet_lin",   "tfloat16-l2-b1-h4-r3-s1-c1-m1-tanh-wmin-0.2-wmax0.8-imin-4-imax8", "2ct", "--scheduler wave2 --wavegraph_checks structure data-race"],
}

# Regression waiver mechanism 
# If the testname matches the regexp then the FAIL status is replaced with
# with the string
testWaiver = [
    ['0-1conv1maxpool_k3d2_wave',   'WAIVE_WAVESC'],
    ['0-1conv1pool_b5_wave',        'WAIVE_WAVESC'],
    ['0-1conv1pool_b5m3_wave',      'WAIVE_WAVESC'],
    ['0-.*concat_.*', 'WAIVE_INCEPTIONV3'],
    ['0-3conv_1concat', 'WAIVE_INCEPTIONV3'],
    ['9-inceptionv3_wave*', 'WAIVE_INCEPTIONV3'],
    ['9-inceptionv3_wave_dog_sg00_tpb', 'WAIVE_INCEPTIONV3'],
    ['0-1conv1avgpool_wave_h16k3d1', 'WAIVE_INCEPTIONV3'],
    ['0-1conv1avgpool_wave_h17k3d1', 'WAIVE_INCEPTIONV3'],
    ['0-1conv1avgpool_wave_h18k3d1', 'WAIVE_INCEPTIONV3'],
    ['0-1conv1avgpool_wave_h32k3d1', 'WAIVE_INCEPTIONV3'],
    ['0-1conv1avgpool_wave_h35c128m128k3d1', 'WAIVE_INCEPTIONV3'],
    ['0-1conv1avgpool_wave_h35c16k3d1', 'WAIVE_INCEPTIONV3'],
    ['0-1conv1avgpool_wave_h35c192m192k3d1', 'WAIVE_INCEPTIONV3'],
    ['0-1conv1avgpool_wave_h35c196k3d1', 'WAIVE_INCEPTIONV3'],
    ['0-1conv1avgpool_wave_h35k3d1', 'WAIVE_INCEPTIONV3'],
    ['0-1conv1avgpool_wave_k3d1', 'WAIVE_INCEPTIONV3'],
    ['0-1conv1maxpool_wave_h17c128k3d1', 'WAIVE_INCEPTIONV3'],
    ['0-1conv1maxpool_wave_h17c196k3d1', 'WAIVE_INCEPTIONV3'],
    ['0-1conv1maxpool_wave_h17k3d1', 'WAIVE_INCEPTIONV3'],
    ['0-1conv1maxpool_wave_h71k3d1', 'WAIVE_INCEPTIONV3'],
    ['0-1maxpool_wave_h65c1m1k3d1_valid', 'WAIVE_INCEPTIONV3'],
    ['0-1maxpool_wave_h71c1m1k3d2_same', 'WAIVE_INCEPTIONV3'],

    #['^0-act_wave$',   'WAIVE-KAENA452'],

    # ME does not support resadd as the first op which is ok
    ['^0-resadd_.*_wave$', 'WAIVE-RESADD_ME'],

    ['1-1conv0_r3h55c256_wave',     'WAIVE_WAVESC'],
    ['1-1conv_transpose_.*',     'WAIVE_KAENA570'],

    ['4-rn50_matmul_fp32_wave$',      'WAIVE-S10_BE_SOFTMAX'],

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
    #['4-ptb_word_small_sigmoid_wave$', 'WAIVE-LSTM'],
    ['0-scaleadd_wave',             'WAIVE-LSTM'],
    #['4-ptb_word_small_sigmoid_2l_wave$',             'WAIVE-LSTM'],
    ['2-ptb_word_unstack_.*',             'WAIVE-SG00'],
    ['4-ptb_word_small_sigmoid_2l_auto_waveopt',   'WAIVE-L_PART'],
    ['4-ptb_word_small_sigmoid_2l_b64_wave',   'WAIVE-LSTM_ME'],
    ['^(4-ptb_word_small_sigmoid_wave|4-ptb_word_small_sigmoid_2l_waveopt|4-ptb_word_small_sigmoid_2l_wave)$',   'WAIVE-ME_HNWC'],

    # Multi-tpb partitioning - passes in host mode so no need to waive
    #['7-rn50_fp16_multi_tpb_o_wave', 'WAIVE_MTPB'],
    #['7-rn50_fp16_multi_tpb_w_wave', 'WAIVE_MTPB'],

    # batching
    #['7-rn50_nne_fp16_waveopt_b\d+$', 'WAIVE_BATCH'],
    ['7-rn50_nne_fp32_wave$', 'WAIVE_SB_PRESERVE'],
    ['8-rn50_nne_fp32_meauto$', 'WAIVE_SB_PRESERVE'],
    ['8-rn50_nne_fp16_b16_wave-fast_dram$',  'WAIVE-ALIGNMENT'],
    ['8-rn50_nne_fp16_b16_wave-two_banks$',  'WAIVE-ALIGNMENT'],
    ['8-rn50_nne_fp16_b16_wave$',  'WAIVE-ALIGNMENT'],

    # bugs
    #['5-rn50_nne_to_act13_b16_wave-fast_dram$', 'WAIVE_BUG_KAENA454'],

    # Replication
    ['^[0-2]-rn50-.*_wave_repl$', 'WAIVE_REPLICATION'],

    # Resnet 152
    ['^9-resnet152', 'WAIVE_RN152'],
    #['0-10conv_relu_wave', 'WAIVE_BUG_KAENA411'],
    #['0-3conv_relu_wave', 'WAIVE_BUG_KAENA411'],

    # Qemu only works on C5 (till we add VDI to S3)
    #['^0-1conv0_qemu_wave$', 'WAIVE_QEMU'],

    [ '0-3resadd_fp16_wave', 'WAIVE_SBSB_ADD'],
  ]

noGpuTestWaiver = [
]

qemuTestWaiver = [
    #['3-1conv1maxpool_k3d2_wave$',  'WAIVE-QEMU'],
    #['3-rn50-16_b2_wave_repl$',  'WAIVE-QEMU'],
    #['3-rn50-16_wave_repl$',  'WAIVE-QEMU'],
    ['5-rn50_nne_to_act13_b16_wave-fast_dram$',  'WAIVE-QEMU'],
    #['5-rn50_nne_to_act4_b4_wave$',  'WAIVE-QEMU'],
    ['6-rn50_nne_to_act40_b16_wave-fast_dram$',  'WAIVE-QEMU'],
    #['7-rn50_nne_fp16_b4_wave-fast_dram$',  'WAIVE-QEMU'],
    #['7-rn50_nne_fp16_b4_wave-two_banks$',  'WAIVE-QEMU'],
    #['7-rn50_nne_fp16_b4_wave$',  'WAIVE-QEMU'],
    #['7-rn50_nne_fp16_wave-fast_dram$',  'WAIVE-QEMU'],
    #['7-rn50_nne_fp16_wave-two_banks$',  'WAIVE-QEMU'],
    #['7-rn50_nne_fp16_wave$',  'WAIVE-QEMU'],
    #['7-rn50_nne_fp16_ap_wave$',  'WAIVE-QEMU_REPLIC'],
    ['8-rn50_nne_conv_wave$',  'WAIVE-QEMU'],
    ['8-rn50_nne_fp16_b16_wave-fast_dram$',  'WAIVE-QEMU'],
    ['8-rn50_nne_fp16_b16_wave-two_banks$',  'WAIVE-QEMU'],
    ['8-rn50_nne_fp16_b16_wave$',  'WAIVE-QEMU'],
    
]

