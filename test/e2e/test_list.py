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
fp16AccJpg = "%s/%s" % (kePath, "images/3404.jpg")

rnPreFp32 = "%s/%s" % (kPath, "compiler/util/res50_img2fmap_fp32")
rnPreFp16 = "%s/%s" % (kPath, "compiler/util/res50_img2fmap_fp16")
rnPost = "%s/%s" % (kPath, "compiler/util/res50_classify")

lstmD0B4 = "%s/%s" % (kePath, "apps/tf/ptb_word_lm/keras_unrolled/data-b4-0.npy")
lstmD0B32 = "%s/%s" % (kePath, "apps/tf/ptb_word_lm/keras_unrolled/data-b32-0.npy")

testConfigMap = {
  "0-1conv0"      : [ "trivnet_conv1",  "tfloat16-b1-h1-r1-s1-c1-m1-wmin2-wmax2.2-imin3-imax3.2", "1conv"],
  "0-1conv0_wave" : [ "trivnet_conv1",  "tfloat16-b1-h1-r1-s1-c1-m1-wmin2-wmax2.2-imin3-imax3.2", "1conv", "--scheduler wave"],
  "0-1conv0_padvalid_wave" : [ "trivnet_conv1_padvalid",  "tfloat16-b1-h229-r7-s2-c1-m1-wmin2-wmax2.2-imin3-imax3.2", "1conv", "--scheduler wave --debug 3"],
  "0-1conv0_h16r2s2_wave"      : [ "trivnet_conv1",  "tfloat16-b1-h16-r2-s2-c1-m1-wmin-2-wmax2.2-imin-3-imax3.3", "1conv", "--scheduler wave"],
  "0-1conv0_h16r3s2_wave"      : [ "trivnet_conv1",  "tfloat16-b1-h16-r3-s2-c1-m1-wmin-2-wmax2.2-imin-3-imax3.3", "1conv", "--scheduler wave"],
  "0-1conv0_c1h2" : [ "trivnet_conv1",  "tfloat16-b1-h2-r1-s1-c1-m1-wmin2-wmax2.2-imin1-imax7", "1conv"],
  "0-1conv0_c1h2_wave" : [ "trivnet_conv1",  "tfloat16-b1-h2-r1-s1-c1-m1-wmin2-wmax2.2-imin1-imax7", "1conv", "--scheduler wave"],
  "0-1conv0_c1h16_wave" : [ "trivnet_conv1",  "tfloat16-b1-h16-r1-s1-c1-m1-wmin2-wmax2.2-imin3-imax3.2", "1conv", "--scheduler wave"],
  "0-2conv0"      : [ "trivnet_conv2",  "b1-h1-r1-s1-c1-m1-wmin2-wmax3-imin5-imax5.5", "2conv"],
  "0-neg"         : [ "trivnet_conv2",  "b1-Zh1-r1-s1-c1-m1-wmin2-wmax3-imin5-imax5.5", "I_ALWAYS_FAIL"],
  "0-1conv_tile"  : [ "trivnet_conv1",  "tfloat16-b1-h35-r3-s1-c1-m1-wmin-0.1-wmax0.12-imin-0.2-imax0.25", "1conv"],
  "0-1conv_tile_wave"  : [ "trivnet_conv1",  "tfloat16-b1-h35-r3-s1-c1-m1-wmin-0.1-wmax0.12-imin-0.2-imax0.25", "1conv", "--scheduler wave"],

  "0-1conv_h4r1_wave"  : [ "trivnet_conv1",  "tfloat16-b1-h4-r1-s1-c1-m1-wmin-0.1-wmax0.12-imin-0.2-imax0.25", "1conv", "--scheduler wave"],
  "0-1conv_h4r2s2_wave"  : [ "trivnet_conv1",  "tfloat16-b1-h4-r2-s2-c1-m1-wmin-0.1-wmax0.12-imin-0.2-imax0.25", "1conv", "--scheduler wave"],
  "0-1conv_h6r2s3_wave"  : [ "trivnet_conv1",  "tfloat16-b1-h6-r2-s3-c1-m1-wmin-0.1-wmax0.12-imin-0.2-imax0.25", "1conv", "--scheduler wave"],
  "0-1conv_h6r3s2_wave"  : [ "trivnet_conv1",  "tfloat16-b1-h6-r3-s2-c1-m1-wmin-0.1-wmax0.12-imin-0.2-imax0.25", "1conv", "--scheduler wave"],
  "0-1conv_h4r3s1_wave"  : [ "trivnet_conv1",  "tfloat16-b1-h4-r3-s1-c1-m1-wmin0-wmax9-imin0-imax15", "1conv", "--scheduler wave"],
  "0-1conv_h4r3s1"  : [ "trivnet_conv1",  "tfloat16-b1-h4-r3-s1-c1-m1-wmin0-wmax9-imin0-imax15", "1conv"],

  "0-1conv_tile_r1_wave"  : [ "trivnet_conv1",  "tfloat16-b1-h35-r1-s1-c1-m1-wmin-0.1-wmax0.12-imin-0.2-imax0.25", "1conv", "--scheduler wave"],
  "0-1conv_tile_r1h32"  : [ "trivnet_conv1",  "tfloat16-b1-h32-r1-s1-c1-m1-wmin-0.1-wmax0.12-imin-0.2-imax0.25", "1conv"],
  "0-1conv_tile_r1h32_wave"  : [ "trivnet_conv1",  "tfloat16-b1-h32-r1-s1-c1-m1-wmin-0.1-wmax0.12-imin-0.2-imax0.25", "1conv", "--scheduler wave"],
  "0-1conv_tile_r1_e1"       : [ "trivnet_conv1",  "tfloat16-b1-h35-r1-s1-c1-m1-F_31_31=3-wmin2-wmax2-imin-0-imax0", "1conv"],
  "0-1conv_tile_r1_e1_wave"  : [ "trivnet_conv1",  "tfloat16-b1-h35-r1-s1-c1-m1-F_31_31=3-wmin2-wmax2-imin-0-imax0", "1conv", "--scheduler wave"],
  "0-2conv3_relu" : [ "trivnet_lin",    "tfloat16-l2-b1-h4-r3-s1-c1-m1-relu-wmin-0.2-wmax0.24-imin-10000-imax10100", "1conv3"],
  "0-2conv3_32b"  : [ "trivnet_lin",    "tfloat32-l2-b1-h4-r3-s1-c1-m1-wmin-1000-wmax1010-imin-10000-imax10100", "2conv32b"],
  "3-rn50_relu_fp32_wave"  : [ "trivnet_lin","tfloat32-l2-b1-h224-r7-s2-c3-m3-relu-wmin-1-wmax1.1-imin-3-imax3.2", "2conv32b", "--scheduler wave"],
  "3-rn50_ba_relu_fp32_wave"  : [ "trivnet_conv_ba","tfloat32-b1-h224-r7-s2-c3-m64-SAME-relu-wmin-1-wmax1.1-imin-3-imax3.2-amin-3-amax3", "2conv32b", "--scheduler wave"],
  "0-ba_relu_fp32_wave"  : [ "trivnet_conv_ba","tfloat32-b1-h1-r1-s1-c1-m1-SAME-relu-wmin-2-wmax2-imin3-imax10-amin-7-amax7", "2conv32b", "--scheduler wave"],
  "0-30conv3"     : [ "trivnet_lin",    "tfloat16-l30-b1-h4-r3-s1-c1-m1-wmin-0.2-wmax0.22-imin-10000-imax10100", "30conv"],
  "0-10conv_relu" : [ "trivnet_lin",    "tfloat16-l10-b1-h4-r3-s1-c1-m1-relu-wmin-0.2-wmax0.4-imin-1000-imax1010", "10cr"],
  "0-10conv_relu_wave" : [ "trivnet_lin",    "tfloat16-l10-b1-h4-r3-s1-c1-m1-relu-wmin-0.2-wmax0.4-imin-1000-imax1010", "10cr", "--scheduler wave"],
  "0-116conv_tanh" : [ "trivnet_lin",   "tfloat16-l116-b1-h4-r3-s1-c1-m1-tanh-wmin-0.2-wmax0.8-imin-4-imax8", "116ct"],
  "0-116conv_tanh_wave" : [ "trivnet_lin",   "tfloat16-l116-b1-h4-r3-s1-c1-m1-tanh-wmin-0.2-wmax0.8-imin-4-imax8", "116ct", "--scheduler wave"],
  "0-2conv_pad2"  : [ "trivnet_conv1",  "tfloat16-b1-h4-r2-s1-c1-m1-wmin-0.1-wmax0.12-imin-0.2-imax0.22", "2conv_pad2"],
  "0-2conv_pad2as": [ "trivnet_conv1",  "tfloat16-b1-h2-r2-s1-c1-m1-wmin1-wmax4-imin5-imax8", "2conv_pad2as"],
  "0-2conv_pad3"  : [ "trivnet_conv1",  "tfloat16-b1-h4-r3-s1-c1-m1-wmin-0.1-wmax0.12-imin-0.2-imax0.22", "2conv_pad3"],
  "0-2conv_pad5"  : [ "trivnet_conv1",  "tfloat16-b1-h4-r5-s1-c1-m1-wmin-0.1-wmax0.12-imin-0.2-imax0.22", "2conv_pad5"],
  "0-1conv_s2"    : [ "trivnet_conv1",  "tfloat16-b1-h4-r1-s2-c1-m1-wmin-0.1-wmax0.12-imin-0.2-imax0.22", "1conv"],
  "0-1conv_s8"    : [ "trivnet_conv1",  "tfloat16-b1-h16-r1-s8-c1-m1-wmin2-wmax22-imin1-imax256", "1conv"],
  "0-1mp0"        : [ "trivnet_mp1",    "b1-h4-r1-s1-c1-m1-wmin0-wmax0.1-imin1-imax16", "1mp"],
  "0-1mp0c64"     : [ "trivnet_mp1",    "b1-h4-r1-s1-c64-m64-wmin0-wmax0.1-imin1-imax1024", "1mp"],
  "0-1mp_r3s2"    : [ "trivnet_mp1",    "b1-h5-r3-s2-c1-m1-wmin0-wmax0.1-imin1-imax25", "1mp"],
  "0-1mp_r3s2_112_55"  : [ "trivnet_mp1", "b1-h112-r3-s2-c1-m1-wmin0-wmax0.1-imin1-imax12544", "1mp"],
  "0-1ap0"        : [ "trivnet_ap1",    "b1-h4-r1-s1-c1-m1-wmin0-wmax0.1-imin1-imax16", "1ap"],
  "0-1ap7x7"      : [ "trivnet_ap1",    "b1-h7-r7-s7-c64-m64-wmin0-wmax0.1-imin1-imax784", "1ap"],
  "0-1conv1pool"       : [ "trivnet_conv_pool", "tfloat16-b1-h4-r1-s1-c1-m1-SAME-MaxPool-k2-d2-wmin2-wmax2.2-imin1-imax16", "1conv1pool"],
  "0-1conv1pool_wave"  : [ "trivnet_conv_pool", "tfloat16-b1-h4-r1-s1-c1-m128-SAME-MaxPool-k2-d2-wmin1-wmax1-imin0-imax127", "1conv1pool", "--scheduler wave"],
  "0-1conv1avgpool_wave"  : [ "trivnet_conv_pool", "tfloat16-b1-h4-r1-s1-c1-m1-SAME-AvgPool-k2-d2-wmin2-wmax2.2-imin1-imax16", "1conv1pool", "--scheduler wave"],
  "0-1conv1pool_b5_wave" : [ "trivnet_conv_pool", "tfloat16-b5-h4-r1-s1-c1-m1-SAME-AvgPool-k2-d2-wmin2-wmax2.2-imin1-imax80", "1conv1pool", "--scheduler wave"],
  "0-1conv1pool_b5m3_wave" : [ "trivnet_conv_pool", "tfloat16-b5-h4-r1-s1-c1-m3-SAME-AvgPool-k2-d2-wmin2-wmax2.2-imin1-imax240", "1conv1pool", "--scheduler wave"],
  "0-1conv1maxpool_k3d2_wave"  : [ "trivnet_conv_pool", "tfloat16-b1-h4-r1-s1-c1-m1-SAME-MaxPool-k3-d2-wmin2-wmax2.2-imin1-imax16", "1conv1pool", "--scheduler wave"],
  
  # Conv, BiasAdd
  "0-1conv1ba1_h4c1m1"       : [ "trivnet_conv_ba", "tfloat16-b1-h4-r1-s1-c1-m1-SAME-wmin2-wmax2.2-imin1-imax16-amin1-amax16", "1conv1ba"],
  "0-1conv1ba1_wave"  : [ "trivnet_conv_ba", "tfloat16-b1-h4-r1-s1-c1-m1-SAME-wmin2-wmax2.2-imin1-imax16-amin3-amax3.2", "1conv1ba", "--scheduler wave"],
  "0-1conv1ba1_h4c1m1_fp32"       : [ "trivnet_conv_ba", "tfloat32-b1-h4-r1-s1-c1-m1-SAME-wmin2-wmax2.2-imin1-imax16-amin1-amax16", "1conv1ba"],
  "0-1conv1ba1_h4c1m1_fp32_wave"  : [ "trivnet_conv_ba", "tfloat32-b1-h4-r1-s1-c1-m1-SAME-wmin2-wmax2.2-imin1-imax16-amin3-amax3.2", "1conv1ba", "--scheduler wave"],
  "0-1conv1ba1_h4c2m2_fp32_wave"  : [ "trivnet_conv_ba", "tfloat32-b1-h4-r1-s1-c2-m2-SAME-wmin2-wmax2.2-imin1-imax16-amin3-amax3.2", "1conv1ba", "--scheduler wave"],
  "0-1conv1ba1_h55c1m1_fp32_wave"  : [ "trivnet_conv_ba", "tfloat32-b1-h55-r1-s1-c1-m1-SAME-wmin2-wmax2.2-imin1-imax16-amin3-amax3.2", "1conv1ba", "--scheduler wave"],
  "0-1conv1ba1_h1c64m64_fp32_wave"  : [ "trivnet_conv_ba", "tfloat32-b1-h1-r1-s1-c64-m64-SAME-wmin2-wmax2.2-imin1-imax16-amin3-amax3.2", "1conv1ba", "--scheduler wave"],
  "0-1conv1ba1_h55c64m64_fp32_wave"  : [ "trivnet_conv_ba", "tfloat32-b1-h55-r1-s1-c64-m64-SAME-wmin2-wmax2.2-imin1-imax16-amin3-amax3.2", "1conv1ba", "--scheduler wave"],
  "0-1conv1ba1_h4c2m2_wave"  : [ "trivnet_conv_ba", "tfloat16-b1-h4-r1-s1-c2-m2-SAME-wmin2-wmax2.2-imin1-imax16-amin3-amax3.2", "1conv1ba", "--scheduler wave"],
  "0-1conv1ba1_h55c1m1_wave"  : [ "trivnet_conv_ba", "tfloat16-b1-h55-r1-s1-c1-m1-SAME-wmin2-wmax2.2-imin1-imax16-amin3-amax3.2", "1conv1ba", "--scheduler wave"],
  "0-1conv1ba1_h1c64m64_wave"  : [ "trivnet_conv_ba", "tfloat16-b1-h1-r1-s1-c64-m64-SAME-wmin2-wmax2.2-imin1-imax16-amin3-amax3.2", "1conv1ba", "--scheduler wave"],
  "0-1conv1ba1_h55c64m64_wave"  : [ "trivnet_conv_ba", "tfloat16-b1-h55-r1-s1-c64-m64-SAME-wmin2-wmax2.2-imin1-imax16-amin3-amax3.2", "1conv1ba", "--scheduler wave"],
  "0-1conv1ba1_fp32_wave"  : [ "trivnet_conv_ba", "tfloat32-b1-h224-r7-s2-c3-m64-SAME-wmin1-wmax1-imin0-imax50175-amin-20000-amax-20000", "1conv1ba", "--scheduler wave"],

  "0-biasadd"     : [ "trivnet_biasadd", "tfloat16-b1-h2-c1-wmin2-wmax2.2-imin3-imax6", "biasadd"],
  "0-biasadd_c2"  : [ "trivnet_biasadd", "tfloat16-b1-h2-c2-wmin2-wmax2.2-imin3-imax6", "biasadd"],
  "0-add"         : [ "trivnet_add",    "tfloat16-b1-h2-c1-wmin2-wmax2.2-imin3-imax6", "add"],
  "0-add_fp32"    : [ "trivnet_add",    "tfloat32-b1-h17-c4-wmin-0.1-wmax0.11-imin1-imax5", "add"],
  "0-add_uint8"   : [ "trivnet_add",    "tuint8-b1-h4-c3-wmin1-wmax4-imin5-imax53", "add"],
  "0-scaleadd_wave"       : [ "trivnet_scaleadd",    "tfloat16-b1-h1-c16-wmin2-wmax2.2-imin3-imax6", "scaleadd", "--scheduler wave"],
  "0-resadd_wave"         : [ "trivnet_add",    "tfloat16-b1-h2-c1-wmin2-wmax2.2-imin3-imax6", "add", "--scheduler wave"],
  "0-resadd_fp32_wave"    : [ "trivnet_add",    "tfloat32-b1-h17-c4-wmin-0.1-wmax0.11-imin1-imax5", "add", "--scheduler wave"],
  "0-resadd_uint8_wave"   : [ "trivnet_add",    "tuint8-b1-h4-c3-wmin1-wmax4-imin5-imax53", "add", "--scheduler wave"],
  "0-3conv_ba_resadd_fp32_wave"  : [ "trivnet_conv_ba_add", "tfloat32-b1-h112-r3-s1-c1-m1-SAME-wmin-1-wmax2-imin-0.1-imax0.3-amin-0.01-amax-0.03", "add", "--scheduler wave"],
  "0-3conv_ba_resadd_h1_fp32_wave"  : [ "trivnet_conv_ba_add", "tfloat32-b1-h1-r3-s1-c1-m1-SAME-wmin-1-wmax2-imin-0.1-imax0.3-amin-0.01-amax-0.03", "add", "--scheduler wave --debug 3"],
  "0-3conv_ba_resadd_h1_fp16_wave"  : [ "trivnet_conv_ba_add", "tfloat16-b1-h1-r3-s1-c1-m1-SAME-wmin-1-wmax2-imin-0.1-imax0.3-amin-0.01-amax-0.03", "add", "--scheduler wave --debug 3"],
  "0-3conv_ba_mult_fp32_wave"  : [ "trivnet_conv_ba_mult", "tfloat32-b1-h112-r3-s1-c1-m1-SAME-wmin-1-wmax2-imin-0.1-imax0.3-amin-0.01-amax-0.03", "mult", "--scheduler wave"],
  "0-2matmult_add_fp32_wave"  : [ "trivnet_matmul_add", "tfloat32-b1-h1-r1-s1-c512-m2048-SAME-wmin-1-wmax2-imin-0.1-imax0.3-amin-0.01-amax-0.03", "matmult", "--scheduler wave"],

  "0-1conv_s8_32b": [ "trivnet_lin",    "tfloat32-l2-b1-h16-r1-s8-c1-m1-wmin-0.1-wmax0.11-imin-0.2-imax0.21", "1conv32"],
  "1-1conv7_64"   : [ "trivnet_conv1",  "tfloat16-b1-h16-r7-s1-c64-m64-wmin-0.1-wmax0.11-imin-0.2-imax0.21", "1conv"],
  "1-1conv9_64"   : [ "trivnet_conv1",  "tfloat16-b1-h16-r9-s1-c64-m64-wmin-0.1-wmax0.11-imin-0.2-imax0.21", "1conv"],

  # Wave graph development tcc reference and tests
  "1-1conv0_wave"      : [ "trivnet_conv1",  "tfloat16-b1-h1-r1-s1-c33-m1-wmin-0.01-wmax0.011-imin-0.02-imax0.022", "1conv", "--scheduler wave"],
  "1-1conv0_c128"           : [ "trivnet_conv1",  "tfloat16-b1-h1-r1-s1-c128-m1-wmin2-wmax2.2-imin3-imax3.2", "1conv"],
  "1-1conv0_c128_wave"      : [ "trivnet_conv1",  "tfloat16-b1-h1-r1-s1-c128-m1-wmin-0.01-wmax0.011-imin-0.022-imax0.023", "1conv", "--scheduler wave"],
  "1-1conv0_c256_wave"      : [ "trivnet_conv1",  "tfloat16-b1-h1-r1-s1-c256-m1-wmin-0.01-wmax0.011-imin-0.022-imax0.023", "1conv", "--scheduler wave"],
  "1-1conv0_m64_wave"      : [ "trivnet_conv1",  "tfloat16-b1-h1-r1-s1-c1-m64-wmin1-wmax1.1-imin2-imax2.2", "1conv", "--scheduler wave"],
  "1-1conv0_m128_wave"      : [ "trivnet_conv1",  "tfloat16-b1-h1-r1-s1-c1-m128-wmin-0.01-wmax0.011-imin-0.02-imax0.022", "1conv", "--scheduler wave"],
  "1-1conv0_m2_wave"      : [ "trivnet_conv1",  "tfloat16-b1-h1-r1-s1-c1-m2-wmin-0.01-wmax0.011-imin-0.02-imax0.022", "1conv", "--scheduler wave"],
  "1-1conv0_h16c128"        : [ "trivnet_conv1",  "tfloat16-b1-h16-r1-s1-c128-m1-wmin2-wmax2.2-imin3-imax3.2", "1conv"],
  "1-1conv0_h16c128_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h16-r1-s1-c128-m1-wmin2-wmax2.2-imin3-imax3.2", "1conv", "--scheduler wave"],
  "1-1conv0_h16c256_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h16-r1-s1-c256-m1-wmin2-wmax2.2-imin3-imax3.2", "1conv", "--scheduler wave"],
  "1-1conv0_h16c1m1_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h16-r1-s1-c2-m1-wmin-0.2-wmax0.3-imin-0.1-imax0.2", "1conv", "--scheduler wave"],
  "1-1conv0_h16c2m1_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h16-r1-s1-c2-m1-wmin-0.2-wmax0.3-imin-0.1-imax0.2", "1conv", "--scheduler wave"],

  "1-1conv0_h16c256m128_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h16-r1-s1-c256-m128-wmin-1-wmax2-imin-1-imax3", "1conv", "--scheduler wave"],
  "1-1conv0_h16c256m128_fp32_wave"   : [ "trivnet_conv1",  "tfloat32-b1-h16-r1-s1-c256-m128-wmin-1-wmax2-imin-1-imax3", "1conv", "--scheduler wave"],

  "1-1conv0_h40c1m1_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h40-r1-s1-c1-m1-wmin2-wmax2.2-imin3-imax3.2", "1conv", "--scheduler wave"],
  "1-1conv0_h40c128m64_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h40-r1-s1-c128-m64-wmin2-wmax2.2-imin3-imax3.2", "1conv", "--scheduler wave"],
  "1-1conv0_h40c256m128_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h40-r1-s1-c256-m128-wmin2-wmax2.2-imin3-imax3.2", "1conv", "--scheduler wave"],
  "1-1conv0_h40c128m128_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h40-r1-s1-c128-m128-wmin2-wmax2.2-imin3-imax3.2", "1conv", "--scheduler wave"],
  "1-1conv0_h40c256m64_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h40-r1-s1-c256-m64-wmin2-wmax2.2-imin3-imax3.2", "1conv", "--scheduler wave"],

  "1-1conv0_h32c128m64_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h32-r1-s1-c128-m64-wmin2-wmax2.2-imin3-imax3.2", "1conv", "--scheduler wave"],
  "1-1conv0_h32c256m128_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h32-r1-s1-c256-m128-wmin2-wmax2.2-imin3-imax3.2", "1conv", "--scheduler wave"],
  "1-1conv0_h32c128m128_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h32-r1-s1-c128-m128-wmin2-wmax2.2-imin3-imax3.2", "1conv", "--scheduler wave"],
  "1-1conv0_h32c256m64_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h32-r1-s1-c256-m64-wmin2-wmax2.2-imin3-imax3.2", "1conv", "--scheduler wave"],

  "1-1conv0_h32c1m1_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h32-r1-s1-c1-m1-wmin0-wmax1-imin0-imax1023", "1conv", "--scheduler wave"],
  "1-1conv0_h32c2m1_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h32-r1-s1-c2-m1-wmin0-wmax1-imin0-imax1023", "1conv", "--scheduler wave"],
  "1-1conv0_h32c4m1_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h32-r1-s1-c4-m1-wmin0-wmax1-imin0-imax1023", "1conv", "--scheduler wave"],
  "1-1conv0_h32c8m1_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h32-r1-s1-c8-m1-wmin0-wmax1-imin0-imax1023", "1conv", "--scheduler wave"],
  "1-1conv0_h32c64m1_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h32-r1-s1-c64-m1-wmin2-wmax2.2-imin3-imax3.2", "1conv", "--scheduler wave"],
  "1-1conv0_h32c128m1_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h32-r1-s1-c128-m1-wmin2-wmax2.2-imin3-imax3.2", "1conv", "--scheduler wave"],
  "1-1conv0_h32c1m2_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h32-r1-s1-c1-m2-wmin2-wmax2.2-imin3-imax3.2", "1conv", "--scheduler wave"],
  "1-1conv0_h40c1m1_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h32-r1-s1-c1-m1-wmin2-wmax2.2-imin3-imax3.2", "1conv", "--scheduler wave"],
  "1-1conv0_h40c128m1_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h32-r1-s1-c128-m1-wmin2-wmax2.2-imin3-imax3.2", "1conv", "--scheduler wave"],

  "1-1conv0_h28c1m1_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h28-r1-s1-c1-m1-wmin2-wmax2.2-imin3-imax3.2", "1conv", "--scheduler wave"],
  "1-1conv0_h56c1m1_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h56-r1-s1-c1-m1-wmin-1-wmax2-imin-2-imax3", "1conv", "--scheduler wave"],
  "1-1conv0_h112c1m1_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h112-r1-s1-c1-m1-wmin2-wmax2.2-imin3-imax3.2", "1conv", "--scheduler wave"],
  "1-1conv0_h224c1m1_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h224-r1-s1-c1-m1-wmin2-wmax2.2-imin3-imax3.2", "1conv", "--scheduler wave"],

  "1-1conv0_h55c256_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h55-r1-s1-c256-m1-wmin-0.1-wmax0.11-imin-0.2-imax0.22", "1conv", "--scheduler wave"],
  "1-1conv0_h55c64m256_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h55-r1-s1-c64-m256-wmin-0.1-wmax0.11-imin-0.2-imax0.22", "1conv", "--scheduler wave"],
  "0-1conv0_h55c256m1_wave" : [ "trivnet_conv1",  "tfloat16-b1-h55-r1-s1-c256-m1-wmin2-wmax2.2-imin3-imax3.2", "1conv", "--scheduler wave"],
  "1-1conv0_h55m64_wave"   : [ "trivnet_conv1",  "tfloat16-b1-h55-r1-s1-c1-m64-wmin-0.1-wmax0.11-imin-0.2-imax0.22", "1conv", "--scheduler wave"],
  "1-1conv0_r3h16c128_wave" : [ "trivnet_conv1",  "tfloat16-b1-h16-r3-s1-c128-m1-wmin2-wmax2.2-imin3-imax3.2", "1conv", "--scheduler wave"],
  "1-1conv0_r3h55c256_wave" : [ "trivnet_conv1",  "tfloat16-b1-h55-r3-s1-c256-m1-wmin-0.1-wmax0.11-imin-0.2-imax0.22", "1conv", "--scheduler wave"],

  "2-1conv3_64s8" : [ "trivnet_conv1",  "tfloat16-b1-h16-r3-s8-c64-m64-wmin-0.1-wmax0.11-imin-0.2-imax0.22", "1conv"],
  "2-1conv9_64s8" : [ "trivnet_conv1",  "tfloat16-b1-h16-r9-s8-c64-m64-wmin-0.1-wmax0.11-imin-0.2-imax0.22", "1conv"],

  "2-padasym_strd_h3r2s2" : [ "trivnet_conv1", "tfloat16-b1-h3-r2-s2-c1-m1-wmin-0.1-wmax0.11-imin-0.2-imax0.22", "1conv"],
  "2-padasym_strd_h5r2s2" : [ "trivnet_conv1", "tfloat16-b1-h5-r2-s2-c1-m1-wmin-0.1-wmax0.11-imin-0.2-imax0.22", "1conv"],
  "2-padasym_strd_h4r2s3" : [ "trivnet_conv1", "tfloat16-b1-h4-r2-s3-c1-m1-wmin-0.1-wmax0.11-imin-0.2-imax0.22", "1conv"],
  "2-padasym_strd_h112r7s2" : [ "trivnet_conv1", "tfloat16-b1-h112-r7-s2-c1-m1-wmin-0.1-wmax0.11-imin-0.2-imax0.22", "1conv"],
  "2-padasym_strd_h112r7s2_wave" : [ "trivnet_conv1", "tfloat16-b1-h112-r7-s2-c1-m1-wmin-0.1-wmax0.11-imin-0.2-imax0.22", "1conv", "--scheduler wave"],
  "2-padasym_strd_h224r7s2_wave" : [ "trivnet_conv1", "tfloat16-b1-h224-r7-s2-c3-m64-wmin-0.1-wmax0.2-imin-0.2-imax0.3", "1conv", "--scheduler wave"],
  "2-padasym_strd_h224r7s2_fp32_wave" : [ "trivnet_conv1", "tfloat32-b1-h224-r7-s2-c3-m64-wmin-0.1-wmax0.2-imin-0.2-imax0.3", "1conv", "--scheduler wave"],

  "2-padsym_strd_h3r3s2" : [ "trivnet_conv1", "tfloat16-b1-h3-r3-s2-c1-m1-wmin-0.1-wmax0.11-imin-0.2-imax0.22", "1conv"],
  "2-padsym_strd_h5r3s2" : [ "trivnet_conv1", "tfloat16-b1-h5-r3-s2-c1-m1-wmin-0.1-wmax0.11-imin-0.2-imax0.22", "1conv"],
  "2-padsym_strd_h4r3s3" : [ "trivnet_conv1", "tfloat16-b1-h4-r3-s3-c1-m1-wmin-0.1-wmax0.11-imin-0.2-imax0.22", "1conv"],

  # Full c, m in resnet50 are 512, 2048
  "3-rn50_pool2"       : [ "trivnet_conv_pool", "tfloat16-b1-h7-r1-s1-c128-m64-SAME-AvgPool-k7-d7-PERM-wmin-0.1-wmax0.1-imin-1-imax2", "1conv1pool"],
  "3-rn50_pool2_wave"  : [ "trivnet_conv_pool", "tfloat16-b1-h7-r1-s1-c128-m64-SAME-AvgPool-k7-d7-PERM-wmin-0.1-wmax0.1-imin-1-imax2", "1conv1pool", "--scheduler wave"],

  # Sprint9 Story 1 milestone - all resnet50 float32 Conv2D layers as unit test
  # The 00 is just for testing the regression harness
  "3-rn50-t00_fp32_wave" : [ "trivnet_conv1",  "tfloat32-b1-h1-r1-s1-c1-m1-wmin-1-wmax1.1-imin-3-imax3.2",       "1conv", "--scheduler wave"],
  "3-rn50-01_fp32_wave" : [ "trivnet_conv1",  "tfloat32-b1-h14-r3-s1-c256-m256-wmin-1-wmax1.1-imin-3-imax3.2",  "1conv", "--scheduler wave"],
  "3-rn50-02_fp32_wave" : [ "trivnet_conv1",  "tfloat32-b1-h14-r1-s1-c256-m1024-wmin-1-wmax1.1-imin-3-imax3.2", "1conv", "--scheduler wave"],
  "3-rn50-03_fp32_wave" : [ "trivnet_conv1",  "tfloat32-b1-h14-r1-s1-c1024-m256-wmin-1-wmax1.1-imin-3-imax3.2", "1conv", "--scheduler wave"],
  "3-rn50-04_fp32_wave" : [ "trivnet_conv1",  "tfloat32-b1-h55-r1-s1-c64-m256-wmin-1-wmax1.1-imin-3-imax3.2",   "1conv", "--scheduler wave"],
  "3-rn50-05_fp32_wave" : [ "trivnet_conv1",  "tfloat32-b1-h28-r3-s1-c128-m128-wmin-1-wmax1.1-imin-3-imax3.2",  "1conv", "--scheduler wave"],
  "3-rn50-06_fp32_wave" : [ "trivnet_conv1",  "tfloat32-b1-h28-r1-s1-c128-m512-wmin-1-wmax1.1-imin-3-imax3.2",  "1conv", "--scheduler wave"],
  "3-rn50-07_fp32_wave" : [ "trivnet_conv1",  "tfloat32-b1-h55-r3-s1-c64-m64-wmin-1-wmax1.1-imin-3-imax3.2",    "1conv", "--scheduler wave"],
  "3-rn50-08_fp32_wave" : [ "trivnet_conv1",  "tfloat32-b1-h7-r3-s1-c512-m512-wmin-1-wmax1.1-imin-3-imax3.2",   "1conv", "--scheduler wave"], 
  "3-rn50-09_fp32_wave" : [ "trivnet_conv1",  "tfloat32-b1-h7-r1-s1-c512-m2048-wmin-1-wmax1.1-imin-3-imax3.2",  "1conv", "--scheduler wave"],
  "3-rn50-10_fp32_wave" : [ "trivnet_conv1",  "tfloat32-b1-h28-r1-s1-c512-m128-wmin-1-wmax1.1-imin-3-imax3.2",  "1conv", "--scheduler wave"],
  "3-rn50-11_fp32_wave" : [ "trivnet_conv1",  "tfloat32-b1-h55-r1-s1-c256-m64-wmin-1-wmax1.1-imin-3-imax3.2",   "1conv", "--scheduler wave"], 
  "3-rn50-12_fp32_wave" : [ "trivnet_conv1",  "tfloat32-b1-h7-r1-s1-c2048-m512-wmin-1-wmax1.1-imin-3-imax3.2",  "1conv", "--scheduler wave"], 
  "3-rn50-13_fp32_wave" : [ "trivnet_conv1",  "tfloat32-b1-h55-r1-s1-c64-m64-wmin-1-wmax1.1-imin-3-imax3.2",    "1conv", "--scheduler wave"], 
  "3-rn50-14_fp32_wave" : [ "trivnet_conv1",  "tfloat32-b1-h28-r1-s2-c512-m256-wmin-1-wmax1.1-imin-3-imax3.2",  "1conv", "--scheduler wave"],
  "3-rn50-15_fp32_wave" : [ "trivnet_conv1",  "tfloat32-b1-h28-r1-s2-c512-m1024-wmin-1-wmax1.1-imin-3-imax3.2", "1conv", "--scheduler wave"],
  "3-rn50-16_fp32_wave" : [ "trivnet_conv1",  "tfloat32-b1-h224-r7-s2-c3-m64-wmin-1-wmax1.1-imin-3-imax3.2",    "1conv", "--scheduler wave"],
  "3-rn50-17_fp32_wave" : [ "trivnet_conv1",  "tfloat32-b1-h55-r1-s2-c256-m512-wmin-1-wmax1.1-imin-3-imax3.2",  "1conv", "--scheduler wave"],
  "3-rn50-18_fp32_wave" : [ "trivnet_conv1",  "tfloat32-b1-h55-r1-s2-c256-m128-wmin-1-wmax1.1-imin-3-imax3.2",  "1conv", "--scheduler wave"],
  "3-rn50-19_fp32_wave" : [ "trivnet_conv1",  "tfloat32-b1-h14-r1-s2-c1024-m512-wmin-1-wmax1.1-imin-3-imax3.2",  "1conv", "--scheduler wave"],
  "3-rn50-20_fp32_wave" : [ "trivnet_conv1",  "tfloat32-b1-h14-r1-s2-c1024-m2048-wmin-1-wmax1.1-imin-3-imax3.2", "1conv", "--scheduler wave"],

  "3-rn50-t00_wave" : [ "trivnet_conv1",  "tfloat16-b1-h1-r1-s1-c1-m1-wmin-1-wmax1.1-imin-3-imax3.2",       "1conv", "--scheduler wave"],
  "3-rn50-01_wave" : [ "trivnet_conv1",  "tfloat16-b1-h14-r3-s1-c256-m256-wmin-1-wmax1.1-imin-3-imax3.2",  "1conv", "--scheduler wave"],
  "3-rn50-02_wave" : [ "trivnet_conv1",  "tfloat16-b1-h14-r1-s1-c256-m1024-wmin-1-wmax1.1-imin-3-imax3.2", "1conv", "--scheduler wave"],
  "3-rn50-03_wave" : [ "trivnet_conv1",  "tfloat16-b1-h14-r1-s1-c1024-m256-wmin-1-wmax1.1-imin-3-imax3.2", "1conv", "--scheduler wave"],
  "3-rn50-04_wave" : [ "trivnet_conv1",  "tfloat16-b1-h55-r1-s1-c64-m256-wmin-1-wmax1.1-imin-3-imax3.2",   "1conv", "--scheduler wave"],
  "3-rn50-05_wave" : [ "trivnet_conv1",  "tfloat16-b1-h28-r3-s1-c128-m128-wmin-1-wmax1.1-imin-3-imax3.2",  "1conv", "--scheduler wave"],
  "3-rn50-06_wave" : [ "trivnet_conv1",  "tfloat16-b1-h28-r1-s1-c128-m512-wmin-1-wmax1.1-imin-3-imax3.2",  "1conv", "--scheduler wave"],
  "3-rn50-07_wave" : [ "trivnet_conv1",  "tfloat16-b1-h55-r3-s1-c64-m64-wmin-1-wmax1.1-imin-3-imax3.2",    "1conv", "--scheduler wave"],
  "3-rn50-08_wave" : [ "trivnet_conv1",  "tfloat16-b1-h7-r3-s1-c512-m512-wmin-1-wmax1.1-imin-3-imax3.2",   "1conv", "--scheduler wave"], 
  "3-rn50-09_wave" : [ "trivnet_conv1",  "tfloat16-b1-h7-r1-s1-c512-m2048-wmin-1-wmax1.1-imin-3-imax3.2",  "1conv", "--scheduler wave"],
  "3-rn50-10_wave" : [ "trivnet_conv1",  "tfloat16-b1-h28-r1-s1-c512-m128-wmin-1-wmax1.1-imin-3-imax3.2",  "1conv", "--scheduler wave"],
  "3-rn50-11_wave" : [ "trivnet_conv1",  "tfloat16-b1-h55-r1-s1-c256-m64-wmin-1-wmax1.1-imin-3-imax3.2",   "1conv", "--scheduler wave"], 
  "3-rn50-12_wave" : [ "trivnet_conv1",  "tfloat16-b1-h7-r1-s1-c2048-m512-wmin-1-wmax1.1-imin-3-imax3.2",  "1conv", "--scheduler wave"], 
  "3-rn50-13_wave" : [ "trivnet_conv1",  "tfloat16-b1-h55-r1-s1-c64-m64-wmin-1-wmax1.1-imin-3-imax3.2",    "1conv", "--scheduler wave"], 
  "3-rn50-14_wave" : [ "trivnet_conv1",  "tfloat16-b1-h28-r1-s2-c512-m256-wmin-1-wmax1.1-imin-3-imax3.2",  "1conv", "--scheduler wave"],
  "3-rn50-15_wave" : [ "trivnet_conv1",  "tfloat16-b1-h28-r1-s2-c512-m1024-wmin-1-wmax1.1-imin-3-imax3.2", "1conv", "--scheduler wave"],
  "3-rn50-16_wave" : [ "trivnet_conv1",  "tfloat16-b1-h224-r7-s2-c3-m64-wmin-1-wmax1.1-imin-3-imax3.2",    "1conv", "--scheduler wave"],
  "3-rn50-17_wave" : [ "trivnet_conv1",  "tfloat16-b1-h55-r1-s2-c256-m512-wmin-1-wmax1.1-imin-3-imax3.2",  "1conv", "--scheduler wave"],
  "3-rn50-18_wave" : [ "trivnet_conv1",  "tfloat16-b1-h55-r1-s2-c256-m128-wmin-1-wmax1.1-imin-3-imax3.2",  "1conv", "--scheduler wave"],
  "3-rn50-19_wave" : [ "trivnet_conv1",  "tfloat16-b1-h14-r1-s2-c1024-m512-wmin-1-wmax1.1-imin-3-imax3.2",  "1conv", "--scheduler wave"],
  "3-rn50-20_wave" : [ "trivnet_conv1",  "tfloat16-b1-h14-r1-s2-c1024-m2048-wmin-1-wmax1.1-imin-3-imax3.2", "1conv", "--scheduler wave"],

  #"5-lstm_ptb"     : [ "tf_pb",          "lstm_ptb_word_lm/ptb_word_lm.pb",  "lstm_ptb", "--input_node Valid/ValidInput/StridedSlice ", "linspace1"],
  "6-alexnet"     : [ "tf_pb",          "alexnet_v100/alexnet_fp32.pb",  "alexnet", "--input_node Variable ", "linspace1"],
  "8-resnet50"                : [ "tf_pb",   "resnet50/resnet_v1_50_fp32_opt.pb",        "resnet50", " --depth 2", "linspace1"],
  "8-resnet50_fp32_keras"     : [ "tf_pb",   "resnet50_keras/resnet50_fp32_keras.pb",    "resnet50", "--input_node input_1  --depth 2", "linspace1"],
  "8-resnet50_fp32_keras_opt" : [ "tf_pb",   "resnet50_keras/resnet50_fp32_keras_opt.pb","resnet50", "--input_node input_1  --depth 2", "linspace1"],
  "8-resnet50_fp16_keras"     : [ "tf_pb",   "resnet50_keras/resnet50_fp16_keras.pb",    "resnet50", "--input_node input_1  --depth 2", "linspace1"],
  "8-resnet50_fp16_keras_opt_b16" : [ "tf_pb",   "resnet50_keras/resnet50_fp16_keras_opt.pb","resnet50", "--input_node input_1  --depth 2 --batch 16", "linspace1"],
  "8-resnet50_fp16_keras_opt" : [ "tf_pb",   "resnet50_keras/resnet50_fp16_keras_opt.pb","resnet50", "--input_node input_1  --depth 2", "linspace1"],
  "8-resnet50_fp16_wave"      : [ "tf_pb",   "resnet50_keras/resnet50_fp16_keras_opt.pb","resnet50", "--input_node input_1  --depth 2 --scheduler wave", "linspace1"],
  "8-resnet50_fp32_wave"      : [ "tf_pb",   "resnet50_keras/resnet50_fp32_keras_opt.pb","resnet50", "--input_node input_1  --depth 2 --scheduler wave", "linspace1"],
  "8-resnet50_fp16_wave_b2"   : [ "tf_pb",   "resnet50_keras/resnet50_fp16_keras_opt.pb","resnet50", "--input_node input_1  --depth 2 --scheduler wave --batch 2", "linspace1"],
  "9-resnet152"               : [ "tf_pb",   "resnet_v2_152/pb/resnet_v2_152_fp32.pb",   "resnet152", " --depth 2", "linspace1"],
  "9-resnet152_waveopt"       : [ "tf_pb",   "resnet_v2_152/pb/resnet_v2_152_fp32.pb",   "resnet152", "--partition from resnet_v2_152/conv1/convolution resnet_v2_152/postnorm/batchnorm/mul_1 --executors host all waveopt 1  --depth 2 --scheduler wave --images %s" % rnDogJpg, "--input_files %s" % rnDogJpg],
  
  # Subgraph partioned flow using neural network executor
  "0-add_nne"      : [ "trivnet_add",    "tfloat16-b1-h2-c1-wmin2-wmax2.2-imin3-imax6", "add", "--partition auto --executors host all tcc 1 --width 2 --debug 1"],
  "0-4conv_relu_nne" : [ "trivnet_lin",    "tfloat16-l3-b1-h4-r3-s1-c1-m1-relu-wmin-0.2-wmax0.4-imin-1-imax2", "4conv_nne", "--partition conv --executors tcc 1 3 host 0 2 4 --debug 1 --scheduler wave"],


  # Resnet
  "8-rn50_nne_auto" : [ "tf_pb",   "resnet50_keras/resnet50_fp16_keras_opt.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 --preprocessor %s --postprocessor %s --partition auto --executors wave all  --scheduler wave --images %s" %(rnPreFp16, rnPost, rnDogJpg) ,"--input_files %s" % rnDogJpg ],
  "8-rn50_nne_fp32_meauto" : [ "tf_pb",   "resnet50_keras/resnet50_fp32_keras_opt.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 --preprocessor %s --postprocessor %s --partition meauto --executors wave all host 17  --scheduler wave --images %s" %(rnPreFp32, rnPost, rnDogJpg) ,"--input_files %s" % rnDogJpg ],
  #"8-rn50_nne_conv" : [ "tf_pb",   "resnet50_keras/resnet50_fp16_keras_opt.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 --preprocessor %s --postprocessor %s --partition conv --executors tcc 2 6 8 13 15 20 22 host 0 --images %s" %(rnPreFp16, rnPost, rnDogJpg), "linspace1"],
  "4-rn50_nne_fc"   : [ "tf_pb",   "resnet50_keras/resnet50_fp16_keras_opt.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 --preprocessor %s --postprocessor %s --partition from avg_pool/AvgPool --executors host 0 host 1 --images %s" %(rnPreFp16, rnPost, rnDogJpg) ,"--input_files %s" % rnDogJpg ],
  "4-rn50_matmul_fp32_wave"   : [ "tf_pb",   "resnet50_keras/resnet50_fp32_keras_opt.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 --preprocessor %s --postprocessor %s --partition from avg_pool/AvgPool --executors host 0 wave 1 --images %s" %(rnPreFp32, rnPost, rnDogJpg) ,"--input_files %s" % rnDogJpg ],
  "4-rn50_nne_from1_fp32_wave"   : [ "tf_pb",   "resnet50_keras/resnet50_fp32_keras_opt.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 --preprocessor %s --postprocessor %s --partition from bn_conv1/batchnorm_1/add_1   --executors wave 0 host 1  --scheduler wave --images %s" %(rnPreFp32, rnPost, rnDogJpg) ,"--input_files %s" % rnDogJpg ],
  "4-rn50_nne_from2_fp32_wave"   : [ "tf_pb",   "resnet50_keras/resnet50_fp32_keras_opt.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 --preprocessor %s --postprocessor %s --partition from activation_1/Relu   --executors wave 0 host 1  --scheduler wave --images %s" %(rnPreFp32, rnPost, rnDogJpg) ,"--input_files %s" % rnDogJpg ],
  "4-rn50_nne_from3_fp32_wave"   : [ "tf_pb",   "resnet50_keras/resnet50_fp32_keras_opt.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 --preprocessor %s --postprocessor %s --partition from max_pooling2d_1/MaxPool   --executors wave 0 host 1  --scheduler wave --images %s" %(rnPreFp32, rnPost, rnDogJpg) ,"--input_files %s" % rnDogJpg ],
  "4-rn50_nne_from1_wave"   : [ "tf_pb",   "resnet50_keras/resnet50_fp16_keras_opt.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 --preprocessor %s --postprocessor %s --partition from bn_conv1/batchnorm_1/add_1   --executors wave 0 host 1  --scheduler wave --images %s" %(rnPreFp16, rnPost, rnDogJpg) ,"--input_files %s" % rnDogJpg ],
  "4-rn50_nne_from2_wave"   : [ "tf_pb",   "resnet50_keras/resnet50_fp16_keras_opt.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 --preprocessor %s --postprocessor %s --partition from activation_1/Relu   --executors wave 0 host 1  --scheduler wave --images %s" %(rnPreFp16, rnPost, rnDogJpg) ,"--input_files %s" % rnDogJpg ],
  "4-rn50_nne_from3_wave"   : [ "tf_pb",   "resnet50_keras/resnet50_fp16_keras_opt.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 --preprocessor %s --postprocessor %s --partition from max_pooling2d_1/MaxPool   --executors wave 0 host 1  --scheduler wave --images %s" %(rnPreFp16, rnPost, rnDogJpg) ,"--input_files %s" % rnDogJpg ],
  "5-rn50_nne_to_act4_fp32_wave"   : [ "tf_pb",   "resnet50_keras/resnet50_fp32_keras_opt.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 --preprocessor %s --postprocessor %s --partition from activation_4/Relu   --executors wave 0 host 1  --scheduler wave --images %s" %(rnPreFp32, rnPost, rnDogJpg) ,"--input_files %s" % rnDogJpg ],
  "5-rn50_nne_to_act4_wave"   : [ "tf_pb",   "resnet50_keras/resnet50_fp16_keras_opt.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 --preprocessor %s --postprocessor %s --partition from activation_4/Relu   --executors wave 0 host 1  --scheduler wave --images %s" %(rnPreFp16, rnPost, rnDogJpg) ,"--input_files %s" % rnDogJpg ],
  "7-rn50_nne_fc_waveopt"     : [ "tf_pb",   "resnet50_keras/resnet50_fp16_keras_opt.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 --preprocessor %s --postprocessor %s --partition from fc1000/Softmax --executors waveopt 0 host 1  --scheduler wave --images %s" %(rnPreFp16, rnPost, rnDogJpg) ,"--input_files %s" % rnDogJpg ],
  "7-rn50_nne_fp32_waveopt"   : [ "tf_pb",   "resnet50_keras/resnet50_fp32_keras_opt.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 --preprocessor %s --postprocessor %s --partition from fc1000/Softmax --executors waveopt 0 host 1  --scheduler wave --images %s" %(rnPreFp32, rnPost, rnDogJpg) ,"--input_files %s" % rnDogJpg ],
  "7-rn50_nne_fp32_wave"   : [ "tf_pb",   "resnet50_keras/resnet50_fp32_keras_opt.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 --preprocessor %s --postprocessor %s --partition from fc1000/Softmax --executors wave 0 host 1  --scheduler wave --images %s" %(rnPreFp32, rnPost, rnDogJpg) ,"--input_files %s" % rnDogJpg ],
  "7-rn50_nne_fp16_wave"   : [ "tf_pb",   "resnet50_keras/resnet50_fp16_keras_opt.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 --preprocessor %s --postprocessor %s --partition from fc1000/Softmax --executors wave 0 host 1  --scheduler wave --images %s" %(rnPreFp16, rnPost, rnDogJpg) ,"--input_files %s" % rnDogJpg ],
  "7-rn50_nne_fp16_accfail"   : [ "tf_pb",   "resnet50_keras/resnet50_fp16_keras_opt.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 --preprocessor %s --postprocessor %s --partition from fc1000/Softmax --executors wave 0 host 1  --scheduler wave --images %s" % (rnPreFp16, rnPost, fp16AccJpg) , "--input_files %s" % fp16AccJpg],
  "7-rn50_nne_fp16_host"   : [ "tf_pb",   "resnet50_keras/resnet50_fp16_keras_opt.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 --preprocessor %s --postprocessor %s --partition from fc1000/Softmax --executors host all --images %s" % (rnPreFp16, rnPost, rnDogJpg) , "--input_files %s" % rnDogJpg],
  "7-rn50_nne_fc_wave"     : [ "tf_pb",   "resnet50_keras/resnet50_fp16_keras_opt.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 --preprocessor %s --postprocessor %s --partition from avg_pool/AvgPool --executors wave 0 host 1  --scheduler wave --images %s" %(rnPreFp16, rnPost, rnDogJpg) ,"--input_files %s" % rnDogJpg ],
  "7-rn50_nne_conv_wave" : [ "tf_pb",   "resnet50_keras/resnet50_fp32_keras_opt.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 --preprocessor %s --postprocessor %s --partition conv --executors host all wave 01 03 04 07 09 12 14 16 19 21 23 26 27 30 32 35 37 39 42 44 46 49 51 53 56 57 60 62 65 67 69 72 74 76 79 81 83 86 88 90 93 95 97 100 101 104 106 109 111 113 116 118 120  --scheduler wave --images %s" %(rnPreFp32, rnPost, rnDogJpg) ,"--input_files %s" % rnDogJpg ],
  "7-rn50_nne_fp32_host"   : [ "tf_pb",   "resnet50_keras/resnet50_fp32_keras_opt.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 --preprocessor %s --postprocessor %s   --partition from fc1000/Softmax --executors host all --images %s" %(rnPreFp32, rnPost, rnDogJpg),"--input_files %s" % rnDogJpg ],


  # Matmult
  "4-rn50_matmul_fp32_wave"   : [ "tf_pb",   "resnet50_keras/resnet50_fp32_keras_opt.pb","resnet50", " --input_node input_1  --depth 2  --debug 1 --preprocessor %s --postprocessor %s  --partition from avg_pool/AvgPool --executors host 0 wave 1 --scheduler wave --images %s "% (rnPreFp32, rnPost, rnDogJpg),"--input_files %s" % rnDogJpg ],
  "4-rn50_matmul_fp32_waveopt"   : [ "tf_pb",   "resnet50_keras/resnet50_fp32_keras_opt.pb","resnet50", " --input_node input_1  --depth 2  --debug 1 --preprocessor %s --postprocessor %s  --partition from avg_pool/AvgPool --executors host 0 waveopt 1 --scheduler wave --images %s"% (rnPreFp32, rnPost, rnDogJpg),"--input_files %s" % rnDogJpg ],
  "4-rn50_matmul_nosm_fp32_wave"   : [ "tf_pb",   "resnet50_keras/resnet50_fp32_keras_opt.pb","resnet50", " --input_node input_1  --depth 2  --debug 1 --preprocessor %s --postprocessor %s  --partition from avg_pool/AvgPool fc1000/Softmax --executors host 0 2 wave 1 --scheduler wave --images %s" %(rnPreFp32, rnPost, rnDogJpg),"--input_files %s" % rnDogJpg ],

  # Resnet50 barching
  "7-rn50_nne_fp16_waveopt_b4"   : [ "tf_pb",   "resnet50_keras/resnet50_fp16_keras_opt.pb","resnet50", "--input_node input_1  --depth 2  --debug 1 --partition from fc1000/Softmax --executors waveopt 0 host 1  --scheduler wave --batch 4 --images linspace1", ],

  # LSTM
  "4-ptb_word_lm1_host"      : [ "tf_pb",   "ptb_word_lm/keras_unrolled/model-b16s32h512.pb","lm", " --input_node embedding_1_input_1  --depth 3  --debug 0   --partition from  lstm_2_1/transpose_1  --executors host all --scheduler wave --input_constants dropout_1/keras_learning_phase:0 False --exclude_ops_from_capture ^dropout_1_1/cond/ --images %s" % lstmD0B32, "--input_files %s" % lstmD0B32],
  "4-ptb_word_lm1"   : [ "tf_pb",   "ptb_word_lm/keras_unrolled/model-b16s32h512.pb", "lm", " --input_node input_1  --depth 3  --debug 0 --preprocessor %s --postprocessor %s  --partition from avg_pool/AvgPool --executors host 0 wave 1 --scheduler wave --images %s"% (rnPreFp32, rnPost, rnDogJpg), "--input_files %s" % rnDogJpg ],
  "4-ptb_word_small1_host"   : [ "tf_pb",   "ptb_word_lm/keras_unrolled/model-b32s4h512.pb","lm", " --input_node embedding_1_input_1  --depth 3  --debug 0   --partition from  lstm_2_1/transpose_1  --executors host all --scheduler wave --input_constants dropout_1/keras_learning_phase:0 False --exclude_ops_from_capture ^dropout_1_1/cond/ --images %s" % lstmD0B4, "--input_files %s" % lstmD0B4],
  "4-ptb_word_small1_wave"   : [ "tf_pb",   "ptb_word_lm/keras_unrolled/model-b32s4h512.pb","lm", " --input_node embedding_1_input_1  --depth 3  --debug 0   --partition from_multi  lstm_1_1/unstack,lstm_1_1/Tile_1  lstm_2_1/transpose_1  --executors host 0 2 wave 1 --scheduler wave --input_constants dropout_1/keras_learning_phase:0 False --exclude_ops_from_capture ^dropout_1_1/cond/ --images %s" % lstmD0B4, "--input_files %s" % lstmD0B4],
  "4-ptb_word_small_sigmoid_wave"   : [ "tf_pb",   "ptb_word_lm/keras_unrolled/sigmoid/model-b32s4h512.pb","lm", " --input_node embedding_1_input_1  --depth 3  --debug 1   --partition from_multi  lstm_1_1/unstack,lstm_1_1/Tile_1,lstm_1_1/Tile,lstm_1_1/Tile_1  lstm_1_1/stack  lstm_2_1/transpose_1  --adjust_node_color lstm_1_1/Tile 0 lstm_1_1/Tile_1 0  --executors host 0 2 wave 1 --scheduler wave --input_constants dropout_1/keras_learning_phase:0 False  --exclude_ops_from_capture ^dropout_1_1/cond/ --images %s" % lstmD0B4, "--input_files %s" % lstmD0B4],
  
  # LSTM small: 5 color 2-layer small host-tpb-host-tpb-host - waveopt and wave versions
  "4-ptb_word_small_sigmoid_2l_waveopt"  : [ "tf_pb",   "ptb_word_lm/keras_unrolled/sigmoid/model-b32s4h512.pb","lm", " --show_op_name_in_kgraph --input_node embedding_1_input_1  --depth 3  --debug 1   --partition from_multi  lstm_1_1/unstack,lstm_1_1/Tile_1,lstm_1_1/Tile,lstm_1_1/Tile_1  lstm_1_1/stack   lstm_2_1/unstack,lstm_2_1/Tile_1,lstm_2_1/Tile,lstm_2_1/Tile_1  lstm_2_1/stack     --adjust_node_color  lstm_1_1/Tile 0 lstm_1_1/Tile_1 0 lstm_2_1/Tile 2 lstm_2_1/Tile_1 2   --executors  waveopt 1 3  --scheduler wave --input_constants dropout_1/keras_learning_phase:0 False  --exclude_ops_from_capture ^dropout_1_1/cond/ --images %s" % lstmD0B4, "--input_files %s" % lstmD0B4],
  "4-ptb_word_small_sigmoid_2l_wave"  : [ "tf_pb",   "ptb_word_lm/keras_unrolled/sigmoid/model-b32s4h512.pb","lm", " --show_op_name_in_kgraph --input_node embedding_1_input_1  --depth 3  --debug 1   --partition from_multi  lstm_1_1/unstack,lstm_1_1/Tile_1,lstm_1_1/Tile,lstm_1_1/Tile_1  lstm_1_1/stack   lstm_2_1/unstack,lstm_2_1/Tile_1,lstm_2_1/Tile,lstm_2_1/Tile_1  lstm_2_1/stack     --adjust_node_color  lstm_1_1/Tile 0 lstm_1_1/Tile_1 0 lstm_2_1/Tile 2 lstm_2_1/Tile_1 2   --executors  wave 1 3  --scheduler wave --input_constants dropout_1/keras_learning_phase:0 False  --exclude_ops_from_capture ^dropout_1_1/cond/ --images %s" % lstmD0B4, "--input_files %s" % lstmD0B4],
 
   # LSTM debug of matmult, only sg00 is usable
  "2-ptb_word_unstack_matmul4"  : [ "tf_pb",   "ptb_word_lm/keras_unrolled/sigmoid/model-b32s4h512.pb","lm", " --show_op_name_in_kgraph --input_node embedding_1_input_1  --depth 3  --debug 1   --partition from_multi  lstm_1_1/unstack,lstm_1_1/Tile_1,lstm_1_1/Tile,lstm_1_1/Tile_1   lstm_1_1/add_6,lstm_1_1/add_4,lstm_1_1/add_2,lstm_1_1/MatMul,lstm_1_1/MatMul_1,lstm_1_1/mul,lstm_1_1/mul_1  --adjust_node_color lstm_1_1/Tile 0 lstm_1_1/Tile_1 0  --executors wave 0 --scheduler wave --input_constants dropout_1/keras_learning_phase:0 False  --exclude_ops_from_capture ^dropout_1_1/cond/ --images %s" % lstmD0B4, "--input_files %s" % lstmD0B4],
  
   # LSTM debug of matmult, only sg00 is usable
  "2-ptb_word_unstack_matmul1"  : [ "tf_pb",   "ptb_word_lm/keras_unrolled/sigmoid/model-b32s4h512.pb","lm", " --show_op_name_in_kgraph --input_node embedding_1_input_1  --depth 3  --debug 1   --partition from_multi  lstm_1_1/unstack,lstm_1_1/Tile_1,lstm_1_1/Tile,lstm_1_1/Tile_1   lstm_1_1/add_6,lstm_1_1/add_4,lstm_1_1/add_2,lstm_1_1/MatMul,lstm_1_1/MatMul_1,lstm_1_1/mul,lstm_1_1/mul_1  --adjust_node_color lstm_1_1/Tile 0 lstm_1_1/Tile_1 0 lstm_1_1/MatMul_2 0  lstm_1_1/MatMul_4 0  --executors wave 0 --scheduler wave --input_constants dropout_1/keras_learning_phase:0 False  --exclude_ops_from_capture ^dropout_1_1/cond/ --images %s" % lstmD0B4, "--input_files %s" % lstmD0B4],
  
}

# Regression waiver mechanism 
# If the testname matches the regexp then the FAIL status is replaced with
# with the string
testWaiver = [
    ['0-1conv1maxpool_k3d2_wave',   'WAIVE_WAVESC'],
    ['0-1conv1pool_b5_wave',        'WAIVE_WAVESC'],
    ['0-1conv1pool_b5m3_wave',      'WAIVE_WAVESC'],

    ['^0-add$',   'WAIVE-RESADD'],
    ['^0-add_fp32',   'WAIVE-RESADD'],
    ['^0-add_uint8',   'WAIVE-RESADD'],

    # ME does not support resadd as the first op which is ok
    ['^0-resadd.*_wave$', 'WAIVE-RESADD_ME'],

    ['1-1conv0_r3h55c256_wave',     'WAIVE_WAVESC'],

    ['4-rn50_matmul_fp32_wave$',      'WAIVE-S10_BE_SOFTMAX'],

    ['^[6]-alexnet',  'WAIVE-BENCH'],
    #['7-rn50_nne_fc_wave$', 'WAIVE-WAVESC'],

    ['^[8]-resnet152',  'WAIVE-BENCH'],
    ['^[8]-resnet50',  'WAIVE-BENCH'],
    ['8-rn50_nne_auto', 'WAIVE-NNE'],
    
    # ME accuracy failure
    ['0-116conv_tanh_wave', 'WAIVE-ME_ACC'],
    
    # LSMT
    ['4-ptb_word_lm1$', 'WAIVE-LSTM'],
    ['4-ptb_word_small1_wave$', 'WAIVE-LSTM'],
    ['4-ptb_word_small_sigmoid_wave$', 'WAIVE-LSTM'],
    ['0-scaleadd_wave',             'WAIVE-LSTM'],
    ['4-ptb_word_small_sigmoid_2l_wave$',             'WAIVE-LSTM'],
    ['2-ptb_word_unstack_.*',             'WAIVE-LSTM'],

    # accuracy fail, fp16
    ['7-rn50_nne_fp16_accfail$', 'WAIVE_FP16_ACC'],

    # batching
    ['7-rn50_nne_fp16_waveopt_b4$', 'WAIVE_BATCH'],

    # Resnet 152
    ['^9-resnet152', 'WAIVE_RN152'],

  ]
