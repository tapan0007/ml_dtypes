# Copyright (C) 2018, Amazon.com. All Rights Reserved
#
# Kaena test for resnet50 residual units
#

kccOpts = ("--show_op_name_in_kgraph --partition none "
          + " --executors wave all  --scheduler wave2 --images linspace1")
rtOpts = "--input_files %s --check_against_ref all_available"

testConfigMap = {
  "4-rn50_res_unit_input_act4" : [
      "trivnet_opt_inf", "resnet50_keras/resnet50_fp16_keras_opt2.pb--input_1--activation_4/Relu--float16--1,224,224,3",
      "resnet50", kccOpts, rtOpts % 'trivnet_input_1:0.npy'],

  "4-rn50_res_unit_act4_7" : [
      "trivnet_opt_inf", "resnet50_keras/resnet50_fp16_keras_opt2.pb--activation_4/Relu--activation_7/Relu--float16--1,55,55,256",
      "resnet50", kccOpts, rtOpts % 'trivnet_activation_4__Relu:0.npy'],

  "4-rn50_res_unit_act7_10" : [
      "trivnet_opt_inf", "resnet50_keras/resnet50_fp16_keras_opt2.pb--activation_7/Relu--activation_10/Relu--float16--1,55,55,256",
      "resnet50", kccOpts, rtOpts % 'trivnet_activation_7__Relu:0.npy' ],

  "4-rn50_res_unit_act10_13" : [
      "trivnet_opt_inf", "resnet50_keras/resnet50_fp16_keras_opt2.pb--activation_10/Relu--activation_13/Relu--float16--1,55,55,256",
      "resnet50", kccOpts, rtOpts % 'trivnet_activation_10__Relu:0.npy' ],

  "4-rn50_res_unit_act13_16" : [
      "trivnet_opt_inf", "resnet50_keras/resnet50_fp16_keras_opt2.pb--activation_13/Relu--activation_16/Relu--float16--1,28,28,512",
      "resnet50", kccOpts, rtOpts % 'trivnet_activation_13__Relu:0.npy' ],

  "4-rn50_res_unit_act16_19" : [
      "trivnet_opt_inf", "resnet50_keras/resnet50_fp16_keras_opt2.pb--activation_16/Relu--activation_19/Relu--float16--1,28,28,512",
      "resnet50", kccOpts, rtOpts % 'trivnet_activation_16__Relu:0.npy' ],

  "4-rn50_res_unit_act19_22" : [
      "trivnet_opt_inf", "resnet50_keras/resnet50_fp16_keras_opt2.pb--activation_19/Relu--activation_22/Relu--float16--1,28,28,512",
      "resnet50", kccOpts, rtOpts % 'trivnet_activation_19__Relu:0.npy' ],

  "4-rn50_res_unit_act22_25" : [
      "trivnet_opt_inf", "resnet50_keras/resnet50_fp16_keras_opt2.pb--activation_22/Relu--activation_25/Relu--float16--1,28,28,512",
      "resnet50", kccOpts, rtOpts % 'trivnet_activation_22__Relu:0.npy' ],

  "4-rn50_res_unit_act25_25" : [
      "trivnet_opt_inf", "resnet50_keras/resnet50_fp16_keras_opt2.pb--activation_25/Relu--activation_28/Relu--float16--1,14,14,1024",
      "resnet50", kccOpts, rtOpts % 'trivnet_activation_25__Relu:0.npy' ],

  "4-rn50_res_unit_act28_31" : [
      "trivnet_opt_inf", "resnet50_keras/resnet50_fp16_keras_opt2.pb--activation_28/Relu--activation_31/Relu--float16--1,14,14,1024",
      "resnet50", kccOpts, rtOpts % 'trivnet_activation_28__Relu:0.npy' ],

  "4-rn50_res_unit_act31_34" : [
      "trivnet_opt_inf", "resnet50_keras/resnet50_fp16_keras_opt2.pb--activation_31/Relu--activation_34/Relu--float16--1,14,14,1024",
      "resnet50", kccOpts, rtOpts % 'trivnet_activation_31__Relu:0.npy' ],

  "4-rn50_res_unit_act34_37" : [
      "trivnet_opt_inf", "resnet50_keras/resnet50_fp16_keras_opt2.pb--activation_34/Relu--activation_37/Relu--float16--1,14,14,1024",
      "resnet50", kccOpts, rtOpts % 'trivnet_activation_34__Relu:0.npy' ],

  "4-rn50_res_unit_act37_40" : [
      "trivnet_opt_inf", "resnet50_keras/resnet50_fp16_keras_opt2.pb--activation_37/Relu--activation_40/Relu--float16--1,14,14,1024",
      "resnet50", kccOpts, rtOpts % 'trivnet_activation_37__Relu:0.npy' ],

  "4-rn50_res_unit_act40_43" : [
      "trivnet_opt_inf", "resnet50_keras/resnet50_fp16_keras_opt2.pb--activation_40/Relu--activation_43/Relu--float16--1,14,14,1024",
      "resnet50", kccOpts, rtOpts % 'trivnet_activation_40__Relu:0.npy' ],

  "4-rn50_res_unit_act43_46" : [
      "trivnet_opt_inf", "resnet50_keras/resnet50_fp16_keras_opt2.pb--activation_43/Relu--activation_46/Relu--float16--1,7,7,2048",
      "resnet50", kccOpts, rtOpts % 'trivnet_activation_43__Relu:0.npy' ],

  "4-rn50_res_unit_act46_49" : [
      "trivnet_opt_inf", "resnet50_keras/resnet50_fp16_keras_opt2.pb--activation_46/Relu--activation_49/Relu--float16--1,7,7,2048",
      "resnet50", kccOpts, rtOpts % 'trivnet_activation_46__Relu:0.npy' ],

  "4-rn50_res_unit_act49_sm" : [
      "trivnet_opt_inf", "resnet50_keras/resnet50_fp16_keras_opt2.pb--activation_46/Relu--fc1000/Softmax--float16--1,7,7,2048",
      "resnet50", kccOpts, rtOpts % 'trivnet_activation_49__Relu:0.npy' ],


}


# Regression waiver mechanism
testWaiver = [
  ['4-rn50_res_unit_act49_sm', 'WAIVE_SOFTMAX'],
]

noGpuTestWaiver = [
]

qemuTestWaiver = [
]

