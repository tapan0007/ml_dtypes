# Copyright (C) 2018, Amazon.com. All Rights Reserved
#
# Kaena test for resnet50 residual units
#

kccOpts = ("--show_op_name_in_kgraph --partition none "
          + " --executors wave all  --scheduler wave2 --images linspace1")
rtOpts = "--input_files %s --check_against_ref all_available"

testConfigMap = {
  
  #### Smaller slices of 4-rn50_res_unit_input_act4 for debugging ####
  "4-rn50_res_unit_input_act1" : [
      "trivnet_opt_inf", "resnet50_keras/resnet50_fp16_keras_opt2.pb--input_1--activation_1/Relu--float16--1,224,224,3",
      "resnet50", kccOpts, rtOpts % 'trivnet_input_1:0.npy'],
  "4-rn50_res_unit_input_mp" : [
      "trivnet_opt_inf", "resnet50_keras/resnet50_fp16_keras_opt2.pb--input_1--max_pooling2d_1/MaxPool--float16--1,224,224,3",
      "resnet50", kccOpts, rtOpts % 'trivnet_input_1:0.npy'],
  "4-rn50_res_unit_input_resadd2a" : [
      "trivnet_opt_inf", "resnet50_keras/resnet50_fp16_keras_opt2.pb--input_1--bn2a_branch1/batchnorm_1/add_1--float16--1,224,224,3",
      "resnet50", kccOpts, rtOpts % 'trivnet_input_1:0.npy'],
  "4-rn50_res_unit_input_act2" : [
      "trivnet_opt_inf", "resnet50_keras/resnet50_fp16_keras_opt2.pb--input_1--activation_2/Relu--float16--1,224,224,3",
      "resnet50", kccOpts, rtOpts % 'trivnet_input_1:0.npy'],
  "4-rn50_res_unit_input_act3" : [
      "trivnet_opt_inf", "resnet50_keras/resnet50_fp16_keras_opt2.pb--input_1--activation_3/Relu--float16--1,224,224,3",
      "resnet50", kccOpts, rtOpts % 'trivnet_input_1:0.npy'],
  
  "4-rn50_res_unit_input_act4" : [
      "trivnet_opt_inf", "resnet50_keras/resnet50_fp16_keras_opt2.pb--input_1--activation_4/Relu--float16--1,224,224,3",
      "resnet50", kccOpts, rtOpts % 'trivnet_input_1:0.npy'],

  "4-rn50_res_unit_act4_7" : [
      "trivnet_opt_inf", "resnet50_keras/resnet50_fp16_keras_opt2.pb--activation_4/Relu--activation_7/Relu--float16--1,55,55,256",
      "resnet50", kccOpts, rtOpts % 'trivnet_activation_4__Relu:0.npy'],

  "4-rn50_res_unit_act7_10" : [
      "trivnet_opt_inf", "resnet50_keras/resnet50_fp16_keras_opt2.pb--activation_7/Relu--activation_10/Relu--float16--1,55,55,256",
      "resnet50", kccOpts, rtOpts % 'trivnet_activation_7__Relu:0.npy' ],

  "4-rn50_res_unit_act4_10" : [
      "trivnet_opt_inf", "resnet50_keras/resnet50_fp16_keras_opt2.pb--activation_4/Relu--activation_10/Relu--float16--1,55,55,256",
      "resnet50", kccOpts, rtOpts % 'trivnet_activation_4__Relu:0.npy'],

  "4-rn50_res_unit_act10_13" : [
      "trivnet_opt_inf", "resnet50_keras/resnet50_fp16_keras_opt2.pb--activation_10/Relu--activation_13/Relu--float16--1,55,55,256",
      "resnet50", kccOpts, rtOpts % 'trivnet_activation_10__Relu:0.npy' ],

  "4-rn50_res_unit_act10_13_fp32" : [
      "trivnet_opt_inf", "resnet50_keras/resnet50_fp32_keras_opt.pb--activation_10/Relu--activation_13/Relu--float32--1,55,55,256",
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
      "trivnet_opt_inf", "resnet50_keras/resnet50_fp16_keras_opt2.pb--activation_49/Relu--fc1000/Softmax--float16--1,7,7,2048",
      "resnet50", kccOpts, rtOpts % 'trivnet_activation_49__Relu:0.npy' ],

  "4-rn50_res_unit_act49_nosm" : [
      "trivnet_opt_inf", "resnet50_keras/resnet50_fp16_keras_opt2.pb--activation_49/Relu--fc1000/BiasAdd--float16--1,7,7,2048",
      "resnet50", kccOpts + ' --jf_data_layout native', rtOpts % 'trivnet_activation_49__Relu:0.npy' ],

  # Full resnet50 without softmax
  "7-rn50_single_sg_nosm_fp16" : [
      "trivnet_opt_inf", "resnet50_keras/resnet50_fp16_keras_opt2.pb--input_1--fc1000/BiasAdd--float16--1,224,224,3",
      "resnet50", kccOpts + ' --jf_data_layout native', rtOpts % 'trivnet_input_1:0.npy' ],
  "7-rn50_single_sg_nosm_fp32" : [
      "trivnet_opt_inf", "resnet50_keras/resnet50_fp32_keras_opt.pb--input_1--fc1000/BiasAdd--float32--1,224,224,3",
      "resnet50", kccOpts + ' --jf_data_layout native', rtOpts % 'trivnet_input_1:0.npy' ],
  # Full resnet50 without FC, softmax
  "7-rn50_single_sg_nofc_fp16" : [
      "trivnet_opt_inf", "resnet50_keras/resnet50_fp16_keras_opt2.pb--input_1--avg_pool/AvgPool--float16--1,224,224,3",
      "resnet50", kccOpts, rtOpts % 'trivnet_input_1:0.npy' ],

  # Resnet50 prefixes
  "6-rn50_single_sg_to_act10_fp16" : [
      "trivnet_opt_inf", "resnet50_keras/resnet50_fp16_keras_opt2.pb--input_1--activation_10/Relu--float16--1,224,224,3",
      "resnet50", kccOpts, rtOpts % 'trivnet_input_1:0.npy'],
  "6-rn50_single_sg_to_act13_fp16" : [
      "trivnet_opt_inf", "resnet50_keras/resnet50_fp16_keras_opt2.pb--input_1--activation_13/Relu--float16--1,224,224,3",
      "resnet50", kccOpts, rtOpts % 'trivnet_input_1:0.npy'],
  "6-rn50_single_sg_to_act16_fp16" : [
      "trivnet_opt_inf", "resnet50_keras/resnet50_fp16_keras_opt2.pb--input_1--activation_16/Relu--float16--1,224,224,3",
      "resnet50", kccOpts, rtOpts % 'trivnet_input_1:0.npy'],
  "6-rn50_single_sg_to_act19_fp16" : [
      "trivnet_opt_inf", "resnet50_keras/resnet50_fp16_keras_opt2.pb--input_1--activation_19/Relu--float16--1,224,224,3",
      "resnet50", kccOpts, rtOpts % 'trivnet_input_1:0.npy'],
  "6-rn50_single_sg_to_act22_fp16" : [
      "trivnet_opt_inf", "resnet50_keras/resnet50_fp16_keras_opt2.pb--input_1--activation_22/Relu--float16--1,224,224,3",
      "resnet50", kccOpts, rtOpts % 'trivnet_input_1:0.npy'],
  "6-rn50_single_sg_to_act25_fp16" : [
      "trivnet_opt_inf", "resnet50_keras/resnet50_fp16_keras_opt2.pb--input_1--activation_25/Relu--float16--1,224,224,3",
      "resnet50", kccOpts, rtOpts % 'trivnet_input_1:0.npy'],
  "6-rn50_single_sg_to_act28_fp16" : [
      "trivnet_opt_inf", "resnet50_keras/resnet50_fp16_keras_opt2.pb--input_1--activation_28/Relu--float16--1,224,224,3",
      "resnet50", kccOpts, rtOpts % 'trivnet_input_1:0.npy'],
  "6-rn50_single_sg_to_act34_fp16" : [
      "trivnet_opt_inf", "resnet50_keras/resnet50_fp16_keras_opt2.pb--input_1--activation_34/Relu--float16--1,224,224,3",
      "resnet50", kccOpts, rtOpts % 'trivnet_input_1:0.npy'],
  "6-rn50_single_sg_to_act37_fp16" : [
      "trivnet_opt_inf", "resnet50_keras/resnet50_fp16_keras_opt2.pb--input_1--activation_37/Relu--float16--1,224,224,3",
      "resnet50", kccOpts, rtOpts % 'trivnet_input_1:0.npy'],

  # Full resnet without 1st res unit and without softmax
  "6-rn50_res_unit_act4_49_nosm" : [
      "trivnet_opt_inf", "resnet50_keras/resnet50_fp16_keras_opt2.pb--activation_4/Relu--activation_49/Relu--float16--1,55,55,256",
      "resnet50", kccOpts, rtOpts % 'trivnet_activation_4__Relu:0.npy'],


}


# Regression waiver mechanism
testWaiver = [
  ['4-rn50_res_unit_act49_sm', 'WAIVE_SOFTMAX'],
  ['7-rn50_single_sg_nosm_fp32', 'WAIVE_ME'],
]

noGpuTestWaiver = [
]

qemuTestWaiver = [
]

