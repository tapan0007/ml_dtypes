# Copyright (C) 2018, Amazon.com. All Rights Reserved
#
# Kaena test for continuous inference
# The compiler is called primarily to compure reference golden files

import os

testConfigMap = {}
testWaiver = []
noGpuTestWaiver = []
qemuTestWaiver = []

from test_list import kPath, rnPreFp16, MEv2

def gen_rn50_nne_to_cinf_variants(ind):
    jpgloc = "/work1/kumrnii/ki/alljpg/J"
    jpglist = "%s/%d.jpg %s/%d.jpg %s/%d.jpg %s/%d.jpg"%(jpgloc,ind,jpgloc,ind+1,jpgloc,ind+2,jpgloc,ind+3)
    rv = [ "tf_pb", "resnet50_keras/resnet50_fp16_keras_opt2.pb","resnet50",
         "--input_node input_1  --depth 2  --debug 1 %s --partition from fc1000/Softmax --executors wave 0 host 1 %s --batch 4 "%(rnPreFp16, MEv2("RN50-Repl")),
         "--input_files %s" % jpglist]
    # pprint.pprint(rv)
    return rv

for i in range(1,10000,4):
    testConfigMap["7-rn50_nne_fp16_b4_wave_cinf_%d"%(i-1)] = gen_rn50_nne_to_cinf_variants(i)
