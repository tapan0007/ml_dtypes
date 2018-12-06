# Copyright (C) 2017, Amazon.com. All Rights Reserved
#
# Unit test for Tffe - single convolution

from trivnet_common import *


# [H, C] x transpose([C, M])
#input_shape_0  = [conf.H, conf.H, conf.C]
#input_shape_1  = [conf.H, conf.M, conf.C]
input_shape_0  = [conf.H, conf.C]
input_shape_1  = [conf.M, conf.C]

i0      = tf.placeholder(conf.tfDataType, shape = input_shape_0, name = "input0")
i1      = tf.placeholder(conf.tfDataType, shape = input_shape_1, name = "input1")

# Not used 
BIAS1 = np.zeros([conf.C])
wval   = conf.gen_array_linspace(conf.WMIN, conf.WMAX, BIAS1.shape)
w      = conf.gen_variable_tensor(name = conf.netName + "/bias1", initializer = wval)

#addend = conf.gen_array_linspace(conf.WMIN, conf.WMAX, input_shape_1)

#result = tf.add(i0, i1 , name = conf.netName + "/add")
#i1_sqrt = tf.add(i0, i1 , name = conf.netName + "/add")
i1_sqrt = tf.sqrt(i1 , name = conf.netName + "/sqrt1")
i0_sqrt = tf.sqrt(i0 , name = conf.netName + "/sqrt0")
result      = tf.matmul(i0_sqrt, i1_sqrt, transpose_b=True, name = conf.netName + "/matmul")

output  = tf.identity(result, name = conf.netName + "/output")

i0val   = conf.gen_array_rand(conf.IMIN, conf.IMAX, input_shape_0)
i1val   = conf.gen_array_rand(conf.IMIN, conf.IMAX, input_shape_1)


conf.gen_graph(output, input_data={"input0:0" : i0val, "input1:0" : i1val}, need_freezing = True)
