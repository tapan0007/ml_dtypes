# Copyright (C) 2017, Amazon.com. All Rights Reserved
#
# Unit test for Tffe - single convolution

from trivnet_common import *

input_shape     = [conf.B, 1, conf.H, conf.C]
weight_shape    = [conf.R, conf.R, conf.C, conf.M]
strides         = [1, conf.S, conf.S, 1]

w1val   = conf.gen_array_linspace(conf.WMIN, conf.WMAX, weight_shape)
w1      = conf.gen_variable_tensor(name = conf.netName + "/weight1", initializer = w1val)

i0      = tf.placeholder(conf.tfDataType, shape = input_shape, name = "input")
i1      = tf.nn.conv2d(i0, w1, strides, "SAME", name = conf.netName + "/i1")
i2      = tf.squeeze(i1, axis=[1,3])
output  = tf.identity(i2, name = conf.netName + "/output")

i0val   = conf.gen_array_rand(conf.IMIN, conf.IMAX, input_shape)

conf.gen_graph(output, input_data=i0val)
