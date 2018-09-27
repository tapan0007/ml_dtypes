# Copyright (C) 2017, Amazon.com. All Rights Reserved
#
# Unit test for Tffe - single convolution

from trivnet_common import *

halfC = conf.C//2
assert(halfC * 2 == conf.C)
assert(conf.C > 1)
input_shape     = [conf.B, conf.H, conf.H, conf.C]
weight_shape    = [conf.R, conf.R, conf.C, conf.M]
strides         = [1, conf.S, conf.S, 1]

w1val   = conf.gen_array_linspace(conf.WMIN, conf.WMAX, weight_shape)
w1      = conf.gen_variable_tensor(name = "weight1", initializer = w1val)

i0      = tf.placeholder(conf.tfDataType, shape = input_shape, name = "input")
i1      = tf.nn.conv2d(i0, w1, strides, "SAME")
i2      = tf.nn.relu(i1[:, :, :, halfC : conf.C])
output  = tf.identity(i2, name = conf.netName + "/output")

i0val   = conf.gen_array_rand(conf.IMIN, conf.IMAX, input_shape)

conf.gen_graph(output, input_data=i0val)
