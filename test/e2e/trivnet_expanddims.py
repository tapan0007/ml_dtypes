# Copyright (C) 2017, Amazon.com. All Rights Reserved
#
# Unit test for Tffe - single convolution

from trivnet_common import *

input_shape     = [conf.B, conf.H, conf.C]
weight_shape    = [1, conf.R, conf.C, conf.M]
bias_shape      = [conf.M]
strides         = [1, 1, conf.S, 1]

w1val   = conf.gen_array_linspace(conf.WMIN, conf.WMAX, bias_shape)
w1      = conf.gen_variable_tensor(name = conf.netName + "/bias1", initializer = w1val)

w3val   = conf.gen_array_linspace(conf.WMIN, conf.WMAX, weight_shape)
w3      = conf.gen_variable_tensor(name = conf.netName + "/weight1", initializer = w3val)

i0      = tf.placeholder(conf.tfDataType, shape = input_shape, name = "input")
i1      = tf.nn.bias_add(i0, w1, name = conf.netName + "/i1")
i2      = tf.expand_dims(i1, 1)
i3      = tf.nn.conv2d(i2, w3, strides, "SAME", name = conf.netName + "/i3")
output  = tf.identity(i3, name = conf.netName + "/output")

i0val   = conf.gen_array_rand(conf.IMIN, conf.IMAX, input_shape)

conf.gen_graph(output, input_data=i0val)
