# Copyright (C) 2017, Amazon.com. All Rights Reserved
#
# Unit test for Tffe - single convolution

from trivnet_common import *

input_shape     = [conf.B, conf.H, conf.H, conf.C]
weight_shape    = [conf.R, conf.R, conf.C, conf.M]
strides         = [1, conf.S, conf.S, 1]
output_shape    = [conf.B, conf.H*conf.S, conf.H*conf.S, conf.C]
padding         = "SAME"

w1val   = conf.gen_array_linspace(conf.WMIN, conf.WMAX, weight_shape)
w1      = conf.gen_variable_tensor(name = conf.netName + "/weight1", initializer=w1val)

i0      = tf.placeholder(conf.tfDataType, shape = input_shape, name = "input")
i1      = tf.nn.conv2d_transpose(i0, w1, output_shape = output_shape, strides = strides, padding = padding, name = conf.netName + "/conv2d_transpose")
output  = tf.identity(i1, name = conf.netName + "/output")

i0val   = conf.gen_array_linspace(conf.IMIN, conf.IMAX, input_shape)

conf.gen_graph(output=output, input_data=i0val)
