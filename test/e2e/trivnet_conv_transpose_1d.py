# Copyright (C) 2017, Amazon.com. All Rights Reserved
#
# Unit test for Tffe - single convolution

from trivnet_common import *

input_shape     = [conf.B, 1, conf.H, conf.C]
weight_shape    = [1, conf.R, conf.M, conf.C]
strides         = [1, 1, conf.S, 1]
output_shape    = [conf.B, 1, conf.H*conf.S, conf.M]
padding         = "SAME"

w1val   = conf.gen_array_linspace(conf.WMIN, conf.WMAX, weight_shape)
w1      = conf.gen_variable_tensor(name = conf.netName + "/weight1", initializer=w1val)

w2val   = conf.gen_array_linspace(conf.WMIN, conf.WMAX, [conf.C])
w2      = conf.gen_variable_tensor(name = conf.netName + "/bias1", initializer = w2val)

i0      = tf.placeholder(conf.tfDataType, shape = input_shape, name = "input")
i1      = tf.nn.conv2d_transpose(i0, w1, output_shape = output_shape, strides = strides, padding = padding, name = conf.netName + "/conv2d_transpose")
#i2      = tf.nn.bias_add(i1, w2, name = conf.netName + "/i2")
#i3      = tf.nn.tanh(i2, name = conf.netName + "/i3")
i3       = i1
if conf.L > 1:
    weight_shape2   = [1, conf.R * 2, conf.M, conf.M]
    output_shape2   = [conf.B, 1, conf.H * conf.S * conf.S * 2, conf.M]
    strides2        = [1, 1, conf.S * 2, 1]
    w3val           = conf.gen_array_linspace(conf.WMIN, conf.WMAX, weight_shape2)
    w3              = conf.gen_variable_tensor(name = conf.netName + "/weight2", initializer=w3val)
    i4  = tf.nn.conv2d_transpose(i3, w3, output_shape = output_shape2, strides = strides2, padding = padding, name = conf.netName + "/conv2d_transpose2")
else:
    i4  = i3
output  = tf.identity(i4, name = conf.netName + "/output")

i0val   = conf.gen_array_linspace(conf.IMIN, conf.IMAX, input_shape)

conf.gen_graph(output=output, input_data=i0val)
