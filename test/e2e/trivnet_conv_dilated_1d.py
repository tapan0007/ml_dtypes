# Copyright (C) 2017, Amazon.com. All Rights Reserved
#
# Unit test for Tffe - single convolution

from trivnet_common import *

input_shape     = [conf.B, 1, conf.H, conf.C]
weight_shape    = [conf.R, 1, conf.C, conf.M]
padding         = "SAME"
strides         = [1, conf.S]
dilation        = [1, conf.D]

w1val   = conf.gen_array_linspace(conf.WMIN, conf.WMAX, weight_shape)
w1      = conf.gen_variable_tensor(name = conf.netName + "/weight1", initializer = w1val)

w2val   = conf.gen_array_linspace(conf.WMIN, conf.WMAX, [conf.C])
w2      = conf.gen_variable_tensor(name = conf.netName + "/bias1", initializer = w2val)

i0      = tf.placeholder(conf.tfDataType, shape = input_shape, name = "input")
i0b     = tf.nn.bias_add(i0, w2, name = conf.netName + "/i0b")
i0c     = i0 + i0b
i1      = tf.nn.convolution(
            input   = i0c, 
            filter  = w1, 
            dilation_rate = dilation, 
            strides = strides, 
            padding = padding, 
            name = conf.netName + "/i1")
i2      = tf.nn.relu(i1, name = conf.netName + "/i2")
output  = tf.identity(i2, name = conf.netName + "/output")

i0val   = conf.gen_array_rand(conf.IMIN, conf.IMAX, input_shape)

# Overide some values:
for row,col,val in conf.fmapValList:
  i0val[0, row, col, 0] = val

conf.gen_graph(output, input_data=i0val)
