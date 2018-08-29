# Copyright (C) 2017, Amazon.com. All Rights Reserved
#
# Unit test for Tffe - single convolution

from trivnet_common import *

input_shape     = [conf.B, conf.H, conf.H, conf.C]
weight_shape    = [conf.R, conf.R, conf.C, conf.M]
padding         = "SAME"
strides         = [conf.S, conf.S]
dilation        = [conf.D, conf.D]

w1val   = conf.gen_array_linspace(conf.WMIN, conf.WMAX, weight_shape)
w1      = conf.gen_variable_tensor(name = conf.netName + "/weight1", initializer = w1val)

i0      = tf.placeholder(conf.tfDataType, shape = input_shape, name = "input")
i1      = tf.nn.convolution(
            input   = i0, 
            filter  = w1, 
            dilation_rate = dilation, 
            strides = strides, 
            padding = padding, 
            name = conf.netName + "/i1")
output  = tf.identity(i1, name = conf.netName + "/output")

i0val   = conf.gen_array_rand(conf.IMIN, conf.IMAX, input_shape)

# Overide some values:
for row,col,val in conf.fmapValList:
  i0val[0, row, col, 0] = val

conf.gen_graph(output, input_data=i0val)
