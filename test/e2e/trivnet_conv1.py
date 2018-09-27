# Copyright (C) 2017, Amazon.com. All Rights Reserved
#
# Unit test for Tffe - single convolution

from trivnet_common import *

input_shape     = [conf.B, conf.H, conf.H, conf.C]
weight_shape    = [conf.R, conf.R, conf.C, conf.M]
strides         = [1, conf.S, conf.S, 1]

if conf.padType is not None:
    padType = conf.padType
else:
    padType = "SAME"

w1val   = conf.gen_array_linspace(conf.WMIN, conf.WMAX, weight_shape)
w1      = conf.gen_variable_tensor(name = conf.netName + "/weight1", initializer = w1val)

i0      = tf.placeholder(conf.tfDataType, shape = input_shape, name = "input")
if hasattr(conf, 'PADW') and hasattr(conf, 'PADE'):
    padding_exp = [[0, 0], [conf.PADW, conf.PADE], [conf.PADW, conf.PADE], [0, 0]]
    i1_pre      = tf.pad(i0, padding_exp)
    i1          = tf.nn.conv2d(i1_pre, w1, strides, "VALID", name = conf.netName + "/i1")
else:
    i1          = tf.nn.conv2d(i0, w1, strides, padType, name = conf.netName + "/i1")
output  = tf.identity(i1, name = conf.netName + "/output")

i0val   = conf.gen_array_rand(conf.IMIN, conf.IMAX, input_shape)

# Overide some values:
for row,col,val in conf.fmapValList:
  i0val[0, row, col, 0] = val

conf.gen_graph(output, input_data=i0val)
