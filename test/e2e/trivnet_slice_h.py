# Copyright (C) 2017, Amazon.com. All Rights Reserved
#
# Unit test for Tffe - single convolution

from trivnet_common import *

input_shape     = [conf.B, conf.H, 1, conf.C]
weight_shape    = [conf.C]
slice_begin     = [0, conf.SBEGIN, 0, 0]
slice_size      = [-1, conf.SSIZE, -1, -1]

w1val   = conf.gen_array_linspace(conf.WMIN, conf.WMAX, weight_shape)
w1      = conf.gen_variable_tensor(name = conf.netName + "/bias1", initializer = w1val)

i0      = tf.placeholder(conf.tfDataType, shape = input_shape, name = "input")
i1      = tf.nn.bias_add(i0, w1, name = conf.netName + "/i1")
i2      = tf.slice(i1, slice_begin, slice_size, name = conf.netName + "/i2")
output  = tf.identity(i2, name = conf.netName + "/output")

i0val   = conf.gen_array_rand(conf.IMIN, conf.IMAX, input_shape)

conf.gen_graph(output, input_data=i0val)
