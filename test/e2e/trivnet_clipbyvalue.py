# Copyright (C) 2017, Amazon.com. All Rights Reserved
#
# Unit test for Tffe - single convolution

from trivnet_common import *

input_shape     = [conf.B, conf.H, conf.H, conf.C]
weight_shape    = [conf.C]
clip_value_min  = conf.XMIN
clip_value_max  = conf.XMAX

w1val   = conf.gen_array_linspace(conf.WMIN, conf.WMAX, weight_shape)
w1      = conf.gen_variable_tensor(name = conf.netName + "/bias1", initializer = w1val)

i0      = tf.placeholder(conf.tfDataType, shape = input_shape, name = "input")
i1      = tf.nn.bias_add(i0, w1, name = conf.netName + "/i1")
i2      = tf.clip_by_value(i1, clip_value_min, clip_value_max, name = conf.netName + "/i2")
output  = tf.identity(i2, name = conf.netName + "/output")

i0val   = conf.gen_array_rand(conf.IMIN, conf.IMAX, input_shape)

conf.gen_graph(output, input_data=i0val)
