# Copyright (C) 2017, Amazon.com. All Rights Reserved
#
# Unit test for Tffe - single convolution

from trivnet_common import *

input_shape     = [conf.B, conf.H, conf.H, conf.C]
weight_shape    = [conf.R, conf.R, conf.C, conf.M]

i0      = tf.placeholder(conf.tfDataType, shape = input_shape, name = "input")
i1      = tf.reduce_sum(i0, 3, name = conf.netName + "/i1", keepdims=True)
output  = tf.identity(i1, name = conf.netName + "/output")

i0val   = conf.gen_array_rand(conf.IMIN, conf.IMAX, input_shape)


conf.gen_graph(output, input_data=i0val, need_freezing=False)


