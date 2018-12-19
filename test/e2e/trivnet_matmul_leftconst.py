# Copyright (C) 2017, Amazon.com. All Rights Reserved
#
# Unit test for Tffe - single matmul where the left operand is const

from trivnet_common import *


# [H, C] x transpose([C, M])
left_shape  = [conf.H, conf.C]
right_shape  = [conf.M, conf.C]

left = conf.gen_array_rand(conf.WMIN, conf.WMAX, left_shape)
right = tf.placeholder(conf.tfDataType, shape=right_shape, name="input")

output = tf.matmul(left, right, transpose_b=True, name=conf.netName+"/output")
right_val = conf.gen_array_rand(conf.IMIN, conf.IMAX, right_shape)


conf.gen_graph(output, input_data=right_val, need_freezing=False)
