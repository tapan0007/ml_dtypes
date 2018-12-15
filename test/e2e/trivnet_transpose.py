# Copyright (C) 2017, Amazon.com. All Rights Reserved
#
# Unit test for Tffe - matrix multiply (LSTM)


from trivnet_common import *

input_shape     = [conf.H, conf.C]
output_shape    = [conf.C, conf.H]

i0 = tf.placeholder(conf.tfDataType, shape = input_shape, name = "input")
output = tf.transpose(i0, perm=[1,0], name = conf.netName + "/output")

i0val   = conf.gen_array_rand(conf.IMIN, conf.IMAX, input_shape)

conf.gen_graph(output, input_data=i0val, need_freezing=False)
