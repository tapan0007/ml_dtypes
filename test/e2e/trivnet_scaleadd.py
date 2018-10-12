# Copyright (C) 2018, Amazon.com. All Rights Reserved

from trivnet_common import *

IF1 = np.zeros([conf.B, conf.C])
BIAS1 = np.zeros([])

wval = conf.gen_array_linspace(conf.WMIN, conf.WMAX, BIAS1.shape)
w = conf.gen_variable_tensor(name = conf.netName + "/mul_const", initializer = wval)
w2val = conf.gen_array_linspace(conf.AMIN, conf.AMAX, BIAS1.shape)
w2 = conf.gen_variable_tensor(name = conf.netName + "/add_const", initializer = w2val)

i0 = tf.placeholder(conf.tfDataType, shape=IF1.shape, name="input")
i1 = tf.scalar_mul(w, i0)
i2 = tf.add(i1, w2)
output = tf.identity(i2, name = conf.netName+"/output")

i0val   = conf.gen_array_rand(conf.IMIN, conf.IMAX, IF1.shape)
conf.gen_graph(output, input_data=i0val)

