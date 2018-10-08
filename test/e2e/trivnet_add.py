# Copyright (C) 2017, Amazon.com. All Rights Reserved
#
# Unit test for Tffe - biasadd followed by residual add
# The weight ranges are used for the bias add

from trivnet_common import *

IF1 = np.zeros([conf.B, conf.H, conf.H, conf.C])
BIAS1 = np.zeros([conf.C])

wval   = conf.gen_array_linspace(conf.WMIN, conf.WMAX, BIAS1.shape)
w      = conf.gen_variable_tensor(name = conf.netName + "/bias1", initializer = wval)

i0 = tf.placeholder(conf.tfDataType, shape=IF1.shape, name="input")
i1 = tf.nn.bias_add(i0, w, name=conf.netName + "/i1")
if conf.joinType == "SUB":
    i2 = i1 - i0
else:    
    i2 = tf.add(i1, i0, name=conf.netName + "/i2")

output = tf.identity(i2, name=conf.netName+"/output")

i0val  = conf.gen_array_rand(conf.IMIN, conf.IMAX, IF1.shape)

conf.gen_graph(output, input_data=i0val)
