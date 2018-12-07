# Copyright (C) 2017, Amazon.com. All Rights Reserved
#
# Unit test for Tffe - uint8 quantize and dequantize

from trivnet_common import *


if __name__ == '__main__':
    input_shape     = [conf.B, conf.H, conf.H, conf.C]

    # numpy inputs
    np.random.seed(15213)
    input_float32_np = np.random.uniform(low=conf.IMIN, high=conf.IMAX,
        size=input_shape).astype(np.float32)

    # tf graph
    input_float32 = tf.placeholder(tf.float32, shape=input_shape, name='input')
    input_uint8, _, _ = tf.quantize(input_float32,
        min_range=conf.IMIN, max_range=conf.IMAX, T=tf.quint8,
        name='%s/quantize_float32_uint8' % conf.netName)
    output = tf.dequantize(input_uint8,
        min_range=conf.IMIN, max_range=conf.IMAX,
        name='%s/output' % conf.netName, mode='MIN_FIRST')
    conf.gen_graph(output, input_data=input_float32_np, need_freezing=False)
