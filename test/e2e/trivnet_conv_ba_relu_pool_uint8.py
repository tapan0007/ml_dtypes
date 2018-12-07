# Copyright (C) 2017, Amazon.com. All Rights Reserved
#
# Unit test for Tffe - uint8 quantized conv biasadd relu

from trivnet_common import *


"""Returns: quantized_value, zero_point
"""
def quantize_np(input_float32, range_min, range_max, T=np.uint8):
    max_minus_min = range_max - range_min
    quant_scale = (np.iinfo(T).max - np.iinfo(T).min) / max_minus_min
    zero_point = np.round(-range_min * quant_scale)
    return (input_float32 * quant_scale + zero_point).astype(np.uint8), zero_point

if __name__ == '__main__':
    input_shape     = [conf.B, conf.H, conf.H, conf.C]
    filter_shape    = [conf.R, conf.R, conf.C, conf.M]
    strides         = [1, conf.S, conf.S, 1]

    # numpy inputs
    np.random.seed(15213)
    input_float32_np = np.random.uniform(low=conf.IMIN, high=conf.IMAX,
        size=input_shape).astype(np.float32)
    filter_float32_np = np.random.uniform(low=conf.WMIN, high=conf.WMAX,
        size=filter_shape).astype(np.float32)
    filter_uint8_np, _ = quantize_np(filter_float32_np,
        range_min=conf.WMIN, range_max=conf.WMAX)

    ## tf graph
    input_float32 = tf.placeholder(tf.float32, shape=input_shape, name='input')
    input_uint8, _, _ = tf.quantize(input_float32,
        min_range=conf.IMIN, max_range=conf.IMAX, T=tf.quint8,
        name='%s/input_uint8'%conf.netName)
    filter_uint8 = tf.constant(filter_uint8_np,
        name='%s/filter'%conf.netName, dtype=tf.quint8)

    # calculate min_output and max_output
    range_uint8 = np.iinfo(np.uint8).max - np.iinfo(np.uint8).min
    factor_input = (conf.IMAX - conf.IMIN) / range_uint8
    factor_filter = (conf.WMAX - conf.WMIN) / range_uint8
    factor_output = factor_input * factor_filter
    min_output_int32 = (np.iinfo(np.int32).min + 1) * factor_output
    max_output_int32 = np.iinfo(np.int32).max * factor_output
    print('min_output_int32:', min_output_int32)
    print('max_output_int32:', max_output_int32)
    conv2d_int32, _, _ = tf.nn.quantized_conv2d(
        input_uint8, filter_uint8,
        min_input=conf.IMIN, max_input=conf.IMAX,
        min_filter=conf.WMIN, max_filter=conf.WMAX,
        strides=strides, padding=conf.padType, out_type=tf.qint32,
        name='%s/quantized_conv2d'%conf.netName)
    result_float32 = tf.dequantize(conv2d_int32,
        min_range=min_output_int32, max_range=max_output_int32,
        name='%s/dequantize_int32_float32' % conf.netName, mode='MIN_FIRST')
    if 'HASBA' in conf.__dict__:
        bias_float32_np = np.random.uniform(low=conf.AMIN, high=conf.AMAX,
            size=filter_shape[-1]).astype(np.float32)
        result_float32 = tf.nn.bias_add(result_float32, bias=bias_float32_np,
            name='%s/biasadd_float32'%conf.netName)
    if 'HASRELU' in conf.__dict__:
        result_float32 = tf.nn.relu(result_float32,
            name='%s/relu_float32'%conf.netName)
    if 'HASPOOL' in conf.__dict__:
        if conf.poolType == 'MAXPOOL':
            pool_func = tf.nn.max_pool
        elif conf.poolType == 'AVGPOOL':
            pool_func = tf.nn.avg_pool
        result_float32 = pool_func(result_float32,
            ksize=[1, conf.K, conf.K, 1],
            strides=[1, conf.D, conf.D, 1],
            padding='VALID',
            name='%s/pool_float32'%conf.netName)
    if 'QUANTIZEBACK' in conf.__dict__:
        output, _, _ = tf.quantize(result_float32,
            min_range=conf.RQMIN, max_range=conf.RQMAX, T=tf.quint8,
            name='%s/output' % conf.netName)
    else:
        output = tf.identity(result_float32, name='%s/output' % conf.netName)

    conf.gen_graph(output, input_data=input_float32_np, need_freezing=False)
