# Copyright (C) 2017, Amazon.com. All Rights Reserved
#
# Unit test for Tffe - uint8 quantized conv biasadd relu

import numpy as np
import tensorflow as tf
from trivnet_common import conf


def conv_ba_relu(input_uint8, min_input, max_input, 
    filter_shape, min_filter, max_filter, stride, filter_uint8_np=None,
    has_ba=False, min_bias=None, max_bias=None, bias_float32_np=None,
    has_relu=False, pool_type=None, pool_ksize=None, pool_stride=None,
    quantize_back=False, min_requantize=None, max_requantize=None):
    if filter_uint8_np is None:
        filter_float32_np = np.random.uniform(low=min_filter, high=max_filter,
            size=filter_shape).astype(np.float32)
        filter_uint8_np = quantize_np(filter_float32_np,
            range_min=min_filter, range_max=max_filter)
    filter_uint8 = tf.constant(filter_uint8_np, dtype=tf.quint8, name='filter')
    conv_int32, _, _ = tf.nn.quantized_conv2d(
        input_uint8, filter_uint8,
        min_input=min_input, max_input=max_input,
        min_filter=min_filter, max_filter=max_filter,
        strides=[1, stride, stride, 1], padding='SAME', out_type=tf.qint32,
        name='quantized_conv2d')
    min_output_int32, max_output_int32 = min_max_output_int32(
        range_input=max_input-min_input, range_filter=max_filter-min_filter)
    output = tf.dequantize(conv_int32,
        min_range=min_output_int32, max_range=max_output_int32,
        name='dequantize_float32', mode='MIN_FIRST')
    if has_ba:
        if bias_float32_np is None:
            bias_float32_np = np.random.uniform(low=min_bias, high=max_bias,
                size=filter_shape[-1]).astype(np.float32)
        output = tf.nn.bias_add(output, bias=bias_float32_np,
            name='biasadd_float32')
    if has_relu:
        output = tf.nn.relu(output, name='relu_float32')
    if pool_type is not None:
        pool_type = pool_type.lower()
        if pool_type == 'maxpool':
            pool_func = tf.nn.max_pool
        elif pool_type == 'avgpool':
            pool_func = tf.nn.avg_pool
        output = pool_func(output,
                ksize=[1, pool_ksize, pool_ksize, 1],
                strides=[1, pool_stride, pool_stride, 1],
                padding='VALID',
                name='pool_float32')
    if quantize_back:
        output, _, _ = tf.quantize(output,
            min_range=min_requantize, max_range=max_requantize, T=tf.quint8,
            name='quantize_uint8')
    return output

def quantize_np(input_float32, range_min, range_max, T=np.uint8):
    max_minus_min = range_max - range_min
    quant_scale = (np.iinfo(T).max - np.iinfo(T).min) / max_minus_min
    zero_point = np.round(-range_min * quant_scale)
    return (input_float32 * quant_scale + zero_point).astype(np.uint8)

def min_max_output_int32(range_input, range_filter, T=np.uint8):
    range_int_type = np.iinfo(T).max - np.iinfo(T).min
    factor_input = range_input / range_int_type
    factor_filter = range_filter / range_int_type
    factor_output = factor_input * factor_filter
    min_output_int32 = (np.iinfo(np.int32).min + 1) * factor_output
    max_output_int32 = np.iinfo(np.int32).max * factor_output
    return min_output_int32, max_output_int32


if __name__ == '__main__':
    input_shape     = [conf.B, conf.H, conf.H, conf.C]
    filter_shape    = [conf.R, conf.R, conf.C, conf.M]

    # numpy inputs
    np.random.seed(15213)
    input_float32_np = np.random.uniform(low=conf.IMIN, high=conf.IMAX,
        size=input_shape).astype(np.float32)

    ## tf graph
    input_float32 = tf.placeholder(tf.float32, shape=input_shape, name='input')
    with tf.name_scope(conf.netName):
        input_uint8, _, _ = tf.quantize(input_float32,
            min_range=conf.IMIN, max_range=conf.IMAX, T=tf.quint8,
            name='input_uint8')
        kwargs = dict(
            min_input=conf.IMIN, max_input=conf.IMAX,
            filter_shape=filter_shape,
            min_filter=conf.WMIN, max_filter=conf.WMAX, stride=conf.S,
            has_relu=('HASRELU' in conf.__dict__))
        kwargs['has_ba'] = 'HASBA' in conf.__dict__
        if kwargs['has_ba']:
            kwargs['min_bias'] = conf.AMIN
            kwargs['max_bias'] = conf.AMAX
        if 'HASPOOL' in conf.__dict__:
            kwargs['pool_type'] = conf.poolType
            kwargs['pool_ksize'] = conf.K
            kwargs['pool_stride'] = conf.D
        kwargs['quantize_back'] = 'QUANTIZEBACK' in conf.__dict__
        if kwargs['quantize_back']:
            kwargs['min_requantize'] = conf.RQMIN
            kwargs['max_requantize'] = conf.RQMAX
        result_float32 = conv_ba_relu(input_uint8, **kwargs)
        output = tf.identity(result_float32, name='output')

    conf.gen_graph(output, input_data=input_float32_np, need_freezing=False)
