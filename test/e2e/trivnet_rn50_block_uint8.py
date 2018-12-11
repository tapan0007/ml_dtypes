# Copyright (C) 2017, Amazon.com. All Rights Reserved
#
# Unit test for Tffe - uint8 quantized conv biasadd relu add

import numpy as np
import tensorflow as tf
from trivnet_common import conf
from trivnet_conv_ba_relu_pool_uint8 import conv_ba_relu


def resnet_block(input_float32, channel_in, channel_hid, channel_out,
    min_filter, max_filter, stride, min_input, max_input, min_bias, max_bias,
    min_requantize, max_requantize, conv_branch):
    input_uint8, _, _ = tf.quantize(input_float32,
        min_range=min_input, max_range=max_input, T=tf.quint8,
        name='input_uint8')
    b1_filter1_shape    = [1, 1, channel_in, channel_hid]
    b1_filter2_shape    = [3, 3, channel_hid, channel_hid]
    b1_filter3_shape    = [1, 1, channel_hid, channel_out]
    b2_filter1_shape    = [1, 1, channel_in, channel_out]
    common_kwargs = dict(
        min_input=min_input, max_input=max_input,
        min_filter=min_filter, max_filter=max_filter,
        min_bias=min_bias, max_bias=max_bias,
        min_requantize=min_requantize, max_requantize=max_requantize,
        has_ba=True)

    # branch 1
    with tf.name_scope('branch_1'):
        with tf.name_scope('conv_1'):
            b1_conv1_uint8 = conv_ba_relu(input_uint8,
                filter_shape=b1_filter1_shape, stride=stride,
                has_relu=True, quantize_back=True, **common_kwargs)
        with tf.name_scope('conv_2'):
            b1_conv2_uint8 = conv_ba_relu(b1_conv1_uint8,
                filter_shape=b1_filter2_shape, stride=1,
                has_relu=True, quantize_back=True, **common_kwargs)
        with tf.name_scope('conv_3'):
            b1_result_float32 = conv_ba_relu(b1_conv2_uint8,
                filter_shape=b1_filter3_shape, stride=1,
                has_relu=False, quantize_back=False, **common_kwargs)

    # branch 2
    if conv_branch:
        with tf.name_scope('branch_2'):
            b2_result_float32 = conv_ba_relu(input_uint8,
                filter_shape=b2_filter1_shape, stride=stride,
                has_relu=False, quantize_back=False, **common_kwargs)
    else:
        b2_result_float32 = input_float32

    # resadd and relu
    result_float32 = tf.add(b1_result_float32, b2_result_float32,
        name='resadd_float32')
    result_float32 = tf.nn.relu(result_float32,
        name='relu_float32')
    return result_float32


if __name__ == '__main__':
    input_shape         = [conf.B, conf.H, conf.H, conf.CIN]

    # numpy inputs
    np.random.seed(15213)
    input_float32_np = np.random.uniform(low=conf.IMIN, high=conf.IMAX,
        size=input_shape).astype(np.float32)

    ## tf graph
    input_float32 = tf.placeholder(tf.float32, shape=input_shape, name='input')
    with tf.name_scope(conf.netName):
        result_float32 = resnet_block(input_float32,
            channel_in=conf.CIN, channel_hid=conf.CHID, channel_out=conf.COUT,
            min_filter=conf.WMIN, max_filter=conf.WMAX, stride=conf.S,
            min_input=conf.IMIN, max_input=conf.IMAX,
            min_bias=conf.AMIN, max_bias=conf.AMAX,
            min_requantize=conf.RQMIN, max_requantize=conf.RQMAX,
            conv_branch=('CONVBRANCH' in conf.__dict__))
        if 'HASAVGPOOL' in conf.__dict__:
            result_float32 = tf.nn.avg_pool(result_float32,
                ksize=[1, 7, 7, 1], strides=[1, 7, 7, 1], padding='VALID',
                name='pool_float32')
        output = tf.identity(result_float32, name='output')
    conf.gen_graph(output, input_data=input_float32_np, need_freezing=False)
