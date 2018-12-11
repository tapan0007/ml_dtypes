# Copyright (C) 2017, Amazon.com. All Rights Reserved
#
# Unit test for Tffe - uint8 quantized conv biasadd relu add

import numpy as np
import tensorflow as tf
from trivnet_common import conf
from trivnet_conv_ba_relu_pool_uint8 import conv_ba_relu
from trivnet_rn50_block_uint8 import resnet_block


if __name__ == '__main__':
    input_shape         = [conf.B, 224, 224, 3]

    # numpy inputs
    np.random.seed(15213)
    input_float32_np = np.random.uniform(low=conf.IMIN, high=conf.IMAX,
        size=input_shape).astype(np.float32)

    common_kwargs = dict(
        min_input=conf.IMIN, max_input=conf.IMAX,
        min_filter=conf.WMIN, max_filter=conf.WMAX,
        min_bias=conf.AMIN, max_bias=conf.AMAX,
        min_requantize=conf.RQMIN, max_requantize=conf.RQMAX,
        )
    ## tf graph
    input_float32 = tf.placeholder(tf.float32, shape=input_shape, name='input')
    with tf.name_scope(conf.netName):
        with tf.name_scope('block_0'):
            input_uint8, _, _ = tf.quantize(input_float32,
                min_range=conf.IMIN, max_range=conf.IMAX, T=tf.quint8,
                name='input_uint8')
            resnet_block0_float32 = conv_ba_relu(input_uint8,
                filter_shape=[7, 7, 3, 64], stride=2,
                has_ba=True, has_relu=True,
                pool_type='maxpool', pool_ksize=3, pool_stride=2,
                quantize_back=False, **common_kwargs)
        with tf.name_scope('block_1'):
            resnet_block1_float32 = resnet_block(resnet_block0_float32,
                channel_in=64, channel_hid=64, channel_out=256, stride=1,
                conv_branch=True, **common_kwargs)
        with tf.name_scope('block_2'):
            resnet_block2_float32 = resnet_block(resnet_block1_float32,
                channel_in=256, channel_hid=64, channel_out=256, stride=1,
                conv_branch=False, **common_kwargs)
        with tf.name_scope('block_3'):
            resnet_block3_float32 = resnet_block(resnet_block2_float32,
                channel_in=256, channel_hid=64, channel_out=256, stride=1,
                conv_branch=False, **common_kwargs)
        with tf.name_scope('block_4'):
            resnet_block4_float32 = resnet_block(resnet_block3_float32,
                channel_in=256, channel_hid=128, channel_out=512, stride=2,
                conv_branch=True, **common_kwargs)
        with tf.name_scope('block_5'):
            resnet_block5_float32 = resnet_block(resnet_block4_float32,
                channel_in=512, channel_hid=128, channel_out=512, stride=1,
                conv_branch=False, **common_kwargs)
        with tf.name_scope('block_6'):
            resnet_block6_float32 = resnet_block(resnet_block5_float32,
                channel_in=512, channel_hid=128, channel_out=512, stride=1,
                conv_branch=False, **common_kwargs)
        with tf.name_scope('block_7'):
            resnet_block7_float32 = resnet_block(resnet_block6_float32,
                channel_in=512, channel_hid=128, channel_out=512, stride=1,
                conv_branch=False, **common_kwargs)
        with tf.name_scope('block_8'):
            resnet_block8_float32 = resnet_block(resnet_block7_float32,
                channel_in=512, channel_hid=256, channel_out=1024, stride=2,
                conv_branch=True, **common_kwargs)
        with tf.name_scope('block_9'):
            resnet_block9_float32 = resnet_block(resnet_block8_float32,
                channel_in=1024, channel_hid=256, channel_out=1024, stride=1,
                conv_branch=False, **common_kwargs)
        with tf.name_scope('block_10'):
            resnet_block10_float32 = resnet_block(resnet_block9_float32,
                channel_in=1024, channel_hid=256, channel_out=1024, stride=1,
                conv_branch=False, **common_kwargs)
        with tf.name_scope('block_11'):
            resnet_block11_float32 = resnet_block(resnet_block10_float32,
                channel_in=1024, channel_hid=256, channel_out=1024, stride=1,
                conv_branch=False, **common_kwargs)
        with tf.name_scope('block_12'):
            resnet_block12_float32 = resnet_block(resnet_block11_float32,
                channel_in=1024, channel_hid=256, channel_out=1024, stride=1,
                conv_branch=False, **common_kwargs)
        with tf.name_scope('block_13'):
            resnet_block13_float32 = resnet_block(resnet_block12_float32,
                channel_in=1024, channel_hid=256, channel_out=1024, stride=1,
                conv_branch=False, **common_kwargs)
        with tf.name_scope('block_14'):
            resnet_block14_float32 = resnet_block(resnet_block13_float32,
                channel_in=1024, channel_hid=512, channel_out=2048, stride=2,
                conv_branch=True, **common_kwargs)
        with tf.name_scope('block_15'):
            resnet_block15_float32 = resnet_block(resnet_block14_float32,
                channel_in=2048, channel_hid=512, channel_out=2048, stride=1,
                conv_branch=False, **common_kwargs)
        with tf.name_scope('block_16'):
            resnet_block16_float32 = resnet_block(resnet_block15_float32,
                channel_in=2048, channel_hid=512, channel_out=2048, stride=1,
                conv_branch=False, **common_kwargs)
            resnet_block16_float32 = tf.nn.avg_pool(resnet_block16_float32,
                ksize=[1, 7, 7, 1],
                strides=[1, 7, 7, 1],
                padding='VALID',
                name='pool_float32')
        with tf.name_scope('fully_connected'):
            fully_connected_uint8, _, _ = tf.quantize(resnet_block16_float32,
                min_range=conf.RQMIN, max_range=conf.RQMAX, T=tf.quint8,
                name='fully_connected_uint8')
            result_float32 = conv_ba_relu(fully_connected_uint8,
                filter_shape=[1, 1, 2048, 1000], stride=1,
                has_ba=True, has_relu=False,
                quantize_back=False, **common_kwargs)
        output = tf.identity(result_float32, name='output')

    conf.gen_graph(output, input_data=input_float32_np, need_freezing=False)
