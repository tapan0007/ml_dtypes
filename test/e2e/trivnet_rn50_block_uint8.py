# Copyright (C) 2017, Amazon.com. All Rights Reserved
#
# Unit test for Tffe - uint8 quantized conv biasadd relu add

from trivnet_common import *


"""Returns: quantized_value, zero_point
"""
def quantize_np(input_float32, range_min, range_max, T=np.uint8):
    max_minus_min = range_max - range_min
    quant_scale = (np.iinfo(T).max - np.iinfo(T).min) / max_minus_min
    zero_point = np.round(-range_min * quant_scale)
    return (input_float32 * quant_scale + zero_point).astype(np.uint8), zero_point

def min_max_output_int32(range_input, range_filter, T=np.uint8):
    range_int_type = np.iinfo(T).max - np.iinfo(T).min
    factor_input = range_input / range_int_type
    factor_filter = range_filter / range_int_type
    factor_output = factor_input * factor_filter
    min_output_int32 = (np.iinfo(np.int32).min + 1) * factor_output
    max_output_int32 = np.iinfo(np.int32).max * factor_output
    return min_output_int32, max_output_int32

if __name__ == '__main__':
    input_shape         = [conf.B, conf.H, conf.H, conf.CIN]
    b1_filter1_shape    = [1, 1, conf.CIN, conf.CHID]
    b1_filter2_shape    = [3, 3, conf.CHID, conf.CHID]
    b1_filter3_shape    = [1, 1, conf.CHID, conf.COUT]
    b2_filter1_shape    = [1, 1, conf.CIN, conf.COUT]
    strides             = [1, 1, 1, 1]

    # numpy inputs
    np.random.seed(15213)
    input_float32_np = np.random.uniform(low=conf.IMIN, high=conf.IMAX,
        size=input_shape).astype(np.float32)

    ## tf graph
    input_float32 = tf.placeholder(tf.float32, shape=input_shape, name='input')
    input_uint8, _, _ = tf.quantize(input_float32,
        min_range=conf.IMIN, max_range=conf.IMAX, T=tf.quint8,
        name='%s/input_uint8' % conf.netName)

    # branch 1
    b1_filter1_float32_np = np.random.uniform(low=conf.WMIN, high=conf.WMAX,
        size=b1_filter1_shape).astype(np.float32)
    b1_filter1_uint8_np, _ = quantize_np(b1_filter1_float32_np,
        range_min=conf.WMIN, range_max=conf.WMAX)
    b1_filter1_uint8 = tf.constant(b1_filter1_uint8_np,
        name='%s/b1_filter1' % conf.netName, dtype=tf.quint8)
    b1_bias1_float32_np = np.random.uniform(low=conf.AMIN, high=conf.AMAX,
        size=conf.CHID).astype(np.float32)
    b1_conv1_int32, _, _ = tf.nn.quantized_conv2d(
        input_uint8, b1_filter1_uint8,
        min_input=conf.IMIN, max_input=conf.IMAX,
        min_filter=conf.WMIN, max_filter=conf.WMAX,
        strides=[1, conf.S, conf.S, 1], padding='SAME', out_type=tf.qint32,
        name='%s/b1_conv1_int32' % conf.netName)
    min_output1_int32, max_output1_int32 = min_max_output_int32(
        range_input=conf.IMAX-conf.IMIN, range_filter=conf.WMAX-conf.WMIN)
    b1_conv1_float32 = tf.dequantize(b1_conv1_int32,
        min_range=min_output1_int32, max_range=max_output1_int32,
        name='%s/b1_dequantize1_float32' % conf.netName, mode='MIN_FIRST')
    b1_conv1_float32 = tf.nn.bias_add(b1_conv1_float32, bias=b1_bias1_float32_np,
        name='%s/b1_biasadd1_float32' % conf.netName)
    b1_conv1_float32 = tf.nn.relu(b1_conv1_float32,
        name='%s/b1_relu1_float32' % conf.netName)
    b1_conv1_uint8, _, _ = tf.quantize(b1_conv1_float32,
        min_range=conf.RQMIN, max_range=conf.RQMAX, T=tf.quint8,
        name='%s/b1_quantize1_uint8' % conf.netName)

    b1_filter2_float32_np = np.random.uniform(low=conf.WMIN, high=conf.WMAX,
        size=b1_filter2_shape).astype(np.float32)
    b1_filter2_uint8_np, _ = quantize_np(b1_filter2_float32_np,
        range_min=conf.WMIN, range_max=conf.WMAX)
    b1_filter2_uint8 = tf.constant(b1_filter2_uint8_np,
        name='%s/b1_filter2' % conf.netName, dtype=tf.quint8)
    b1_bias2_float32_np = np.random.uniform(low=conf.AMIN, high=conf.AMAX,
        size=conf.CHID).astype(np.float32)
    b1_conv2_int32, _, _ = tf.nn.quantized_conv2d(
        b1_conv1_uint8, b1_filter2_uint8,
        min_input=conf.RQMIN, max_input=conf.RQMAX,
        min_filter=conf.WMIN, max_filter=conf.WMAX,
        strides=strides, padding='SAME', out_type=tf.qint32,
        name='%s/b1_conv2_int32' % conf.netName)
    min_output2_int32, max_output2_int32 = min_max_output_int32(
        range_input=conf.RQMAX-conf.RQMIN, range_filter=conf.WMAX-conf.WMIN)
    b1_conv2_float32 = tf.dequantize(b1_conv2_int32,
        min_range=min_output2_int32, max_range=max_output2_int32,
        name='%s/b1_dequantize2_float32' % conf.netName, mode='MIN_FIRST')
    b1_conv2_float32 = tf.nn.bias_add(b1_conv2_float32, bias=b1_bias2_float32_np,
        name='%s/b1_biasadd2_float32' % conf.netName)
    b1_conv2_float32 = tf.nn.relu(b1_conv2_float32,
        name='%s/b1_relu2_float32' % conf.netName)
    b1_conv2_uint8, _, _ = tf.quantize(b1_conv2_float32,
        min_range=conf.RQMIN, max_range=conf.RQMAX, T=tf.quint8,
        name='%s/b1_quantize2_uint8' % conf.netName)

    b1_filter3_float32_np = np.random.uniform(low=conf.WMIN, high=conf.WMAX,
        size=b1_filter3_shape).astype(np.float32)
    b1_filter3_uint8_np, _ = quantize_np(b1_filter3_float32_np,
        range_min=conf.WMIN, range_max=conf.WMAX)
    b1_filter3_uint8 = tf.constant(b1_filter3_uint8_np,
        name='%s/b1_filter3' % conf.netName, dtype=tf.quint8)
    b1_bias3_float32_np = np.random.uniform(low=conf.AMIN, high=conf.AMAX,
        size=conf.COUT).astype(np.float32)
    b1_conv3_int32, _, _ = tf.nn.quantized_conv2d(
        b1_conv2_uint8, b1_filter3_uint8,
        min_input=conf.RQMIN, max_input=conf.RQMAX,
        min_filter=conf.WMIN, max_filter=conf.WMAX,
        strides=strides, padding='SAME', out_type=tf.qint32,
        name='%s/b1_conv3_int32' % conf.netName)
    min_output3_int32, max_output3_int32 = min_max_output_int32(
        range_input=conf.RQMAX-conf.RQMIN, range_filter=conf.WMAX-conf.WMIN)
    b1_conv3_float32 = tf.dequantize(b1_conv3_int32,
        min_range=min_output3_int32, max_range=max_output3_int32,
        name='%s/b1_dequantize3_float32' % conf.netName, mode='MIN_FIRST')
    b1_result_float32 = tf.nn.bias_add(b1_conv3_float32, bias=b1_bias3_float32_np,
        name='%s/b1_biasadd3_float32' % conf.netName)

    # branch 2
    if 'CONVBRANCH' in conf.__dict__:
        b2_filter1_float32_np = np.random.uniform(low=conf.WMIN, high=conf.WMAX,
            size=b2_filter1_shape).astype(np.float32)
        b2_filter1_uint8_np, _ = quantize_np(b2_filter1_float32_np,
            range_min=conf.WMIN, range_max=conf.WMAX)
        b2_filter1_uint8 = tf.constant(b2_filter1_uint8_np,
            name='%s/b2_filter' % conf.netName, dtype=tf.quint8)
        b2_bias1_float32_np = np.random.uniform(low=conf.AMIN, high=conf.AMAX,
            size=conf.COUT).astype(np.float32)
        b2_conv1_int32, _, _ = tf.nn.quantized_conv2d(
            input_uint8, b2_filter1_uint8,
            min_input=conf.IMIN, max_input=conf.IMAX,
            min_filter=conf.WMIN, max_filter=conf.WMAX,
            strides=[1, conf.S, conf.S, 1], padding='SAME', out_type=tf.qint32,
            name='%s/b2_quantized_conv2d' % conf.netName)
        b2_conv1_float32 = tf.dequantize(b2_conv1_int32,
            min_range=min_output1_int32, max_range=max_output1_int32,
            name='%s/b2_dequantize_int32_float32' % conf.netName, mode='MIN_FIRST')
        b2_result_float32 = tf.nn.bias_add(b2_conv1_float32, bias=b2_bias1_float32_np,
            name='%s/b2_biasadd_float32' % conf.netName)
    else:
        b2_result_float32 = input_float32

    # resadd
    result_float32 = tf.add(b1_result_float32, b2_result_float32,
        name='%s/resadd' % conf.netName)

    if 'HASRELU' in conf.__dict__:
        result_float32 = tf.nn.relu(result_float32,
            name='%s/relu_float32'%conf.netName)
    if 'HASAVGPOOL' in conf.__dict__:
        result_float32 = tf.nn.avg_pool(result_float32,
            ksize=[1, 7, 7, 1],
            strides=[1, 7, 7, 1],
            padding='VALID',
            name='%s/pool_float32'%conf.netName)

    output = tf.identity(result_float32, name='%s/output' % conf.netName)

    conf.gen_graph(output, input_data=input_float32_np, need_freezing=False)
