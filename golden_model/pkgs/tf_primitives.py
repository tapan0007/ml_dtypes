#!/usr/bin/python 
"""

"""
import sys
import numpy as np
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3' #supress warnings
import tensorflow as tf

# transpose from NCHW to NHWC
def _i_mx_to_tf(i):
    return np.transpose(i, (0,2,3,1))

# transpose from NHWC to NCHW
def _i_tf_to_mx(i):
    return np.transpose(i, (0,3,1,2))

# transpose from HWCN to NHWC
def _f_mx_to_tf(f):
    return np.transpose(f, (2,3,1,0))

# get min/max range of dtype
def _np_min_max(dtype):
    if issubclass(np.dtype(dtype).type, np.integer):
        return (np.iinfo(np.dtype(dtype)).min, np.iinfo(np.dtype(dtype)).max)
    else:
        return (np.finfo(np.dtype(dtype)).min, np.finfo(np.dtype(dtype)).max)

# map from numpy type to tensorflow type
_np_to_tf_dtype = {
                 np.dtype(np.uint8) : tf.quint8,
                 np.dtype(np.float16) : tf.float16,
                 np.dtype(np.float32) : tf.float32 
                 }


######################################################
# Inputs:
#   i: 4D numpy ifmap in mxnet format (NCHW)
#   f: 4D numpy filter in mxnet format (NCHW)
#   strides: 4-entry list of convolve strides
#   padding: tbd
# Output: 4D numpy ofmap in mxnet format (NCHW)
######################################################
def convolve(i, f, stride, dilate, padding, mn_vs_tf = False):
    # TODO: make error reporting an option?
    itype_to_otype = {
                     tf.quint8  : tf.qint32,
                     tf.float16 : tf.float32,
                     tf.float32 : tf.float32
    }
    # get info from input images
    i_dtype = _np_to_tf_dtype[np.dtype(i.dtype)]
    o_dtype = itype_to_otype[i_dtype]

    # tensor flow strides/padding are 4D (NHWC), not 2D like mxnet
    stride = [1, stride[0], stride[1], 1]
    padding = [[0,0], [padding[0],padding[0]], [padding[1],padding[1]], [0,0]]

    # transpose to tf dims
    ii = _i_mx_to_tf(i)
    ff = _f_mx_to_tf(f)

    # create input to nn
    if i_dtype.is_quantized: 
        # FIXME, I can't case to tf_uint8, so I am evaluating to get padding back into numpy, so I can cast
        sess = tf.Session()
        ii_pad = tf.pad(ii, padding, 'CONSTANT')
        out = sess.run(ii_pad)
        iii = tf.constant(out, i_dtype, ii_pad.shape)
        fff = tf.constant(ff, i_dtype, ff.shape)
    else:
        iii = tf.cast(tf.pad(ii[:,::stride[1], ::stride[2], :], padding, 'CONSTANT'), o_dtype)
        fff = tf.constant(ff, o_dtype, ff.shape)

    # feed nn
    if i_dtype.is_quantized: 
        if dilate != [0, 0]:
            logging.warn("dilation is not suppored by quantized op in TF")
            return None
        (i_min, i_max) = _np_min_max(i.dtype)
        (f_min, f_max) = _np_min_max(f.dtype)
        tf_conv = tf.nn.quantized_conv2d(iii, fff,  out_type=o_dtype, strides=stride, padding='VALID', min_input=i_min, max_input=i_max, min_filter=f_min, max_filter=f_max)
    else:
        if dilate == [0,0]:
            tf_conv = tf.nn.conv2d(iii, fff, strides=stride, padding='VALID')
        else:
            # dilation w/ striding is not directly supported in tf, see: 
            # https://github.com/tensorflow/tensorflow/issues/2415
            # is a bit tricky to write an efficient implementation for general 
            # strides and rate based on the current approach. Note however 
            #that if rate is a multiple of stride, i.e., rate = k * stride for some positive k, then:
            if (dilate[1] + 1) % stride[1]:
                logging.warn("Tensorflow cannot calculate convolution when stride {} is not multiple of dilate +1 {} see https://github.com/tensorflow/tensorflow/issues/2415".format(stride[1], dilate[1] +1))
                return None 
            # rate = k, means k-1 pixels dilation, so adjust 
            k = (dilate[1] + 1)/stride[1]
            tf_conv = tf.nn.atrous_conv2d(iii, fff, rate=k, padding='VALID')

    # run
    sess = tf.Session()
    out = sess.run(tf_conv)

    # quantized returns tuple (out, min_output, max_output) - but regular conv2d just returns (out)
    if i_dtype.is_quantized: 
        out = out[0]

    return _i_tf_to_mx(out)

######################################################
# Private function called by max/avg pool
# Inputs:
#   ifmaps: 4D numpy ifmap in mxnet format (NCHW)
#   ksize: 4 entry size of kernel
#   strides: 4 entry list of strides
#   padding: tbd
# Output: 4D numpy ofmap in mxnet format (NCHW)
######################################################
def _pool(pool_function, ifmaps, ksize, strides=None, padding='VALID'):
    if strides is None:
        strides = ksize
    # get info from input images
    dtype = _np_to_tf_dtype[np.dtype(ifmaps.dtype)]

    # transpose to tf dims
    ii = _i_mx_to_tf(ifmaps)
    kk = (ksize[0], ksize[2], ksize[3], ksize[1])
    ss = (strides[0], strides[2], strides[3], strides[1])

    # create constants for input to nn
    iii = tf.constant(ii, dtype, ii.shape)

    # feed nn
    if dtype.is_quantized: 
        (i_min, i_max) = _np_min_max(ifmaps.dtype)
        tf_conv = pool_function(iii, min_input=i_min, max_input=i_max, ksize=kk, strides=ss, padding=padding)
    else:
        tf_conv = pool_function(value=iii, ksize=kk, strides=ss, padding=padding)

    # run
    sess = tf.Session()
    out = sess.run(tf_conv)

    # quantized returns tuple (out, min_output, max_output) - but regular conv2d just returns (out)
    if dtype.is_quantized: 
        out = out[0]

    return _i_tf_to_mx(out)

def max_pool(ifmaps, ksize, strides=None, padding='VALID'):
    if _np_to_tf_dtype[np.dtype(ifmaps.dtype)].is_quantized:
        pool_function = tf.nn.quantized_max_pool
    else:
        pool_function = tf.nn.max_pool
    return _pool(pool_function, ifmaps, ksize, strides, padding)

def avg_pool(ifmaps, ksize, strides=None, padding='VALID'):
    if _np_to_tf_dtype[np.dtype(ifmaps.dtype)].is_quantized:
        pool_function = tf.nn.quantized_avg_pool
    else:
        pool_function = tf.nn.avg_pool
    return _pool(pool_function, ifmaps, ksize, strides, padding)

def relu(ifmaps):
    dtype = _np_to_tf_dtype[np.dtype(ifmaps.dtype)]

    # transpose to tf dims
    ii = _i_mx_to_tf(ifmaps)

    # create constants for input to nn
    iii = tf.constant(ii, dtype, ii.shape)

    # feed nn
    if dtype.is_quantized: 
        (i_min, i_max) = _np_min_max(ifmaps.dtype)
        tf_conv = tf.nn.quantized_relu_x(iii, max_value=i_max, min_features=i_min, max_features=i_max)
    else:
        tf_conv = tf.nn.relu(iii)

    # run
    sess = tf.Session()
    out = sess.run(tf_conv)

    # quantized returns tuple (out, min_output, max_output) - but regular conv2d just returns (out)
    if dtype.is_quantized: 
        out = out[0]

    return _i_tf_to_mx(out)

def leakyrelu(ifmaps):
    dtype = _np_to_tf_dtype[np.dtype(ifmaps.dtype)]

    # transpose to tf dims
    ii = _i_mx_to_tf(ifmaps)

    # create constants for input to nn
    iii = tf.constant(ii, dtype, ii.shape)

    # feed nn
    if dtype.is_quantized: 
        # not supported yet
        return None
    else:
        tf_conv = tf.maximum(.01*iii,iii)

    # run
    sess = tf.Session()
    out = sess.run(tf_conv)

    # quantized returns tuple (out, min_output, max_output) - but regular conv2d just returns (out)
    if dtype.is_quantized: 
        out = out[0]

    return _i_tf_to_mx(out)

def tanh(ifmaps):
    dtype = _np_to_tf_dtype[np.dtype(ifmaps.dtype)]

    # transpose to tf dims
    ii = _i_mx_to_tf(ifmaps)

    # create constants for input to nn
    iii = tf.constant(ii, dtype, ii.shape)

    # feed nn
    if dtype.is_quantized: 
        # not supported yet
        return None
    else:
        tf_conv = tf.nn.tanh(iii)

    # run
    sess = tf.Session()
    out = sess.run(tf_conv)

    # quantized returns tuple (out, min_output, max_output) - but regular conv2d just returns (out)
    if dtype.is_quantized: 
        out = out[0]

    return _i_tf_to_mx(out)

def sigmoid(ifmaps):
    dtype = _np_to_tf_dtype[np.dtype(ifmaps.dtype)]

    # transpose to tf dims
    ii = _i_mx_to_tf(ifmaps)

    # create constants for input to nn
    iii = tf.constant(ii, dtype, ii.shape)

    # feed nn
    if dtype.is_quantized: 
        # not supported yet
        return None
    else:
        tf_conv = tf.nn.sigmoid(iii)

    # run
    sess = tf.Session()
    out = sess.run(tf_conv)

    # quantized returns tuple (out, min_output, max_output) - but regular conv2d just returns (out)
    if dtype.is_quantized: 
        out = out[0]

    return _i_tf_to_mx(out)

def fullyconnected(i, weight, bias, num_hidden):
    dtype = _np_to_tf_dtype[np.dtype(i.dtype)]
    i_flat_shape = reduce((lambda x,y: x*y), i.shape[1:])
    if bias == None:
        bias = np.zeros(num_hidden, dtype=i.dtype)

    # transpose to tf dims and create constants for input to nn
    # we are doing a matmul, so we do not need to convert formats w/ _i_mx_to_tf
    ii = np.reshape(i, [i.shape[0], i_flat_shape])
    ww = np.reshape(weight, [num_hidden, i_flat_shape])

    if dtype.is_quantized: 
        # not supported yet
        return None
    else:
        fc = tf.add(tf.matmul(ii, ww, transpose_b=True), bias)

    # run
    sess = tf.Session()
    out = sess.run(fc)

    return out
