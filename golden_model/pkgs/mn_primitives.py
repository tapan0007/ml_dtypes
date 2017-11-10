#!/usr/bin/python

import numpy as np
import math
from scipy.special import expit
import logging
import mpmath as mp
import sys
   
# sweeping configs - hacky?
mp.prec = 16
np.seterr(all='raise')
# Get on tensorflows page for testing, esp for type converstion
    
" The following _type* functions are helpers for all of the different  \
  input/output types for calculations.  They do some casting to/from \
  numpy/mpmath to hit targetted precisions (because numpy fp16 is not \
  IEEE compliant fp16, it internally upcasts to fp32). \
"
xnt32 = np.uint32

# dtype for array (mpf requires object type)
def _type_to_nparray(dtype):
    return {
        np.dtype('int8')  : np.int32,
        np.dtype('uint8') : xnt32,
        np.dtype('float16') :  object,
        np.dtype('float32') : np.float32
    }[dtype]

# manipulating value into type we want for calculation
def _type_cast(val):
    #  up_cast - "friendly" (non-numpy) upcast type that is applied to input before we 
    #   apply calculation mask (mpf does not recognize numpy types for casting).
    #  calc_cast - calculation for casting (e.g. conversino to mpf for float16)
    dtype = val.dtype
    (up_cast, calc_cast) = {
        np.dtype('int8') : (long, np.int32),
        np.dtype('uint8') : (long, xnt32),
        np.dtype('float16') : (float, mp.mpf),
        np.dtype('float32') : (float, np.float32)
    }[dtype]
    return calc_cast(up_cast(val))

# output_type required by API,,
def _type_to_otype(dtype):
    return {
        np.dtype('int8') : np.int32,
        np.dtype('uint8') : xnt32,
        np.dtype('float16') : np.float32,
        np.dtype('float32') : np.float32
    }[dtype]

###########################################
# ifmaps:  4D numpy array (#inputs,  #channels, #rows, #cols)
# filters: 4D numpy array (#filters, #channels, #rows, #cols)
# return: convolved 4D numpy array
###########################################
def convolve(ifmaps, filters, stride, dilate, padding, mn_vs_tf=False):
    if mn_vs_tf:
        global xnt32
        xnt32 = np.int32
    t = ifmaps.dtype

    # shape info
    (i_batches, i_channels, i_rows, i_cols) = ifmaps.shape
    (f_filters, f_channels, f_rows, f_cols) = filters.shape
    (s_rows, s_cols) = stride
    (d_rows, d_cols) = dilate
    (p_rows, p_cols) = padding
    (f_rows_dilated, f_cols_dilated) = (f_rows + (f_rows - 1) * d_rows,
                                        f_cols + (f_cols - 1) * d_cols)

    # for derivation of output dims, see https://arxiv.org/pdf/1603.07285.pdf
    # we use the dilated filter_rows/cols to compensate for the widened span 
    # of the filter
    o_rows = (i_rows - f_rows_dilated + 2 * p_rows) / s_rows + 1
    o_cols = (i_cols - f_cols_dilated + 2 * p_cols) / s_cols + 1

    # init output
    ofmap = np.zeros((i_batches, f_filters, o_rows, o_cols), 
            dtype=_type_to_nparray(t))

    # given an output pixel location, (i,j), calculate ifmap pixels 
    for batch in range(i_batches):
        for filtr in range(f_filters):
            for i in range(o_rows):
                for j in range(o_cols):
                    for m in range(f_rows):
                        for n in range(f_cols):
                            # systolic array sums across channels first
                            for channel in range(i_channels): 
                                # which ifmap pixel are we going to use to get ofmap pixel i,j?
                                #  i * srows :  stride offset
                                #  + m : pixel filter
                                #  - p_rows : slide left for account for padding offset from LHS
                                #  + m * d_rows : all of the above calcs are for contiguous ifmap,
                                #                 dilate the contiguous calc by offseting from 
                                #                 filter location * dilation stride
                                ii = (i*s_rows + m - p_rows) + m * d_rows
                                jj = (j*s_cols + n - p_cols) + n * d_cols
                                # stay in bounds of ifmap
                                if (ii >= 0) and (jj >= 0) and (ii < i_rows) and (jj  < i_cols):
                                    # FIXME - catching exception makes the operation incomplete, so we have to replay it
                                    try:
                                        # arithmetic error could come from multipliation or addition, so seperate into two ops
                                        rhs = _type_cast(ifmaps[batch,channel,ii,jj]) * _type_cast(filters[filtr,channel,m,n])
                                        ofmap[batch,filtr,i,j] += rhs
                                    except FloatingPointError as err:
                                        logging.warn("Convolve Error: {}, in batch={} filter={}, ofmap_i={}, ofmap_j={}".format(err, batch, filtr, i, j))
                                        with np.errstate(all='ignore'):
                                            ofmap[batch,filtr,i,j] +=  _type_cast(ifmaps[batch,channel,ii,jj]) * _type_cast(filters[filtr,channel,m,n])
    return np.array(ofmap, dtype=_type_to_otype(t))

############################################
# ifmaps:  4D numpy array (#inputs,  #channels, #rows, #cols)
# ksize: 4-tuple, pool dimensions
# strides: 4-tuple, stride dimensions
#############################################
def max_pool(ifmaps, ksize, strides):
    (i_batches, i_channels, i_rows, i_cols) = ifmaps.shape
    (k_batches, k_channels, k_rows, k_cols) = ksize
    (s_batches, s_channels, s_rows, s_cols) = strides
    (o_batches, o_channels, o_rows, o_cols) = tuple([(i-k)/s + 1 for (i,k,s) in zip(ifmaps.shape, ksize, strides)])
    dtype = ifmaps.dtype
    ofmap = np.zeros((o_batches, o_channels, o_rows, o_cols),
            dtype=dtype)
    for batch in range(o_batches):
        for channel in range(o_channels):
            for i in range(o_rows):
                for j in range(o_cols):
                    ofmap[batch, channel, i, j] = np.amax(ifmaps[
                                s_batches*batch:s_batches*batch + k_batches,
                                s_channels*channel:s_channels*channel + k_channels,
                                s_rows*i:s_rows*i + k_rows,
                                s_cols*j:s_cols*j + k_cols])
    
    return ofmap
    
############################################
# ifmaps:  4D numpy array (#inputs,  #channels, #rows, #cols)
# ksize: 4-tuple, pool dimensions
# strides: 4-tuple, stride dimensions
#############################################
def avg_pool(ifmaps, ksize, strides):
    (i_batches, i_channels, i_rows, i_cols) = ifmaps.shape
    (k_batches, k_channels, k_rows, k_cols) = ksize
    (o_batches, o_channels, o_rows, o_cols) = tuple([(i-k)/s + 1 for (i,k,s) in zip(ifmaps.shape, ksize, strides)])
    (s_batches, s_channels, s_rows, s_cols) = strides
    dtype = ifmaps.dtype
    ofmap = np.zeros((o_batches, o_channels, o_rows, o_cols), dtype=dtype)
    for batch in range(o_batches):
        for channel in range(o_channels):
            for i in range(o_rows):
                for j in range(o_cols):
                    try:
                        ofmap[batch, channel, i, j] = np.mean(ifmaps[
                                s_batches*batch:s_batches*batch + k_batches,
                                s_channels*channel:s_channels*channel + k_channels,
                                s_rows*i:s_rows*i + k_rows,
                                s_cols*j:s_cols*j + k_cols])
                    except FloatingPointError as err:
                        logging.warning("AvgPool Error: {}, at batch={} filter={}, ofmap_i={}, ofmap_j={}".format(
                                    err, s_batches*batch, s_channels*channel, s_rows*i, s_cols*j))
                        with np.errstate(all='ignore'): #redo with op off
                            l = ifmaps[
                                s_batches*batch:s_batches*batch + k_batches,
                                s_channels*channel:s_channels*channel + k_channels,
                                s_rows*i:s_rows*i + k_rows,
                                s_cols*j:s_cols*j + k_cols].flatten()
                            if isinstance(dtype, mpf):
                                s = mpf.fsum(l)/len(l)
                            else:
                                ofmap[batch, channel, i, j] = np.mean(l)
    
    return ofmap
            
############################################
# helper function for activations
# ifmaps:  4D numpy array (#inputs,  #channels, #rows, #cols)
#############################################
def _activation(f, ifmaps):
    (o_batches, o_channels, o_rows, o_cols) = ifmaps.shape
    dtype = ifmaps.dtype
    ofmap = np.zeros((o_batches, o_channels, o_rows, o_cols), dtype=dtype)
    for batch in range(o_batches):
        for channel in range(o_channels):
            for i in range(o_rows):
                for j in range(o_cols):
                    ofmap[batch, channel, i, j] = f(ifmaps[batch, channel, i,j])
    return ofmap

############################################
# ifmaps:  4D numpy array (#inputs,  #channels, #rows, #cols)
#############################################
def relu(ifmaps):
    return _activation(lambda x:max([x,0]), ifmaps)

            
############################################
# ifmaps:  4D numpy array (#inputs,  #channels, #rows, #cols)
#############################################
def leakyrelu(ifmaps):
    return _activation(lambda x:max([x,0.01*x]), ifmaps)

############################################
# ifmaps:  4D numpy array (#inputs,  #channels, #rows, #cols)
#############################################
def tanh(ifmaps):
    return _activation(math.tanh, ifmaps)

############################################
# ifmaps:  4D numpy array (#inputs,  #channels, #rows, #cols)
#############################################
def sigmoid(ifmaps):
    return _activation(lambda x:1/(1+math.exp(-x)), ifmaps)


############################################
# ifmaps:  4D numpy array (#batches,  #channels, #rows, #cols)
# weight:  2D numpy array (#hidden, #inputs-flattened)
# bias:    1D numpy array (# hidden)
# num_hidden:  int (# of hidden nodes in hidden layer)
#############################################
def fullyconnected(i, weight, bias, num_hidden):
    (num_batches, i_shape) = (i.shape[0], i.shape[1:])
    i_flat_shape = reduce((lambda x,y: x*y), i_shape)
    dtype = i.dtype

    if bias == None:
        bias = np.zeros(num_hidden, dtype=dtype)
    
    ii = np.reshape(i, [i.shape[0], i_flat_shape])

    o = np.zeros((num_batches, num_hidden), 
            dtype=_type_to_nparray(dtype))
    for batch in range(num_batches):
        for h in range(num_hidden):
            for i in range(i_flat_shape):
                rhs = _type_cast(weight[h, i]) * _type_cast(ii[batch,i]) + _type_cast(bias[h])
                try:
                    o[batch, h] += rhs
                except FloatingPointError as err:
                    logging.warning("Fully Connected: {}, in batch={} hidden={}, i={}".format(err, h, i))
                    with np.errstate(all='ignore'):
                        o[batch, h] += _type_cast(weight[h, i]) * _type_cast(ii[batch,i]) + _type_cast(bias[h])
    return np.array(o, dtype=_type_to_otype(dtype))

