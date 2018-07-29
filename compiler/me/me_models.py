"""
Copyright 2018, Amazon.com, Inc. or its affiliates. All Rights Reserved
"""

"""Properties and methods for each of the engines within TPB
"""

import numpy as np
from skimage.util.shape import view_as_windows

""" PE Array properties and methods
"""
class PEArray:

    NUM_ROWS = 128
    NUM_COLS = 64
    PSUM_NUM_BANKS = 4
    MAX_WAVE_SIZE = 256
    num_wave_fp16_mm = 0
    num_of_ops_executed = 0
    total_pearray_latency_cycles = 0
    total_pearray_wave_elems = 0

    def __init__(self):
        self.psum_buf = np.zeros((self.PSUM_NUM_BANKS, self.MAX_WAVE_SIZE, self.NUM_COLS), dtype=np.float32)
        self.Tn = 0
        self.last_psum_bank_used = 0

    def extract_psum (self, psum_bank, start_entry, num_entries):
        assert(start_entry < self.MAX_WAVE_SIZE)
        #assert((start_entry+num_entries) < self.MAX_WAVE_SIZE)
        return self.psum_buf[psum_bank, start_entry:start_entry+num_entries, :]

    def write_psum (self, psum_bank, start_entry, num_entries, psum_temp):
        assert(start_entry < self.MAX_WAVE_SIZE)
        #assert((start_entry+num_entries) < self.MAX_WAVE_SIZE)
        self.psum_buf[psum_bank, start_entry:start_entry+num_entries, :] = psum_temp

    # Do wave fp16->fp32 matrix-multiply        
    #   packed_ifmaps: must be 256x128 matrix, float16
    #   packet_weights: must be 128x64 matrix, float16
    #   psum_bank: the PSUM bank number to write result to
    #   psum_add: if True, add to PSUM value in buffer; if False, replace with new value
    def wave_fp16_mm(self, packed_ifmaps, packet_weights, psum_bank, psum_add):
        assert (packed_ifmaps.shape == (self.MAX_WAVE_SIZE, self.NUM_ROWS))
        assert (packet_weights.shape == (self.NUM_ROWS, self.NUM_COLS))
        assert (psum_bank < self.PSUM_NUM_BANKS)
        a = packed_ifmaps.astype(np.float32)
        b = packet_weights.astype(np.float32)
        #print(a.dtype, b.dtype, self.psum_buf.dtype)
        self.matmul_result = np.matmul(a, b)
        if (psum_add):
            self.psum_buf[psum_bank] += self.matmul_result
        else:            
            self.psum_buf[psum_bank] = self.matmul_result
        self.num_wave_fp16_mm += 1

"""Pooling properties and methods
"""
class Pool:

    def resadd(self, array_a, array_b):
        return array_a + array_b

    def multiply(self, array_a, array_b):
        return array_a * array_b

    def pool(self, type, in_array, stride, pool_window_size, Tn, ifmap_tilex_sz, ifmap_tiley_sz, ofmap_tilex_sz, ofmap_tiley_sz):
        num_cols = in_array.shape[1]
        # view_as_windows needs in_array to be in the same dimension as window_shape
        # need to make sure the third dimension of stride_shape to be '1' since that is the column direction
        #print("ifmap_tilex_sz ", ifmap_tilex_sz, " ifmap_tiley_sz ", ifmap_tiley_sz)
        input_tilex_with_pad = ofmap_tilex_sz * stride + pool_window_size - stride
        input_tiley_with_pad = ofmap_tiley_sz * stride + pool_window_size - stride
        input_tile_with_pad_sz = input_tilex_with_pad*input_tiley_with_pad
        tile_array = np.empty((input_tiley_with_pad, input_tilex_with_pad))
        tile_array[:] = -np.inf  # set all padding values to -inf to allow only actual tile values to be analyzed
        ifmap_tile_sz = ifmap_tilex_sz*ifmap_tiley_sz
        ofmap_tile_sz = ofmap_tilex_sz*ofmap_tiley_sz
        pool_result = np.zeros((ofmap_tile_sz * Tn, num_cols))
        for j in range(Tn):
            for i in range(num_cols):
                tile_array[0:ifmap_tiley_sz, 0:ifmap_tilex_sz] = in_array[j*ifmap_tile_sz : (j+1)*ifmap_tile_sz, i].reshape(ifmap_tiley_sz, ifmap_tilex_sz) # ignoring Tn for now
                window_shape = (pool_window_size, pool_window_size)
                stride_shape = (stride, stride)
                pool_result_temp = view_as_windows(tile_array, window_shape, stride_shape)
                if (type == "MaxPool"):
                    pool_result[j*ofmap_tile_sz : (j+1)*ofmap_tile_sz, i] = pool_result_temp.max(axis=(2,3)).reshape(-1)
                elif (type == "AvgPool"):                    
                    pool_result[j*ofmap_tile_sz : (j+1)*ofmap_tile_sz, i] = pool_result_temp.mean(axis=(2,3)).reshape(-1)
                else:                    
                    print("ERROR: unknown type %s Pool.pool"%type)
                    exit(-1)
        return pool_result

    def reciprocate(self, in_array, num_active_channels):
        reciprocate_result = np.zeros(in_array.shape)
        reciprocate_result[:, 0:num_active_channels] = 1/in_array[:, 0:num_active_channels]
        return reciprocate_result

    def scale(self, in_array, scale_value):
        return in_array*scale_value

"""Bias-Add and Activate properties and methods
"""
class BiasAddAct:

    def biasadd(self, in_array, bias_array):
        return in_array + bias_array

    def act(self, type, in_array):
        if (type == 'Relu'):
            return self.__relu(in_array)
        elif (type == 'Sigmoid'):
            return 1/(1 + np.exp(-in_array))
        elif (type == 'Exp'):
            return np.exp(in_array)
        elif (type == 'Tanh'):
            return np.tanh(in_array)
        else:
            print("ERROR BiasAddAct.act: unrecognized activation type %s"%type)
            exit(-1)

    def __relu(self, in_array):
        return np.maximum(np.zeros(in_array.shape, dtype=in_array.dtype), in_array)

